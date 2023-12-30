import torch
import torch.nn as nn
from torch_scatter import scatter_max
import torch.nn.functional as F

INF = 1e12

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3

class RefinementNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(RefinementNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class PreliminaryDecoder(nn.Module):
    def __init__(
        self,
        config,
        embeddings
    ):
        super(PreliminaryDecoder, self).__init__()
        device = config.device
        self.trg_vocab_size = config.vocab_size
        embed_size = config.emb_dim
        dropout = config.dropout
        num_layers = config.decoder_num_layers
        heads = config.num_heads
        max_length = config.max_position_embeddings
        forward_expansion = config.forward_expansion

        if embeddings is not None:
            # Use pre-trained embeddings if provided
            self.word_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        else:
            self.word_embedding = nn.Embedding(self.trg_vocab_size, embed_size)

        

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        # self.fc_out = nn.Linear(embed_size, self.trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        # out = self.fc_out(x)

        return x

class RefinementDecoder(nn.Module):
    def __init__(self, config, embeddings):
        super(RefinementDecoder, self).__init__()
        self.vocab_size = config.vocab_size
        embedding_size = config.emb_dim
        hidden_size = config.hidden_dim * 2
        num_layers = config.decoder_num_layers
        dropout = config.dropout

        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        if embeddings is not None:
            self.embedding = nn.Embedding(self.vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=True)

        if num_layers == 1:
            dropout = 0.0
        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.linear_combined_input = nn.Linear(embedding_size*2 + hidden_size , embedding_size)
        self.reduce_layer = nn.Linear(embedding_size + hidden_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.logit_layer = nn.Linear(hidden_size, self.vocab_size)

    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        energy = energy.squeeze(1).masked_fill(mask, value=-1e12)
        attn_dist = F.softmax(energy, dim=1).unsqueeze(dim=1)  # [b, 1, t]
        context_vector = torch.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy

    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)

    def forward(self, y, ext_x, prev_states, prev_context, encoder_outputs, encoder_mask, decoder_out):
        # forward one step lstm
        # y : [b]

        embedded = self.embedding(y.unsqueeze(1))
        encoder_features = self.get_encoder_features(encoder_outputs)
        combined_input = torch.cat([embedded, encoder_features, decoder_out], dim=-1)
        lstm_inputs = self.reduce_layer(torch.cat([combined_input, prev_context], dim=2))
        output, states = self.lstm(lstm_inputs, prev_states)
        context, energy = self.attention(output, encoder_features, encoder_mask)
        concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)  # [b, |V|]

        if True:
            batch_size = y.size(0)
            num_oov = max(torch.max(ext_x - self.vocab_size + 1), 0)
            zeros = torch.zeros((batch_size, num_oov)).cuda()
            # , device=config.device)
            extended_logit = torch.cat([logit, zeros], dim=1)
            out = torch.zeros_like(extended_logit) - INF
            out, _ = scatter_max(energy, ext_x, out=out)
            out = out.masked_fill(out == -INF, 0)
            logit = extended_logit + out
            logit = logit.masked_fill(logit == -INF, 0)
            # forcing UNK prob 0
            logit[:, UNK_ID] = -INF

        return logit, states, context

class MainDecoder(nn.Module):
    def __init__(self, config, embeddings) -> None:
        super(MainDecoder).__init__()
        self.vocab_size = config.vocab_size
        embedding_size = config.emb_dim
        hidden_size = config.hidden_dim * 2
        dropout = config.dropout
        self.RefinedEncoder = RefinementNetwork(embedding_size,hidden_size,embedding_size,dropout)
        self.Preliminary = PreliminaryDecoder(config,embeddings)
        self.Refined = RefinementDecoder(config,embeddings)

    def forward(self,x,enc_out,src_mask, trg_mask, ext_x, prev_states, prev_context):
        refined_input = self.RefinedEncoder(enc_out)
        combined_input = torch.cat((refined_input, enc_out), dim=-1)
        prelim_out = self.Preliminary(x,combined_input,src_mask,trg_mask)
        logits,refined_out,context = self.Refined(x,ext_x, prev_states, prev_context, enc_out, src_mask, prelim_out)
        return logits



        