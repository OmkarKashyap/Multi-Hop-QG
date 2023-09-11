import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from graphSage_layers import SageLayer

from model_utils import cal_position_id_2



class GraphSAGEEncoder(nn.Module):
    def __init__(self,config):
        super(GraphSAGEEncoder, self).__init__()
        self.config=config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.emb_dim)

        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        input_dim = self.num_features

        for output_dim in self.hidden_dims:
            layers.append(SageLayer(input_dim, output_dim, self.aggregator))
            input_dim = output_dim

        return nn.ModuleList(layers)

    def forward(self, batch, nodes, adjs, features):
        
        answer_mapping = batch['ans_mask']
        entity_mask = batch['entity_mask']
        context_ids = batch['context_idxs']
        answer_ids = batch['ans_idxs']
        context_mask = batch['context_mask']
        position_ids = cal_position_id_2(context_ids, batch['y1'], batch['y2'], context_mask)
        src_len = batch['context_lens']

        if self.config.use_cuda:
            answer_mapping = answer_mapping.cuda()
            entity_mask = entity_mask.cuda()
            context_ids = context_ids.cuda()
            answer_ids = answer_ids.cuda()
            position_ids = position_ids.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            if self.position_embeddings is not None:
                self.position_embeddings = self.position_embeddings.cuda()
                
        x = features
        for layer in self.layers:
            x = layer(nodes, adjs, x)
        return x
