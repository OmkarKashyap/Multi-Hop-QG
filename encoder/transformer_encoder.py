import torch
import torch.nn as nn
from layers import InputEmbeddings, TokenTypeEmbeddings, PositionalEncoding, MultiHeadAttentionBlock, FeedForwardBlock, Encoder, EncoderBlock
import torch
from torch.utils.data import DataLoader
import os
import sys
from layers import TransformerModel
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from data.dataset import QGDataset, dataloader




# Define model hyperparameters
d_model = 512  
num_heads = 8 
seq_len = 128  
vocab_size = 10000
num_token_types = 2  
batch_size = 32

dataset = QGDataset(root_dir="hotpot/data", max_seq_length=128)
model = TransformerModel(d_model, num_heads, seq_len, vocab_size, num_token_types)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=dataset.collate_fn
)

for batch in dataloader:
    input_ids = batch['input_ids']    
    token_type_ids = batch['token_type_ids']
    attention_mask = batch['attention_mask']
    print(input_ids.shape)
    # output = model(input_ids, token_type_ids, attention_mask)
    # print(output)
    
#--------------------

# Prepare sample data
sample_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
sample_input_ids = sample_input_ids.view(4,1,128)
sample_token_type_ids = torch.randint(0, num_token_types, (batch_size, seq_len))
sample_mask = torch.ones(batch_size, seq_len)

print(sample_input_ids.shape)

# Initialize and load the model
model = TransformerModel(d_model, num_heads, seq_len, vocab_size, num_token_types)

# Perform a forward pass
output = model(sample_input_ids, sample_token_type_ids, sample_mask)
print(output)
