# import argparse
# import os
# import json
# from os.path import join




# def set_config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--vocab_size", type=int, default=45000)
#     parser.add_argument("--dropout", type=float, default=0.2)
#     parser.add_argument("--position_emb_size", type=int, default=3)
#     parser.add_argument("--max_position_embeddings", type=int, default=3)
#     parser.add_argument("--emb_dim", type=int, default=300)
#     parser.add_argument("--decoder_num_layers", type=int, default=2)
#     parser.add_argument("--num_heads", type=int, default=2)
#     parser.add_argument('--forward_expansion', type=int, default=0.5)
#     parser.add_argument("--hidden_dim", type=int, default=300)
    
#     args = parser.parse_args()

#     return args

from typing import Any


class Config():
    def __init__(self):
        self.vocab_size=45000
        self.dropout=0.2
        self.position_emb_size=3
        self.max_position_embeddings=3
        self.emb_dim=300
        self.decoder_num_layers=2
        self.num_head=2
        self.forward_expansion=0.5
        self.hidden_dim=300
