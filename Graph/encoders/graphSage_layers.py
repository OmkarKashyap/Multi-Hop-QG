import torch
import torch.nn as nn
import torch.nn.functional as F

class SageLayer(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, input_size, out_size, aggregator): 
		super(SageLayer, self).__init__()

		self.input_size = input_size
		self.out_size = out_size
		self.aggregator=aggregator

		self.weight = nn.Parameter(torch.FloatTensor(out_size, 2 * self.input_size))

		self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)
   
	def reset_parameters(self):
		nn.init.xavier_uniform_(self.weight)
		nn.init.zeros_(self.bias)
           

	def forward(self, nodes, adjs, features):
        # Sample neighbors, perform aggregation, and update node embeddings
		aggregated = self.aggregator(nodes, adjs, features)
		output = torch.matmul(aggregated, self.weight) + self.bias
		return F.relu(output)
    
	# def forward(self, self_feats, aggregate_feats, neighs=None):
	# 	"""
	# 	Generates embeddings for a batch of nodes.

	# 	nodes	 -- list of nodes
	# 	"""

	# 	combined = aggregate_feats
  	# 	aggregated = self.aggregator(nodes, adjs, features)
	# 	combined = F.relu(self.weight.mm(combined.t())).t()
	# 	return combined

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


        