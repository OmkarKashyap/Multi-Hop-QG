import torch
import torch.nn as nn


class MeanAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MeanAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, nodes, adjs, node_embeddings):
        # nodes: A tensor containing the IDs of the target nodes for aggregation
        # adjs: A tensor containing the adjacency matrix (binary adjacency matrix)
        # node_embeddings: A tensor containing the embeddings of all nodes in the graph

        # Extract the embeddings of neighboring nodes
        neighbor_embeddings = torch.matmul(adjs, node_embeddings)

        # Calculate the degree of each node (number of neighbors)
        degree = torch.sum(adjs, dim=1, keepdim=True)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-7
        degree = degree + epsilon

        # Calculate the mean aggregation
        aggregated_embeddings = torch.div(neighbor_embeddings, degree)

        return aggregated_embeddings
    
class MeanPoolingAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MeanPoolingAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, nodes, adjs, node_embeddings):
        # nodes: A tensor containing the IDs of the target nodes for aggregation
        # adjs: A tensor containing the adjacency matrix (binary adjacency matrix)
        # node_embeddings: A tensor containing the embeddings of all nodes in the graph

        # Extract the embeddings of neighboring nodes
        neighbor_embeddings = torch.matmul(adjs, node_embeddings)

        # Calculate the degree of each node (number of neighbors)
        degree = torch.sum(adjs, dim=1, keepdim=True)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-7
        degree = degree + epsilon

        # Calculate the mean pooling aggregation
        aggregated_embeddings = torch.div(torch.sum(neighbor_embeddings, dim=1, keepdim=True), degree)

        return aggregated_embeddings


class LSTMAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTMAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # LSTM layer for aggregation
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)

    def forward(self, nodes, adjs, node_embeddings):
        # nodes: A tensor containing the IDs of the target nodes for aggregation
        # adjs: A tensor containing the adjacency matrix (binary adjacency matrix)
        # node_embeddings: A tensor containing the embeddings of all nodes in the graph

        # Extract the embeddings of neighboring nodes
        neighbor_embeddings = torch.matmul(adjs, node_embeddings)

        # Gather the embeddings of the target nodes based on their IDs
        target_embeddings = torch.gather(node_embeddings, dim=0, index=nodes.view(-1, 1).expand(-1, neighbor_embeddings.size(2)))

        # Concatenate target node embeddings with neighbor embeddings
        input_sequence = torch.cat([target_embeddings.unsqueeze(1), neighbor_embeddings], dim=1)

        # Pass the input sequence through the LSTM
        lstm_output, _ = self.lstm(input_sequence)

        # Take the last hidden state as the aggregation result
        aggregated_embeddings = lstm_output[:, -1, :]

        return aggregated_embeddings
