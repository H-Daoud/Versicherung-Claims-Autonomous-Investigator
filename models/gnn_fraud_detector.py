import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FraudGNN(torch.nn.Module):
    """
    A Graph Neural Network (GraphSAGE) to detect fraudulent claims 
    based on network relationships (e.g., shared phone numbers, repair shops).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FraudGNN, self).__init__()
        # GraphSAGE layer 1: Aggregates info from immediate neighbors
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        # GraphSAGE layer 2: Aggregates info from the neighbor's neighbors (2-hop)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        # Final classification layer
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # x = Node Features, edge_index = Graph connectivity
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        # Final prediction (Fraud vs. Legit)
        out = self.classifier(h)
        return F.log_softmax(out, dim=1)
