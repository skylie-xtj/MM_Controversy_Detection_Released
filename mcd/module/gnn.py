import torch.nn as nn
from torch_geometric.nn import GAT, GCN, GraphSAGE, GCNConv, global_max_pool
import torch.nn.functional as F


class MCDGnn(nn.Module):
    def __init__(self, in_feature_dim=1024, hidden_dim=128, num_layers=1):
        super().__init__()
        # self.num_classes = num_classes
        self.gat = GAT(in_channels=in_feature_dim, hidden_channels=hidden_dim, num_layers=num_layers)
        # self.gat = GraphSAGE(in_channels=in_feature_dim, hidden_channels=hidden_dim, num_layers=num_layers)
        # self.linear = nn.Linear(self.hdim, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gat(x, edge_index))
        x = global_max_pool(x, batch)
        # x = self.linear(x)
        return x

# class MCDGnn(nn.Module):
#     def __init__(self, in_feature_dim=1024, hidden_dim=128, num_layers=1):
#         super().__init__()
#         # 第一层 GCN
#         self.conv1 = GCNConv(in_channels=in_feature_dim, out_channels=512)
#         # 第二层 GCN
#         self.conv2 = GCNConv(in_channels=in_feature_dim, out_channels=hidden_dim)
#     def forward(self, x, edge_index):
#         # 第一层 GCN
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         # 第二层 GCN
#         x = self.conv2(x, edge_index)
#         return x