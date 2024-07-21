import torch
import torch.nn as nn
from usfutils.load import instantiate_from_config
from torch_geometric.data import Data, Batch
import numpy as np
from mcd.src.mlp import MLP
from mcd.src.text_graph import TextGraphClf
from mcd.src.moe_all import MoE
from mcd.src.constants import VERY_SMALL_NUMBER
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, GCN

class MCDModel(nn.Module):
    def __init__(
        self,
        config,
        cls_dim=128 * 3,
        feature_dim=1024,
        feature_hidden_dim=128,
        num_trans_heads=2,
        dropout=0.1,
    ):
        super().__init__()
        self.config = config
        # video content
        config1 = config.v_moe
        cls_dim = config1.cls_dim * 3
        feature_dim = config1.feature_dim
        feature_hidden_dim = config1.feature_hidden_dim
        num_trans_heads = config1.num_trans_heads
        dropout = config1.dropout
        self.linear_video = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_title = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_author = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_asr = nn.Sequential(
            nn.Linear(feature_dim, feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_comment = nn.Sequential(
            torch.nn.Linear(feature_dim, feature_hidden_dim),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, int(cls_dim / 2)),
            nn.LayerNorm(int(cls_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(int(cls_dim / 2), 2),
        )
        self.transformer = nn.TransformerEncoderLayer(
            d_model=feature_hidden_dim * config1.num_feature, nhead=num_trans_heads, batch_first=True
        )
        self.moe = MoE(
            input_size=config1.moe_in_feature * config1.num_feature,
            hidden_size=config1.moe_hidden_feature,
            output_size=config1.moe_out_feature,
            num_experts=config1.num_experts,
            model=MLP,
            k=config1.k,
        )
        # video-comment gcn
        config2 = config.vc_gcn
        self.gnn1 = GCN(in_channels=config2['in_feature'], hidden_channels=config2['hidden_feature'], num_layers=1)
        self.gnn2 = GCN(in_channels=config2['hidden_feature'], hidden_channels=config2['out_feature'], num_layers=1)
        # comments gcn
        self.comments_gcn = TextGraphClf(config.comments_gcn)

    def forward(self, **kwargs):
        video_fea = kwargs["video_feas"]
        title_fea = kwargs["title_feas"]
        author_fea = kwargs["author_feas"]
        asr_fea = kwargs["asr_feas"]
        title_fea = self.linear_title(title_fea)  
        video_fea = self.linear_video(video_fea)  
        asr_fea = self.linear_asr(asr_fea)  
        video_fea = torch.mean(video_fea, -2)
        video_feature = video_fea
        author_fea = self.linear_author(author_fea)  
        fea = torch.cat((video_fea, title_fea, asr_fea, author_fea), 1) 
        fea = self.transformer(fea)
        fea, moe_loss = self.moe(fea)
        # vc_gcn
        comments_feature = kwargs["comment_feas"]
        comments_feature = self.linear_comment(comments_feature)  
        num_comms = comments_feature.shape[1]
        graph_list = []
        for idx in range(comments_feature.shape[0]):
            sub_comment_fea = comments_feature[idx]
            sub_title_fea = video_feature[idx].unsqueeze(0)
            x = torch.cat((sub_title_fea, sub_comment_fea), dim=0)
            edge_list = []
            edge_list += [[0, i] for i in range(1, num_comms + 1)]
            edge_list += [[i, 0] for i in range(1, num_comms + 1)]
            edge_list += [[i, i] for i in range(x.shape[0])]
            edge_pair = torch.tensor(edge_list, dtype=torch.long)
            edge_index = edge_pair.t().contiguous()
            graph_list.append(Data(x=x, edge_index=edge_index))
        graph_batch = Batch.from_data_list(graph_list).to(kwargs["device"])
        x, edge_index, batch = (
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.batch,
        )
        x = F.dropout(F.relu(self.gnn1(x, edge_index)), 0.35)
        x = self.gnn2(x, edge_index)
        x = global_max_pool(x, batch)
        # comments_gcn
        network = self.comments_gcn
        network.train()
        is_test = False
        context, context_lens, sen_adj = (
            kwargs["comment_feas"],
            kwargs["comment_lens"],
            kwargs["sen_adj"],
        )
        raw_context_vec, context_vec, context_mask = network.prepare_init_graph(
            context, context_lens
        )
        # Init
        raw_node_vec = raw_context_vec
        init_node_vec = context_vec
        node_mask = context_mask
        cur_raw_adj, cur_sen_adj = network.learn_graph(
            network.graph_learner,
            raw_node_vec,
            graph_skip_conn=network.graph_skip_conn,
            node_mask=node_mask,
            init_adj=sen_adj,
        )
        # Add mid GNN layers
        t = 1
        for encoder in network.encoder.graph_encoders[1:-1]:  # none
            if t % 2 == 1:
                node_vec = torch.relu(encoder(init_node_vec, cur_sen_adj))
                node_vec = F.dropout(
                    node_vec, network.dropout, training=network.training
                )

            else:
                node_vec = torch.relu(encoder(node_vec))
                node_vec = F.dropout(
                    node_vec, network.dropout, training=network.training
                )

            t = t + 1
        # BP to update weights
        if t % 2 == 1:
            output = network.encoder.graph_encoders[-1](init_node_vec, cur_sen_adj)
        else:
            output = network.encoder.graph_encoders[-1](node_vec)
        output = network.compute_output(
            output, context_vec=init_node_vec, test=is_test, node_mask=node_mask
        )
        loss = self.add_batch_graph_loss(cur_raw_adj, raw_node_vec)  
        fea = torch.cat([fea, x, output], dim=-1) 
        output = self.classifier(fea)
        return output, moe_loss + loss

    def add_batch_graph_loss(self, out_adj, features, keep_batch_dim=False):
        # Graph regularization
        if keep_batch_dim:
            graph_loss = []
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss.append(
                    self.config["smoothness_ratio"]
                    * torch.trace(
                        torch.mm(
                            features[i].transpose(-1, -2), torch.mm(L, features[i])
                        )
                    )
                    / int(np.prod(out_adj.shape[1:]))
                )

            graph_loss = torch.Tensor(graph_loss).cuda()

            ones_vec = torch.ones(out_adj.shape[:-1]).cuda()
            graph_loss += (
                -self.config["degree_ratio"]
                * torch.matmul(
                    ones_vec.unsqueeze(1),
                    torch.log(
                        torch.matmul(out_adj, ones_vec.unsqueeze(-1))
                        + VERY_SMALL_NUMBER
                    ),
                )
                .squeeze(-1)
                .squeeze(-1)
                / out_adj.shape[-1]
            )
            graph_loss += (
                self.config["sparsity_ratio"]
                * torch.sum(torch.pow(out_adj, 2), (1, 2))
                / int(np.prod(out_adj.shape[1:]))
            )

        else:
            graph_loss = 0
            for i in range(out_adj.shape[0]):
                L = torch.diagflat(torch.sum(out_adj[i], -1)) - out_adj[i]
                graph_loss += (
                    self.config["smoothness_ratio"]
                    * torch.trace(
                        torch.mm(
                            features[i].transpose(-1, -2), torch.mm(L, features[i])
                        )
                    )
                    / int(np.prod(out_adj.shape))
                )

            ones_vec = torch.ones(out_adj.shape[:-1]).cuda()
            graph_loss += (
                -self.config["degree_ratio"]
                * torch.matmul(
                    ones_vec.unsqueeze(1),
                    torch.log(
                        torch.matmul(out_adj, ones_vec.unsqueeze(-1))
                        + VERY_SMALL_NUMBER
                    ),
                ).sum()
                / out_adj.shape[0]
                / out_adj.shape[-1]
            )
            graph_loss += (
                self.config["sparsity_ratio"]
                * torch.sum(torch.pow(out_adj, 2))
                / int(np.prod(out_adj.shape))
            )
        return graph_loss
