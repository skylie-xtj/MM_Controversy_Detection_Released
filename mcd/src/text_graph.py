import torch
import torch.nn as nn
import torch.nn.functional as F

from mcd.layers.graphlearn import GraphLearner
from mcd.layers.common import dropout, EncoderRNN
from mcd.layers.gnn import GCN, GAT
from mcd.utils.generic_utils import create_mask


class TextGraphClf(nn.Module):
    def __init__(self, config):  # , w_embedding, word_vocab):
        super(TextGraphClf, self).__init__()
        self.config = config
        self.name = "TextGraphClf"
        self.device = "cuda"

        # Shape
        word_embed_dim = config["word_embed_dim"]  # 1024
        hidden_size = config["hidden_size"]  # 128

        # Dropout
        self.dropout = config["dropout"]  # 0.5
        self.word_dropout = config.get("word_dropout", config["dropout"])  # 0.5
        self.rnn_dropout = config.get("rnn_dropout", config["dropout"])  # 0.5

        # Graph
        self.graph_learn = config["graph_learn"]  # True
        self.graph_metric_type = config["graph_metric_type"]  # 'attention'
        self.graph_module = config["graph_module"]  # gcn
        self.graph_skip_conn = config["graph_skip_conn"]  # 情感图的参数 0.1
        self.graph_skip_conn1 = config["graph_skip_conn1"]  # 依赖图的参数 0.05

        # Text
        # self.word_embed = w_embedding
        # if config["fix_vocab_embed"]:
        #     print("[ Fix word embeddings ]")
        #     for param in self.word_embed.parameters():
        #         param.requires_grad = False

        # self.lstm_encoder = BLSTM(
        #     input_dim=word_embed_dim,
        #     hidden_dim=hidden_size,
        #     num_layers=1,
        #     dropout=self.rnn_dropout,
        # )

        self.ctx_rnn_encoder = EncoderRNN(
            word_embed_dim,
            hidden_size,
            bidirectional=True,
            num_layers=1,
            rnn_type="lstm",
            rnn_dropout=self.rnn_dropout,
            device=self.device,
        )

        self.linear_out = nn.Linear(hidden_size, 2, bias=False)

        self.scalable_run = config.get("scalable_run", False)

        if not config.get("no_gnn"):
            print("[ Using TextGNN ]")

            if self.graph_module == "gcn":
                gcn_module = GCN
                self.encoder = gcn_module(
                    nfeat=hidden_size,
                    nhid=hidden_size,
                    nclass=hidden_size,
                    graph_hops=config.get("graph_hops", 2),
                    dropout=self.dropout,
                    batch_norm=config.get("batch_norm", False),
                )

            elif self.graph_module == "gat":
                gcn_module = GAT
                self.encoder = gcn_module(
                    nfeat=hidden_size,
                    nhid=hidden_size,
                    nclass=hidden_size,
                    dropout=self.dropout,
                    alpha=config.get("gat_alpha"),
                    nheads=config.get("gat_nhead"),
                )

            else:
                raise RuntimeError("Unknown graph_module: {}".format(self.graph_module))

            if self.graph_learn:
                graph_learn_fun = GraphLearner
                self.graph_learner = graph_learn_fun(
                    word_embed_dim,
                    config["graph_learn_hidden_size"],
                    topk=config["graph_learn_topk"],
                    epsilon=config["graph_learn_epsilon"],
                    num_pers=config["graph_learn_num_pers"],
                    metric_type=config["graph_metric_type"],
                    device=self.device,
                )

                self.graph_learner2 = graph_learn_fun(
                    hidden_size,
                    config.get(
                        "graph_learn_hidden_size2", config["graph_learn_hidden_size"]
                    ),
                    topk=config.get("graph_learn_topk2", config["graph_learn_topk"]),
                    epsilon=config.get(
                        "graph_learn_epsilon2", config["graph_learn_epsilon"]
                    ),
                    num_pers=config["graph_learn_num_pers"],
                    metric_type=config["graph_metric_type"],
                    device=self.device,
                )

                print("[ Graph Learner ]")

                if config["graph_learn_regularization"]:
                    print("[ Graph Regularization]")
            else:
                self.graph_learner = None
                self.graph_learner2 = None

        else:
            print("[ Using RNN ]")

    def compute_no_gnn_output(self, raw_context_vec, context_lens):
        # raw_context_vec = self.word_embed(context)
        raw_context_vec = dropout(
            raw_context_vec, self.word_dropout, shared_axes=[-2], training=self.training
        )

        # Shape: [batch_size, hidden_size]
        # context_vec = self.lstm_encoder(raw_context_vec, context_lens)[1][0].squeeze(0)
        context_vec = self.ctx_rnn_encoder(raw_context_vec, context_lens)[1][0].squeeze(
            0
        )
        output = self.linear_out(context_vec)
        output = F.log_softmax(output, dim=-1)
        return output

    def learn_graph(
        self,
        graph_learner,
        node_features,
        graph_skip_conn=None,
        node_mask=None,
        anchor_mask=None,
        init_adj=None,
        anchor_features=None,
    ):
        if self.graph_learn:
            if self.scalable_run:
                node_anchor_adj = graph_learner(
                    node_features, anchor_features, node_mask, anchor_mask
                )
                return node_anchor_adj

            else:  # this way
                raw_adj = graph_learner(node_features, node_mask)  # attention score
                adj = torch.softmax(raw_adj, dim=-1)
                init_adj = torch.softmax(init_adj, dim=-1)
                adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

                return raw_adj, adj

        else:
            raw_adj = None
            adj = init_adj

            return raw_adj, adj

    def compute_output(self, node_vec, context_vec, test, node_mask=None):

        vec = self.graph_maxpool(node_vec.transpose(-1, -2), node_mask=node_mask)

        output = vec
        # output = self.linear_out(vec)

        # output = F.log_softmax(output, dim=-1)
        return output

    def prepare_init_graph(self, raw_context_vec, context_lens):
        context_mask = create_mask(
            context_lens, raw_context_vec.size(-2), device=self.device
        )
        # Shape: [batch_size, max_length, word_embed_dim]
        # raw_context_vec = self.word_embed(context)
        raw_context_vec = dropout(
            raw_context_vec, self.word_dropout, shared_axes=[-2], training=self.training
        )

        # Shape: [batch_size, max_length, hidden_size]
        # context_vec = self.lstm_encoder(
        #     raw_context_vec, context_lens
        # )
        # context_vec = raw_context_vec
        context_vec = self.ctx_rnn_encoder(raw_context_vec, context_lens)[0].transpose(
            0, 1
        )

        return raw_context_vec, context_vec, context_mask

    def graph_maxpool(self, node_vec, node_mask=None):
        # Maxpool
        # Shape: (batch_size, hidden_size, num_nodes)
        graph_embedding = F.max_pool1d(node_vec, kernel_size=node_vec.size(-1)).squeeze(
            -1
        )
        return graph_embedding

    def context_maxpool(self, node_vec, node_mask=None):
        # Maxpool
        # Shape: (batch_size, num_nodes,hidden_size)
        graph_embedding = F.max_pool1d(node_vec, kernel_size=node_vec.size(-1)).squeeze(
            -1
        )
        return graph_embedding
