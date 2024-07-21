import torch
import torch.nn as nn
from .attention import Co_Attention

device = "cuda"


class MCDTransformer(torch.nn.Module):
    def __init__(
        self,
        video_dim=1024,
        asr_dim=1024,
        feature_dim=256,
        video_len=80,
        asr_len=50,
        num_attn_heads=4,
        num_trans_heads=2,
        dropout=0.1,
        experiment_type="",
    ):
        super(MCDTransformer, self).__init__()
        self.linear_video = nn.Sequential(
            nn.Linear(video_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.linear_asr = nn.Sequential(
            nn.Linear(asr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.video_transformer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_trans_heads, batch_first=True
        )
        self.co_attention = Co_Attention(
            d_k=feature_dim,
            d_v=feature_dim,
            d_model=feature_dim,
            n_heads=num_attn_heads,
            dropout=dropout,
            video_len=video_len,
            asr_len=asr_len,
            fea_v=feature_dim,
            fea_a=feature_dim,
            pos=False,
        )
        self.experiment_type = experiment_type

    def forward(self, video_fea, asr_fea):  # , comment_likes, metadata_fea):
        video_fea = self.linear_video(video_fea)  # (batch, 80,128)
        video_fea = self.video_transformer(video_fea)
        asr_fea = self.linear_asr(asr_fea)  # (batch, 50, 128)
        asr_fea = asr_fea.unsqueeze(1)
        video_fea, asr_fea = self.co_attention(
            v=video_fea, a=asr_fea, v_len=video_fea.shape[1], a_len=asr_fea.shape[1]
        )
        video_fea = torch.mean(video_fea, -2)
        asr_fea = torch.mean(asr_fea, -2)
        return video_fea + asr_fea
