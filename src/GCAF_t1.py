import torch
import torch.nn as nn
import torch.nn.functional as F

# —— 直接复用已有模块 —— #
# LocalTemporalModel, GlobalTemporalModel 来自你的层次化时序模型代码
# CGAFWithSelfAttention 来自你的双流融合代码

from performer_t9 import LocalTemporalModel, GlobalTemporalModel
from src.models.cross_fusion import CGAFWithSelfAttention


class CrossModalHierarchicalTemporalFusion(nn.Module):
    """
    分层时序跨模态融合网络：
      1) 局部时序建模：LocalTemporalModel
      2) 窗口级跨模态融合：CGAFWithSelfAttention
      3) 全局时序聚合：GlobalTemporalModel
      4) 线性分类头
    """
    def __init__(self,
                 embed_dim=512,
                 num_heads=8,
                 local_window_size=5,
                 cgaf_stages=2,
                 ffn_mult=4,
                 global_blocks=2,
                 num_classes=25,
                 dropout=0.1):
        super().__init__()

        # 骨骼流编码器
        self.skel_flatten = nn.Flatten(start_dim=2)
        self.skel_encoder = nn.Linear(21 * 2, embed_dim)
        self.skel_norm = nn.LayerNorm(embed_dim)

        # RGB 流编码器
        self.rgb_encoder = nn.Linear(embed_dim, embed_dim)
        self.rgb_norm = nn.LayerNorm(embed_dim)

        # —— 1) 局部时序建模 —— #
        self.local_model_r = LocalTemporalModel(dim=embed_dim,
                                                window_size=local_window_size,
                                                num_heads=num_heads)
        self.local_model_p = LocalTemporalModel(dim=embed_dim,
                                                window_size=local_window_size,
                                                num_heads=num_heads)

        # —— 2) 窗口级跨模态融合 —— #
        self.cgaf = CGAFWithSelfAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          num_stages=cgaf_stages,
                                          ffn_dim_multiplier=ffn_mult,
                                          dropout=dropout,
                                          output_dim=embed_dim)

        # —— 3) 全局时序聚合 —— #
        # 这里用 MSTCN 构成的 GlobalTemporalModel
        self.global_model = GlobalTemporalModel(dim=embed_dim,
                                                num_blocks=global_blocks,
                                                kernel_sizes=[3,5,7])

        # —— 4) 分类头 —— #
        # self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, rgb_features, skel_sequence):
        """
        Args:
          rgb_features: (B, T_r, embed_dim)
          skel_sequence: (B, T_p, 21, 2)
        Returns:
          logits: (B, num_classes)
        """
        B, T_r, _ = rgb_features.shape

        # 1. 骨骼流编码
        sk = self.skel_flatten(skel_sequence)           # (B, T_p, 21*2)
        sk = self.skel_encoder(sk)                     # (B, T_p, embed_dim)
        sk = self.skel_norm(sk)

        # 2. RGB 流编码
        rg = self.rgb_encoder(rgb_features)             # (B, T_r, embed_dim)
        rg = self.rgb_norm(rg)

        # —— 1) 局部时序建模 —— #
        tokens_r = self.local_model_r(rg)               # (B, num_win_r, embed_dim)
        tokens_p = self.local_model_p(sk)               # (B, num_win_p, embed_dim)

        # —— 2) 窗口级跨模态融合 —— #
        # CGAFWithSelfAttention 允许两条序列长度不同
        fused_tokens = self.cgaf(tokens_r, tokens_p)    # (B, num_win_r, embed_dim)

        # —— 3) 全局时序聚合 —— #
        global_feat = self.global_model(fused_tokens)   # (B, embed_dim)

        # —— 4) 分类 —— #
        # logits = self.classifier(global_feat)           # (B, num_classes)
        return global_feat


# —— 简单示例 —— #
if __name__ == "__main__":
    B = 2
    T_r = 10    # RGB 窗口帧数
    T_p = 40    # 骨骼帧数
    EMB = 512

    model = CrossModalHierarchicalTemporalFusion(
        embed_dim=EMB,
        num_heads=8,
        local_window_size=5,
        cgaf_stages=2,
        ffn_mult=4,
        global_blocks=2,
        num_classes=25,
        dropout=0.1
    )

    dummy_rgb  = torch.randn(B, T_r, EMB)
    dummy_skel = torch.randn(B, T_p, 21, 2)

    out = model(dummy_rgb, dummy_skel)
    print("Logits shape:", out.shape)  # (2, 25)
