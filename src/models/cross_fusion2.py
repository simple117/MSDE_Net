import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.performer_t9 import LocalSelfAttentionBlock

# --- CGAF Component Definitions (FeedForward remains the same) ---

class FeedForward(nn.Module):
    """ 标准的前馈网络 (FFN) 模块。 """
    def __init__(self, embed_dim, ffn_dim, dropout=0.1, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear1(x); x = self.activation(x); x = self.dropout1(x)
        x = self.linear2(x); x = self.dropout2(x)
        return x

class CrossAttentionBlockWithLocalTemporal(nn.Module):
    """
    在原来 CGAFWithSelfAttention 的基础上：
    - 跨模态交叉注意力（不变）
    - Add&Norm
    - **LocalSelfAttentionBlock**（局部时序建模，新的）
    - Add&Norm
    - 全局自注意力（可选，保留 or 抛弃）
    - Add&Norm
    - FFN
    - Add&Norm
    """
    def __init__(self, dim, num_heads, ffn_dim, local_window_size=5, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.local_window_size = local_window_size

        # --- 跨模态交叉注意力 --- #
        self.cross_attn_1 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross_r = nn.LayerNorm(dim)
        self.dropout_cross_r = nn.Dropout(dropout)

        self.cross_attn_2 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross_p = nn.LayerNorm(dim)
        self.dropout_cross_p = nn.Dropout(dropout)

        # --- 新增：局部自注意力块 --- #
        self.local_block_r = LocalSelfAttentionBlock(dim, num_heads=num_heads)
        self.norm_local_r  = nn.LayerNorm(dim)
        self.dropout_local_r = nn.Dropout(dropout)

        self.local_block_p = LocalSelfAttentionBlock(dim, num_heads=num_heads)
        self.norm_local_p  = nn.LayerNorm(dim)
        self.dropout_local_p = nn.Dropout(dropout)

        # --- 原有自注意力层（可选） --- #
        self.self_attn_r = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self_r = nn.LayerNorm(dim)
        self.dropout_self_r = nn.Dropout(dropout)

        self.self_attn_p = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self_p = nn.LayerNorm(dim)
        self.dropout_self_p = nn.Dropout(dropout)

        # --- FFN --- #
        self.ffn_r = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )
        self.norm_ffn_r = nn.LayerNorm(dim)
        self.dropout_ffn_r = nn.Dropout(dropout)

        self.ffn_p = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )
        self.norm_ffn_p = nn.LayerNorm(dim)
        self.dropout_ffn_p = nn.Dropout(dropout)

    def _apply_local(self, x, local_block):
        """
        x: (B, T, D)
        1) 计算 pad_len
        2) 把时间维转到最后： (B, D, T)，再对最后一维 pad
        3) 转回 (B, T_pad, D)
        4) 分窗口、局部注意力、重拼、裁剪到原长
        """
        B, T, D = x.shape
        w = self.local_window_size

        # 1) 计算 pad 长度
        pad_len = round((w - T % w) % w)
        if pad_len > 0:
            # 2) 转到 (B, D, T)，在最后一维 pad
            x = x.transpose(1, 2)  # (B, D, T)
            x = F.pad(x, (0, pad_len))  # pad 最后一个维度
            x = x.transpose(1, 2)  # (B, T+pad_len, D)

        T_pad = x.size(1)
        n_win = round(T_pad // w)

        # 3) 划窗 + 局部注意力
        windows = x.view(B, n_win, w, D)  # (B, n_win, w, D)
        windows = windows.reshape(B * n_win, w, D)
        windows = local_block(windows)  # (B*n_win, w, D)
        windows = windows.view(B, n_win, w, D)

        # 4) 合并 + 裁剪回原序列长度
        out = windows.view(B, T_pad, D)[:, :T, :]
        return out

    def forward(self, seq1, seq2):
        # 1) 跨模态交叉注意力
        attn1, _ = self.cross_attn_1(query=seq1, key=seq2, value=seq2)
        seq1_c = self.norm_cross_r(seq1 + self.dropout_cross_r(attn1))
        attn2, _ = self.cross_attn_2(query=seq2, key=seq1, value=seq1)
        seq2_c = self.norm_cross_p(seq2 + self.dropout_cross_p(attn2))

        # 2) 局部时序建模
        local1 = self._apply_local(seq1_c, self.local_block_r)
        seq1_l = self.norm_local_r(seq1_c + self.dropout_local_r(local1))
        local2 = self._apply_local(seq2_c, self.local_block_p)
        seq2_l = self.norm_local_p(seq2_c + self.dropout_local_p(local2))

        # 3) （可选）全局自注意力
        self1, _ = self.self_attn_r(query=seq1_l, key=seq1_l, value=seq1_l)
        seq1_s = self.norm_self_r(seq1_l + self.dropout_self_r(self1))
        self2, _ = self.self_attn_p(query=seq2_l, key=seq2_l, value=seq2_l)
        seq2_s = self.norm_self_p(seq2_l + self.dropout_self_p(self2))

        # 4) FFN
        ffn1 = self.ffn_r(seq1_s)
        out1 = self.norm_ffn_r(seq1_s + self.dropout_ffn_r(ffn1))
        ffn2 = self.ffn_p(seq2_s)
        out2 = self.norm_ffn_p(seq2_s + self.dropout_ffn_p(ffn2))

        return out1, out2
# --- 修改后的 CGAF 类，使用新的 Block ---
class CGAFWithSelfAttention(nn.Module):
    """
    修改后的 CGAF 模块，其内部 Block 包含自注意力。
    """
    def __init__(self, embed_dim=512, num_heads=8, num_stages=2, ffn_dim_multiplier=4, dropout=0.1, output_dim=None):
        super().__init__()
        if embed_dim % num_heads != 0: raise ValueError(f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.embed_dim = embed_dim
        if output_dim is None: output_dim = embed_dim
        self.output_dim = output_dim
        ffn_hidden_dim = embed_dim * ffn_dim_multiplier

        # --- One-Stage: 使用包含自注意力的新 Block ---
        self.stages = nn.ModuleList([
            # *** 使用修改后的 CrossAttentionBlockWithSelfAttention ***
            CrossAttentionBlockWithLocalTemporal(embed_dim, num_heads, ffn_hidden_dim, dropout=dropout)
            for _ in range(num_stages)
        ])

        # --- Two-Stage: 最终融合 (保持不变) ---
        self.concat_dim = embed_dim * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.concat_dim, ffn_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_projection = None
        if output_dim != embed_dim: self.output_projection = nn.Linear(embed_dim, output_dim)

    def forward(self, feat_r, feat_p):
        # --- One-Stage (使用新的 Block) ---
        current_r, current_p = feat_r, feat_p
        for stage_block in self.stages:
            current_r, current_p = stage_block(current_r, current_p)
        f_hat_r, f_hat_p = current_r, current_p

        # --- Two-Stage (逻辑保持不变) ---
        if f_hat_r.shape[1] != f_hat_p.shape[1]:
            f_hat_p_permuted = f_hat_p.permute(0, 2, 1)
            # 动态创建 adapter 以确保在正确的设备上
            adapter = nn.AdaptiveAvgPool1d(f_hat_r.shape[1]).to(f_hat_p_permuted.device)
            f_hat_p_adapted_permuted = adapter(f_hat_p_permuted)
            f_hat_p_adapted = f_hat_p_adapted_permuted.permute(0, 2, 1)
        else:
            f_hat_p_adapted = f_hat_p
        combined = torch.cat([f_hat_r, f_hat_p_adapted], dim=-1)
        mlp_output = self.fusion_mlp(combined)
        mlp_output = self.final_norm(mlp_output) # Norm MLP 输出
        f_v = f_hat_r + mlp_output # 残差连接
        # if self.output_projection:
        #     f_v = self.output_projection(f_v)
        return f_v


# --- GatedCrossAttention (保持不变，新 CGAF 不直接使用它) ---
class GatedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, init_alpha=0.0):
        super().__init__()
        self.embed_dim = embed_dim; self.num_heads = num_heads; self.dropout_p = dropout
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.alpha_xattn = nn.Parameter(torch.tensor(init_alpha))
        self.norm = nn.LayerNorm(embed_dim); self.dropout = nn.Dropout(dropout); self.tanh = nn.Tanh()
    def forward(self, query, key, value):
        residual = query; attn_output, attn_weights = self.mha(query, key, value)
        attn_output_dropped = self.dropout(attn_output)
        gated_attn_contribution = self.tanh(self.alpha_xattn) * attn_output_dropped
        output = residual + gated_attn_contribution; output = self.norm(output)
        return output, attn_weights

# --- 主融合模型，现在使用包含自注意力的 CGAF ---
class CrossModalFusionWithCGAF_SelfAttn(nn.Module):
    """
    使用初始编码器和包含自注意力的 CGAF 模块融合 RGB 特征和骨骼序列的模型。
    """
    def __init__(self, embed_dim=512, num_heads=8, num_cgaf_stages=2, ffn_dim_multiplier=4, output_dim=512, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0: raise ValueError(f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.embed_dim = embed_dim; self.num_heads = num_heads; self.output_dim = output_dim
        self.target_seq_len = 10

        # --- 1. 骨骼序列编码器 (保持不变) ---
        self.skel_flatten = nn.Flatten(start_dim=2)
        self.skel_encoder = nn.Linear(21 * 2, embed_dim)
        self.skel_norm = nn.LayerNorm(embed_dim)

        # --- 2. RGB 特征编码器 (保持不变) ---
        self.rgb_encoder = nn.Linear(embed_dim, embed_dim)
        self.rgb_norm = nn.LayerNorm(embed_dim)

        # --- 3. 使用包含自注意力的 CGAF 融合模块 ---
        self.cgaf_fusion = CGAFWithSelfAttention( # *** 使用新的 CGAF 类 ***
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_stages=num_cgaf_stages,
            ffn_dim_multiplier=ffn_dim_multiplier,
            dropout=dropout,
            output_dim=output_dim
        )

    def forward(self, rgb_features, skel_sequence):
        b, rgb_len, d_rgb = rgb_features.shape
        _, skel_len, _, _ = skel_sequence.shape
        assert d_rgb == self.embed_dim, f"RGB 输入特征维度 ({d_rgb}) 与模型 embed_dim ({self.embed_dim}) 不匹配"
        assert rgb_len == self.target_seq_len, f"RGB 输入序列长度 ({rgb_len}) 与目标长度 ({self.target_seq_len}) 不匹配"

        # 1. 编码骨骼序列
        skel_proc = self.skel_flatten(skel_sequence)
        skel_encoded = self.skel_encoder(skel_proc)
        skel_encoded = self.skel_norm(skel_encoded)

        # 2. 编码 RGB 特征
        rgb_encoded = self.rgb_encoder(rgb_features)
        rgb_encoded = self.rgb_norm(rgb_encoded)

        # --- 3. 使用新的 CGAF 进行融合 ---
        fused_output = self.cgaf_fusion(feat_r=rgb_encoded, feat_p=skel_encoded)

        return fused_output

# --- 示例用法 ---
if __name__ == '__main__':
    # 定义参数
    B = 2; RGB_LEN = 10; SKEL_LEN = 40; N_KP = 21; D_KP = 2
    EMBED_DIM = 512; NUM_HEADS = 8; NUM_CGAF_STAGES = 2; FFN_MULT = 4; OUTPUT_DIM = 512

    # 创建虚拟输入数据
    dummy_rgb = torch.randn(B, RGB_LEN, EMBED_DIM)
    dummy_skel = torch.randn(B, SKEL_LEN, N_KP, D_KP)

    # 实例化使用包含自注意力的 CGAF 的融合模型
    fusion_model_cgaf_sa = CrossModalFusionWithCGAF_SelfAttn(
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_cgaf_stages=NUM_CGAF_STAGES,
        ffn_dim_multiplier=FFN_MULT, output_dim=OUTPUT_DIM, dropout=0.1
    )

    # 前向传播
    final_fused_output_cgaf_sa = fusion_model_cgaf_sa(dummy_rgb, dummy_skel)

    # 打印形状
    print("输入 RGB 特征形状:", dummy_rgb.shape)
    print("输入骨骼序列形状:", dummy_skel.shape)
    print("---")
    print("使用带自注意力的 CGAF 融合后输出形状:", final_fused_output_cgaf_sa.shape)
    assert final_fused_output_cgaf_sa.shape == (B, RGB_LEN, OUTPUT_DIM)