import torch
import torch.nn as nn
import torch.nn.functional as F


#############################
# 局部自注意力块：在单个小窗口内建模局部时序
#############################
class LocalSelfAttentionBlock(nn.Module):
    """
    对一个局部窗口内（例如5帧）的时序特征进行自注意力建模，
    并利用残差连接和前馈网络增强特征表达。
    """

    def __init__(self, dim, num_heads=4):
        super(LocalSelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, window_size, dim)
        res = x
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(res + attn_out)

        res = x
        ff_out = self.ffn(x)
        x = self.norm2(res + ff_out)
        return x  # (B, window_size, dim)


#############################
# 局部时序建模模块：将整个序列分为多个窗口，对每个窗口内部建模，然后提取窗口摘要 token
#############################
class LocalTemporalModel(nn.Module):
    """
    局部建模阶段：将输入时序 (B, T, dim) 分割成多个窗口（每个窗口长度 window_size），
    对每个窗口内采用 LocalSelfAttentionBlock 进行局部建模，再将每个窗口的输出归约为一个 token（例如平均）。
    """

    def __init__(self, dim, window_size=5, num_heads=4):
        super(LocalTemporalModel, self).__init__()
        self.window_size = window_size
        self.dim = dim
        self.local_block = LocalSelfAttentionBlock(dim, num_heads=num_heads)

    def forward(self, x):
        # x: (B, T, dim)
        B, T, D = x.shape
        # 如果 T 不能被 window_size 整除，pad到整除
        pad_len = (self.window_size - T % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # 在时间维 pad
        T_pad = x.shape[1]
        num_windows = T_pad // self.window_size
        # 划分窗口，得到 shape: (B, num_windows, window_size, D)
        windows = x.view(B, num_windows, self.window_size, D)
        # 将每个窗口合并到 batch 维度： (B * num_windows, window_size, D)
        windows = windows.reshape(B * num_windows, self.window_size, D)
        # 对每个窗口内做局部自注意力建模
        windows = self.local_block(windows)  # (B*num_windows, window_size, D)
        # 将每个窗口归约为一个 token，例如取均值
        window_tokens = windows.mean(dim=1)  # (B*num_windows, D)
        # 恢复为 (B, num_windows, D)
        window_tokens = window_tokens.view(B, num_windows, D)
        return window_tokens  # 每个窗口的摘要特征


# #############################
# # 全局聚合模块：对所有窗口 token 进行全局自注意力建模
# #############################
# class GlobalTemporalModel(nn.Module):
#     """
#     全局聚合阶段：将所有窗口摘要 token 组成短时序，
#     使用 Transformer Encoder 或多头自注意力对窗口间依赖进行建模，
#     最后汇聚为全局特征表示。
#     """
#
#     def __init__(self, dim, num_layers=1, num_heads=4):
#         super(GlobalTemporalModel, self).__init__()
#         encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#     def forward(self, x):
#         # x: (B, num_windows, D)
#         # TransformerEncoder输出同样维度
#         x = self.encoder(x)  # (B, num_windows, D)
#         # 对窗口 token 进行全局平均聚合，得到 (B, D)
#         x = x.mean(dim=1)
#         return x

class MSTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7], dilation=1):
        super(MSTCNBlock, self).__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = ((k - 1) // 2) * dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=padding, dilation=dilation),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
        self.fusion = nn.Sequential(
            nn.Conv1d(len(kernel_sizes) * out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        out = self.fusion(out)
        out = out.transpose(1, 2)  # (B, T, C)
        return out

class GlobalTemporalModel(nn.Module):
    def __init__(self, dim, num_blocks=2, kernel_sizes=[3,5,7]):
        super(GlobalTemporalModel, self).__init__()
        self.blocks = nn.Sequential(
            *[MSTCNBlock(dim, dim, kernel_sizes=kernel_sizes) for _ in range(num_blocks)]
        )

    def forward(self, x):
        # x: (B, T, D)
        x = self.blocks(x)
        x = x.mean(dim=1)  # Global average pooling over time
        return x

#############################
# 整体层次化时序模型
#############################
class HierarchicalTemporalModel(nn.Module):
    """
    结合局部建模与全局聚合的层次化时序模型，
    输入: ASFF 后的空间特征经过全局平均池化得到时序特征向量 (B*T, dim)
    重构为 (B, T, dim) 后经过局部建模和全局聚合，得到最终全局特征用于分类。
    """

    def __init__(self, feature_dim=512, num_frames=40, local_window_size=5, num_heads=4, global_layers=1,
                 num_classes=25):
        super(HierarchicalTemporalModel, self).__init__()
        self.num_frames = num_frames
        self.feature_dim = feature_dim



        # 局部建模阶段：将连续帧特征序列分为多个窗口，得到窗口摘要 token
        self.local_model = LocalTemporalModel(dim=feature_dim, window_size=local_window_size, num_heads=num_heads)

        # 全局聚合阶段：对所有窗口 token 进行 Transformer Encoder 建模，再取均值得到全局表示
        self.global_model = GlobalTemporalModel(dim=feature_dim)



    def forward(self, x):
        """
        输入:
            x: Tensor, (B*T, feature_dim, H, W) —— 例如 ASFF 后输出 (B*T, 512, 6,8)
            其中 T 为连续帧数（例如40），B 为 batch size
        输出:
            logits: (B, num_classes)
        """
        B, T, C = x.shape




        # 2. 局部建模阶段
        window_tokens = self.local_model(x)  # (B, num_windows, feature_dim)

        # 3. 全局聚合阶段
        global_feat = self.global_model(window_tokens)  # (B, feature_dim)


        return global_feat


#############################
# 示例用法
#############################
if __name__ == '__main__':
    # 假设 batch size = 8, T = 40, 每帧ASFF输出的特征形状 (512, 6, 8)
    T = 40
    a = torch.randn([2, 40, 512])

    # 构造层次化时序模型
    model = HierarchicalTemporalModel(feature_dim=512, num_frames=T, local_window_size=5, num_heads=4, global_layers=1,
                                      num_classes=25)
    logits = model(a)  # 输出形状 (B, num_classes)
    print("输出形状:", logits.shape)
