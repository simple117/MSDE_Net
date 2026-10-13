import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DeformableConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 偏移量卷积层
        self.offset_conv = nn.Conv1d(
            in_channels,
            kernel_size,  # 1D 偏移量，每个卷积核位置一个偏移
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # 标准卷积层用于特征提取
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # 初始化偏移量卷积权重
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, x):
        # 输入 x 形状: (batch_size, in_channels, length)

        # 生成偏移量
        offset = self.offset_conv(x)  # 形状: (batch_size, kernel_size, length_out)

        # 获取输出长度
        batch_size, _, length_out = offset.size()

        # 生成基础网格（时间维度上的位置）
        length = x.size(2)
        grid = torch.arange(0, length, device=x.device, dtype=x.dtype)
        grid = grid.view(1, 1, length).repeat(batch_size, self.kernel_size, 1)  # 形状: (batch_size, kernel_size, length)

        # 应用偏移量到网格
        offset = offset.view(batch_size, self.kernel_size, length_out)  # 形状: (batch_size, kernel_size, length_out)
        sample_points = grid[:, :, :length_out] + offset  # 广播对齐形状

        # 归一化采样点到 [-1, 1]
        sample_points = 2.0 * sample_points / (length - 1) - 1.0  # 形状: (batch_size, kernel_size, length_out)

        # 为 grid_sample 调整形状: (batch_size, length_out, kernel_size, 2)
        # 需要 2D 网格，因此添加一个虚拟维度
        sample_points = sample_points.permute(0, 2, 1).unsqueeze(-1)  # 形状: (batch_size, length_out, kernel_size, 1)
        sample_points = torch.cat([sample_points, torch.zeros_like(sample_points)],
                                  dim=-1)  # 形状: (batch_size, length_out, kernel_size, 2)

        # 为 grid_sample 扩展输入: (batch_size, in_channels, length, 1)
        x_expanded = x.unsqueeze(-1)

        # 执行可变形采样
        x_offset = F.grid_sample(
            x_expanded,  # 形状: (batch_size, in_channels, length, 1)
            sample_points,  # 形状: (batch_size, length_out, kernel_size, 2)
            align_corners=True,
            mode='bilinear'
        )  # 输出形状: (batch_size, in_channels, length_out, kernel_size)

        # 调整输出形状以匹配 Conv1d 输入: (batch_size, in_channels, length_out)
        x_offset = x_offset.sum(dim=-1)  # 对 kernel_size 维度求和，模拟卷积核采样

        # 应用标准卷积
        out = self.conv(x_offset)

        return out


# 示例用法
if __name__ == "__main__":
    # 模拟手势识别时间序列数据
    batch_size = 16
    in_channels = 3  # 例如加速度计的 x, y, z 轴
    length = 100  # 时间序列长度
    out_channels = 64
    kernel_size = 3

    # 创建随机输入张量
    x = torch.randn(batch_size, in_channels, length)

    # 初始化可变形卷积层
    deform_conv = DeformableConv1d(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=kernel_size,
        stride=1,
        padding=1
    )

    # 前向传播
    output = deform_conv(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")