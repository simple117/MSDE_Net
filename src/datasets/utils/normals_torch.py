import torch
import torch.nn.functional as F
import numpy as np

def normals(depthmap, normalize=True, keep_dims=True):
    """
    计算深度图的法向量，公式为：
      normals = (-dF/dx, -dF/dy, 1)

    Args:
        depthmap (torch.Tensor): 单通道深度图，支持任意 dtype，
            要求深度图维度为 (H, W) 或 (H, W, 1)（内部会转换为 (H, W)）
        normalize (bool, optional): 如果 True，则法向量归一化为单位模
            默认: True
        keep_dims (bool, optional):
            True 时，输出法向量的空间尺寸与原深度图相同；
            False 时，输出尺寸减小（去掉边界），默认: True

    Returns:
        torch.Tensor: 计算得到的法向量，数据类型为 float32，
                      若 keep_dims=True，则形状 (H, W, 3)；否则形状 (H-2, W-2, 3)
    """
    # 若输入深度图为 (H, W, 1)，则转换为 (H, W)
    if depthmap.ndim == 3 and depthmap.shape[-1] == 1:
        depthmap = depthmap.squeeze(-1)
    depthmap = depthmap.float()

    if keep_dims:
        # mask: (H, W)
        mask = depthmap != 0
    else:
        mask = depthmap[1:-1, 1:-1] != 0

    H, W = depthmap.shape
    if keep_dims:
        normals_tensor = torch.zeros(H, W, 3, dtype=torch.float32, device=depthmap.device)
        # 使用中心差分计算：内部像素采用 (i-1, i+1) 差分
        normals_tensor[1:-1, 1:-1, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals_tensor[1:-1, 1:-1, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    else:
        normals_tensor = torch.zeros(H - 2, W - 2, 3, dtype=torch.float32, device=depthmap.device)
        normals_tensor[:, :, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals_tensor[:, :, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    # z 分量固定为 1
    normals_tensor[..., 2] = 1

    # 将 mask 中为 False 的位置设为 [0,0,0]
    normals_tensor[~mask.expand_as(normals_tensor)] = 0

    if normalize:
        # 计算 norm（加上一个极小值防止除零）
        # 注意：仅对 mask 为 True 的像素归一化
        norm_val = torch.norm(normals_tensor[mask.expand_as(normals_tensor)].view(-1, 3), p=2, dim=1,
                              keepdim=True) + 1e-12
        # 将归一化因子扩展到 [N, 3] 后赋值
        normalized = normals_tensor[mask.expand_as(normals_tensor)].view(-1, 3) / norm_val
        normals_tensor[mask.expand_as(normals_tensor)] = normalized.view(-1)
    return normals_tensor


def normals_multi(depthmaps, normalize=True, keep_dims=True):
    """
    针对多个深度图计算法向量。

    Args:
        depthmaps (torch.Tensor): 多个深度图，假设形状为 (B, H, 1, N)，
            其中 N 表示不同深度图的数量。
        normalize (bool, optional): 是否归一化输出法向量。默认 True。
        keep_dims (bool, optional): 输出是否保持与输入相同的空间尺寸。默认 True。

    Returns:
        torch.Tensor: 计算得到的法向量，形状为 (B, H, 3, N)
    """
    B, H, _, N = depthmaps.shape
    n_out = torch.zeros(B, H, 3, N, dtype=torch.float32, device=depthmaps.device)
    for i in range(N):
        # 对每个深度图，提取形状 (B, H)
        # 这里取第一个通道即可，因为输入只有一个通道
        for b in range(B):
            n_out[b, :, :, i] = normals(depthmaps[b, :, 0, i], normalize, keep_dims)
    return n_out


# =============== 测试 ===============
if __name__ == '__main__':
    # 单个深度图测试，形状 (H, W) 或 (H, W, 1)
    depth_np = (np.random.rand(100, 150) * 10).astype(np.float32)
    depth_torch = torch.from_numpy(depth_np)  # (100, 150)
    n1 = normals(depth_torch, normalize=True, keep_dims=True)
    print("单张深度图 normals, shape:", n1.shape)

    # 多深度图测试：设 B=2, H=100, N=3，此处输入形状 (B, H, 1, N)
    depthmaps_np = (np.random.rand(2, 100, 1, 3) * 10).astype(np.float32)
    depthmaps_torch = torch.from_numpy(depthmaps_np)
    n_multi = normals_multi(depthmaps_torch, normalize=True, keep_dims=True)
    print("多张深度图 normals_multi, shape:", n_multi.shape)
