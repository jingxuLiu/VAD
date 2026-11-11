"""
生成13148模型节点的mask文件
从activated_voxels_idx.txt读取节点索引，生成20484长度的bool mask
"""
import numpy as np

# 读取激活体素索引
activated_voxels_idx = np.loadtxt(
    "/home/students/ljx_4090_2/MDD/src/activated_voxels_idx.txt",
    dtype=int
)

print(f"读取到 {len(activated_voxels_idx)} 个激活体素索引")
print(f"索引范围: {activated_voxels_idx.min()} ~ {activated_voxels_idx.max()}")

# 创建20484长度的mask（全脑fsaverage5顶点数）
full_vertices = 20484
node_mask = np.zeros(full_vertices, dtype=bool)

# 将激活的体素位置设为True
node_mask[activated_voxels_idx] = True

print(f"Mask中True的数量: {node_mask.sum()}")
print(f"Mask中False的数量: {(~node_mask).sum()}")

# 保存mask文件
output_path = "/home/students/ljx_4090_2/MDD/isc_analysis_ROI/model_13148_mask.npy"
np.save(output_path, node_mask)

print(f"\nMask文件已保存到: {output_path}")
print(f"文件大小: {np.load(output_path).nbytes / 1024:.2f} KB")