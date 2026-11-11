# import os
# import numpy as np
# import SimpleITK as sitk
# from scipy.stats import ttest_ind
# import matplotlib.pyplot as plt

# # 设置路径
# data_dir = '/home/students/ljx_4090_1/MDD/fsaverage5'
# all_files = os.listdir(data_dir)

# # 匹配健康人和抑郁症文件
# healthy_files = sorted([f for f in all_files if f.startswith('H') and f.endswith('.nii.gz')])
# depression_files = sorted([f for f in all_files if f.startswith('M') and f.endswith('.nii.gz')])

# # 用 SimpleITK 加载并计算均值
# def load_mean_activation_sitk(filename):
#     image = sitk.ReadImage(os.path.join(data_dir, filename))
#     data = sitk.GetArrayFromImage(image)  # shape: (230, 20484)
#     return data.mean(axis=0)  # 每个顶点取时间平均 -> shape: (20484,)

# # 构建数据集
# healthy_data = np.array([load_mean_activation_sitk(f) for f in healthy_files])
# depression_data = np.array([load_mean_activation_sitk(f) for f in depression_files])

# # t检验
# t_vals, p_vals = ttest_ind(healthy_data, depression_data, axis=0)

# # 打印显著差异点
# print("p < 0.05 的顶点数量:", np.sum(p_vals < 0.05))

# # 可视化
# plt.figure(figsize=(8, 6))
# plt.hist(p_vals, bins=50, color='tomato', edgecolor='black')
# plt.title('P-value Histogram (t-test using sitk-loaded fMRI)')
# plt.xlabel('p-value')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.grid(True)
# plt.savefig('sitk_ttest_pvalue_histogram.png')
# print("保存直方图至 sitk_ttest_pvalue_histogram.png")

# import os
# import numpy as np
# import SimpleITK as sitk
# from scipy.stats import ttest_ind
# import matplotlib.pyplot as plt

# # 设定路径
# data_dir = '/home/students/ljx_4090_1/MDD/fsaverage5'
# all_files = os.listdir(data_dir)

# healthy_files = sorted([f for f in all_files if f.startswith('H') and f.endswith('.nii.gz')])
# depression_files = sorted([f for f in all_files if f.startswith('M') and f.endswith('.nii.gz')])

# def load_fmri_data(filename):
#     image = sitk.ReadImage(os.path.join(data_dir, filename))
#     data = sitk.GetArrayFromImage(image)  # shape: (230, 20484)
#     return data

# # 加载所有数据
# healthy_data = np.array([load_fmri_data(f) for f in healthy_files])      # shape: (88, 230, 20484)
# depression_data = np.array([load_fmri_data(f) for f in depression_files])  # shape: (112, 230, 20484)

# # 转换为 (230, 88, 20484) 和 (230, 112, 20484)
# healthy_data = np.transpose(healthy_data, (1, 0, 2))
# depression_data = np.transpose(depression_data, (1, 0, 2))

# # 在每个时间点上对每个顶点做 t 检验，输出 shape = (230, 20484)
# t_vals_all = []
# p_vals_all = []

# for t in range(230):
#     t_vals, p_vals = ttest_ind(healthy_data[t], depression_data[t], axis=0, equal_var=False)
#     t_vals_all.append(t_vals)
#     p_vals_all.append(p_vals)

# t_vals_all = np.array(t_vals_all)   # shape: (230, 20484)
# p_vals_all = np.array(p_vals_all)

# # 对 t 值在时间维上平均（230个时间点）
# avg_t_vals = t_vals_all.mean(axis=0)  # shape: (20484,)
# avg_p_vals = p_vals_all.mean(axis=0)

# # 可视化平均 t 值分布
# plt.figure(figsize=(8, 6))
# plt.hist(p_vals, bins=50, color='tomato', edgecolor='black')
# plt.title('Averaged P-value Across Time Points')
# plt.xlabel('P-value')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.grid(True)
# plt.savefig('avg_pvalue_histogram.png')
# print("保存平均 t 值直方图到 avg_pvalue_histogram.png")

# # 输出统计
# significant = np.sum(avg_p_vals < 0.05)
# print(f"平均 p 值 < 0.05 的顶点数量: {significant}")

import os
import numpy as np
import SimpleITK as sitk
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# 读取函数
def load_avg_fmri(filename, path='/home/students/ljx_4090_1/MDD/fsaverage5'):
    img = sitk.ReadImage(os.path.join(path, filename))
    arr = sitk.GetArrayFromImage(img)  # shape: (230, 20484)
    return arr.mean(axis=0)  # 取时间平均后 shape: (20484,)

# 读取文件列表
data_path = '/home/students/ljx_4090_1/MDD/fsaverage5'
all_files = os.listdir(data_path)
healthy_files = sorted([f for f in all_files if f.startswith('H') and f.endswith('.nii.gz')])
depressed_files = sorted([f for f in all_files if f.startswith('M') and f.endswith('.nii.gz')])

# 加载数据
healthy_data = np.array([load_avg_fmri(f) for f in healthy_files])    # shape: (88, 20484)
depressed_data = np.array([load_avg_fmri(f) for f in depressed_files])  # shape: (112, 20484)

# 拼接整体数据
all_data = np.concatenate([healthy_data, depressed_data], axis=0)  # shape: (200, 20484)
labels = np.array([0]*len(healthy_data) + [1]*len(depressed_data))  # 0: 健康, 1: 抑郁

# PCA 降维
pca = PCA(n_components=20)
pca_data = pca.fit_transform(all_data)  # shape: (200, 20)

# KS 检验
ks_stats = []
p_values = []

for i in range(pca_data.shape[1]):
    h_group = pca_data[labels == 0, i]
    d_group = pca_data[labels == 1, i]
    stat, p = ks_2samp(h_group, d_group)
    ks_stats.append(stat)
    p_values.append(p)

# 可视化 KS 统计量
plt.figure(figsize=(10, 6))
plt.bar(range(1, 21), ks_stats, color='darkorange', edgecolor='black')
plt.xlabel('PCA Component Index')
plt.ylabel('K-S Statistic')
plt.title('Kolmogorov-Smirnov Test on PCA Components')
plt.tight_layout()
plt.savefig('ks_statistic_pca.png')
print("保存 K-S 统计量图到 ks_statistic_pca.png")

# 输出显著性统计
sig_count = sum(p < 0.05 for p in p_values)
print(f"P < 0.05 的主成分个数: {sig_count}")

