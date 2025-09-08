from cellpose import models, io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图

# 步骤1: 加载图像
image_path = 'test2/tile_00167.jpg'
img = io.imread(image_path)  # RGB 图像, shape (H, W, 3)

if img.ndim != 3 or img.shape[2] != 3:
    raise ValueError("输入图像必须是RGB三通道")

# 步骤2: 加载 Cellpose 模型
model = models.CellposeModel(gpu=True, pretrained_model="cyto")  # 细胞模型

# 步骤3: 运行分割
masks, flows, styles = model.eval(
    img, 
    channels=[2,0],   # 这里指定: 第一个通道=G，第二个不用（根据你的图像通道情况可改）
    diameter=30
)

print(f"掩码最大值: {np.max(masks)}, 最小值: {np.min(masks)}")

# 步骤4: 计算每个细胞的平均RGB
cell_rgbs = []
for cell_id in range(1, masks.max()+1):  # 0是背景
    coords = np.where(masks == cell_id)
    if coords[0].size == 0:
        continue
    mean_rgb = img[coords].mean(axis=0)
    cell_rgbs.append(mean_rgb)

cell_rgbs = np.array(cell_rgbs)  # shape (N, 3)

# 步骤5: 可视化 RGB 分布
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cell_rgbs[:,0], cell_rgbs[:,1], cell_rgbs[:,2], c=cell_rgbs/255.0, s=20)

ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")
plt.title("细胞平均RGB分布 (3D)")
plt.show()

# 步骤6: 2D 投影（方便聚类/肉眼分辨）
plt.figure(figsize=(6,6))
plt.scatter(cell_rgbs[:,2], cell_rgbs[:,1], c=cell_rgbs/255.0, s=20)
plt.xlabel("B")
plt.ylabel("G")
plt.title("细胞RGB分布 (B vs G)")
plt.show()
