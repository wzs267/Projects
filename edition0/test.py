from cellpose import models, io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def detect_stained_cells(image_path, diameter=30, diff_threshold=0.2, min_cells=40):
    """
    输入: 图像路径
    输出: (mask_stained, cell_rgbs, labels)
    """
    # 1. 加载图像 & 分割
    img = io.imread(image_path)
    model = models.CellposeModel(gpu=True, pretrained_model="cyto")
    masks, flows, styles = model.eval(img, channels=[2,0], diameter=diameter)

    # 2. 提取每个细胞的平均RGB
    cell_rgbs, cell_ids = [], []
    for cell_id in range(1, masks.max()+1):
        coords = np.where(masks == cell_id)
        if coords[0].size == 0:
            continue
        mean_rgb = img[coords].mean(axis=0)
        cell_rgbs.append(mean_rgb)
        cell_ids.append(cell_id)

    if len(cell_rgbs) == 0:
        print("⚠️ 没有检测到细胞")
        return None, None, None

    cell_rgbs = np.array(cell_rgbs)   # (N,3)
    X = cell_rgbs[:, [1,2]]           # (G,B)

    # 3. 少细胞 → 全保留
    if len(cell_rgbs) < min_cells:
        print(f"⚠️ 细胞数太少 ({len(cell_rgbs)} 个)，直接保留")
        return masks > 0, cell_rgbs, np.zeros(len(cell_rgbs))

    # 4. KMeans 聚类
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    labels = kmeans.labels_

    # 5. 检查聚类差异
    cluster_means = []
    for i in range(2):
        mean_g = X[labels==i,0].mean()
        mean_b = X[labels==i,1].mean()
        cluster_means.append(mean_b / (mean_g+1e-6))

    diff = abs(cluster_means[0] - cluster_means[1])

    if diff < diff_threshold:
        print(f"ℹ️ 聚类差异不明显 (diff={diff:.2f})，全部保留")
        return masks > 0, cell_rgbs, np.zeros(len(cell_rgbs))

    # 6. 差异明显 → 选择染色细胞类
    stained_cluster = np.argmax(cluster_means)
    stained_ids = np.array(cell_ids)[labels == stained_cluster]
    mask_stained = np.isin(masks, stained_ids)

    return mask_stained, cell_rgbs, labels


# ========== 示例运行 ==========
image_path = "test2/tile_00167.jpg"
mask_stained, cell_rgbs, labels = detect_stained_cells(image_path)

if mask_stained is not None:
    plt.imshow(mask_stained, cmap="gray")
    plt.title("检测到的染色细胞")
    plt.show()

    plt.scatter(cell_rgbs[:,2], cell_rgbs[:,1], c=labels, cmap="coolwarm", s=20)
    plt.xlabel("B")
    plt.ylabel("G")
    plt.title("细胞 RGB 分布 (B vs G)")
    plt.show()
