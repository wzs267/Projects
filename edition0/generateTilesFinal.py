import os
import numpy as np
from cellpose import models, io
from sklearn.cluster import KMeans
from skimage import io as skio
from skimage import exposure
def detect_and_crop(image_path, outdir="tiles", diameter=30, diff_threshold=0.2, min_cells=20, margin=2):
    """
    1. Cellpose 检测细胞
    2. 聚类筛选染色细胞（或全保留）
    3. 按细胞位置切 tiles 保存
    """
    os.makedirs(outdir, exist_ok=True)

    # 1. 加载图像 & 分割
    img = skio.imread(image_path)
    model = models.CellposeModel(gpu=True, model_type='cpsam')
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
        print(f"⚠️ {image_path} 没有检测到细胞")
        return

    cell_rgbs = np.array(cell_rgbs)
    X = cell_rgbs[:, [1,2]]  # G,B

    # 3. 少细胞 → 全保留
    if len(cell_rgbs) < min_cells:
        keep_ids = cell_ids
    else:
        # KMeans 聚类
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        labels = kmeans.labels_

        # 判断两类差异
        cluster_means = [X[labels==i,1].mean()/(X[labels==i,0].mean()+1e-6) for i in range(2)]
        diff = abs(cluster_means[0] - cluster_means[1])

        if diff < diff_threshold:
            print(f"ℹ️ {image_path}: 聚类差异不明显 (diff={diff:.2f}) → 全部保留")
            keep_ids = cell_ids
        else:
            stained_cluster = np.argmax(cluster_means)
            keep_ids = np.array(cell_ids)[labels == stained_cluster]

        # 4. 按照 bounding box 切割 tile
    for i, cid in enumerate(keep_ids):
        coords = np.where(masks == cid)
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()

        # 保证固定大小: 以直径为参考，加 margin
        h, w = y2-y1, x2-x1
        size = max(h, w, diameter) + margin
        cy, cx = (y1+y2)//2, (x1+x2)//2
        y1, y2 = max(0, cy-size//2), min(img.shape[0], cy+size//2)
        x1, x2 = max(0, cx-size//2), min(img.shape[1], cx+size//2)

        tile = img[y1:y2, x1:x2]

        # 🚨 新增：过滤低对比度 tile
        if exposure.is_low_contrast(tile, fraction_threshold=0.05):
            print(f"⚠️ 跳过低对比度 tile {i} ({cid}) from {image_path}")
            continue

        save_path = os.path.join(outdir, f"{os.path.basename(image_path).split('.')[0]}_cell{i}.png")
        skio.imsave(save_path, tile)


    print(f"✅ {image_path}: 提取并保存了 {len(keep_ids)} 个染色细胞 tiles 到 {outdir}/")


# ===== 批量处理文件夹 =====
def process_folder(indir="images", outdir="tiles"):
    for fname in os.listdir(indir):
        if fname.lower().endswith((".jpg", ".bmp", ".png", ".tif")):
            detect_and_crop(os.path.join(indir, fname), outdir=outdir)


# 示例
process_folder("test2", outdir="99test2_output")
