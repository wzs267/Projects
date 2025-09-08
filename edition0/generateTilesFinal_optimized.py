"""
优化版本：使用单进程GPU处理 + 多进程图像预处理
解决GPU共享问题，提高整体效率
"""

import os
import glob
import numpy as np
import time
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from cellpose import models, io as cellpose_io
from sklearn.cluster import KMeans
from skimage import io as skio, exposure
import matplotlib.pyplot as plt
from queue import Queue
import threading

def preprocess_image(image_path):
    """预处理：只加载和基本处理图像，不进行CellPose推理"""
    try:
        img = skio.imread(image_path)
        if img is None or img.size == 0:
            return None, None, image_path
        return img, os.path.basename(image_path), image_path
    except Exception as e:
        print(f"❌ 预处理失败: {image_path} - {str(e)}")
        return None, None, image_path

def detect_and_crop_gpu_batch(image_batch, outdir="tiles", diameter=30, 
                            diff_threshold=0.2, min_cells=70, margin=2):
    """
    使用GPU批量处理图像（CellPose推理部分） - 与原始逻辑完全一致
    """
    
    # 只在主进程中创建模型，使用GPU
    model = models.CellposeModel(gpu=True, model_type='cpsam')  # 使用CPSAM模型
    total_saved = 0
    
    for img, filename, full_path in image_batch:
        if img is None:
            continue
            
        try:
            print(f"🔄 GPU处理: {filename}")
            
            # 1. CellPose 分割 - 与原始逻辑一致
            masks, flows, styles = model.eval(img, channels=[2,0], diameter=diameter)
            
            # 2. 提取每个细胞的平均RGB - 与原始逻辑一致
            cell_rgbs, cell_ids = [], []
            for cell_id in range(1, masks.max()+1):
                coords = np.where(masks == cell_id)
                if coords[0].size == 0:
                    continue
                mean_rgb = img[coords].mean(axis=0)
                cell_rgbs.append(mean_rgb)
                cell_ids.append(cell_id)

            if len(cell_rgbs) == 0:
                print(f"⚠️ {filename}: 没有检测到细胞")
                continue

            cell_rgbs = np.array(cell_rgbs)
            X = cell_rgbs[:, [1,2]]  # G,B通道

            # 3. 聚类筛选 - 与原始逻辑完全一致
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
                    print(f"ℹ️ {filename}: 聚类差异不明显 (diff={diff:.2f}) → 全部保留")
                    keep_ids = cell_ids
                else:
                    stained_cluster = np.argmax(cluster_means)
                    keep_ids = np.array(cell_ids)[labels == stained_cluster]

            # 4. 按照 bounding box 切割 tile - 与原始逻辑完全一致
            saved_count = 0
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

                # 🚨 新增：过滤低对比度 tile - 与原始逻辑一致
                if exposure.is_low_contrast(tile, fraction_threshold=0.05):
                    continue

                save_path = os.path.join(outdir, f"{filename.split('.')[0]}_cell{i}.png")
                skio.imsave(save_path, tile)
                saved_count += 1

            total_saved += saved_count
            print(f"✅ {filename}: 提取并保存了 {saved_count} 个染色细胞 tiles")
            
        except Exception as e:
            print(f"❌ GPU处理失败: {filename} - {str(e)}")
            traceback.print_exc()
    
    return total_saved

def process_folder_optimized(indir="images", outdir="tiles", batch_size=20, 
                           preprocess_workers=4, diameter=30, diff_threshold=0.2, 
                           min_cells=20, margin=2, max_images=None):
    """
    优化的处理流程：
    1. 多进程预处理（CPU密集型）
    2. 单进程GPU批处理（GPU密集型）
    """
    
    os.makedirs(outdir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']:
        image_files.extend(glob.glob(os.path.join(indir, ext)))
    
    if not image_files:
        print(f"❌ 在 {indir} 中没有找到图像文件")
        return
    
    # 限制处理的图像数量
    if max_images and max_images < len(image_files):
        image_files = image_files[:max_images]
        print(f"📁 发现 {len(glob.glob(os.path.join(indir, '*.jpg')))} 个图像文件，限制处理前 {max_images} 个")
    else:
        print(f"📁 发现 {len(image_files)} 个图像文件")
    
    print(f"🔧 配置: 预处理进程={preprocess_workers}, GPU批大小={batch_size}")
    
    start_time = time.time()
    total_tiles = 0
    
    # 分批处理
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        
        print(f"\n🔄 处理批次 {batch_start//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} "
              f"({len(batch_files)} 个文件)")
        
        # 步骤1: 多进程预处理
        print("📋 步骤1: 多进程预处理图像...")
        batch_images = []
        
        with ProcessPoolExecutor(max_workers=preprocess_workers) as executor:
            futures = [executor.submit(preprocess_image, img_path) for img_path in batch_files]
            
            for future in as_completed(futures):
                img, filename, full_path = future.result()
                if img is not None:
                    batch_images.append((img, filename, full_path))
        
        print(f"✅ 预处理完成: {len(batch_images)}/{len(batch_files)} 个图像成功加载")
        
        # 步骤2: GPU批处理
        if batch_images:
            print("🎯 步骤2: GPU批处理（CellPose推理 + 切割）...")
            batch_tiles = detect_and_crop_gpu_batch(
                batch_images, outdir, diameter, diff_threshold, min_cells, margin
            )
            total_tiles += batch_tiles
            print(f"✅ 批次完成: 生成 {batch_tiles} 个tiles")
    
    # 统计结果
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n🎉 全部处理完成!")
    print(f"📊 统计信息:")
    print(f"   - 处理图像: {len(image_files)} 个")
    print(f"   - 生成tiles: {total_tiles} 个")
    print(f"   - 总耗时: {elapsed_time:.1f} 秒")
    print(f"   - 平均速度: {len(image_files)/elapsed_time:.2f} 图像/秒")
    print(f"   - Tiles生成速度: {total_tiles/elapsed_time:.2f} tiles/秒")
    print(f"📁 输出目录: {outdir}")

def main():
    parser = argparse.ArgumentParser(description='并行细胞检测和切割（优化版）')
    parser.add_argument('--input_dir', type=str, default='99all612_512', help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='tiles_output', help='输出tiles目录')
    parser.add_argument('--batch_size', type=int, default=20, help='GPU批处理大小')
    parser.add_argument('--preprocess_workers', type=int, default=4, help='预处理进程数')
    parser.add_argument('--diameter', type=int, default=30, help='细胞直径参数')
    parser.add_argument('--diff_threshold', type=float, default=0.2, help='聚类差异阈值')
    parser.add_argument('--min_cells', type=int, default=20, help='最小细胞数阈值')
    parser.add_argument('--margin', type=int, default=2, help='切割边距')
    parser.add_argument('--max_images', type=int, default=1000, help='最大处理图像数量（测试用）')
    
    args = parser.parse_args()
    
    print("🚀 启动优化版并行细胞检测系统")
    print(f"📂 输入目录: {args.input_dir}")
    print(f"📂 输出目录: {args.output_dir}")
    print(f"🔢 最大处理数量: {args.max_images}")
    
    process_folder_optimized(
        indir=args.input_dir,
        outdir=args.output_dir,
        batch_size=args.batch_size,
        preprocess_workers=args.preprocess_workers,
        diameter=args.diameter,
        diff_threshold=args.diff_threshold,
        min_cells=args.min_cells,
        margin=args.margin,
        max_images=args.max_images
    )

if __name__ == "__main__":
    main()
