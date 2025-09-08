# Rare Cell Detection for tiles_output
# 基于DAE_RED_TF2设计的异常检测流水线
# 适配32x32x3 RGB细胞tiles

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from skimage import io
import argparse
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class CellDAE:
    """基于DAE_RED_TF2设计的细胞异常检测模型"""
    
    def __init__(self, input_shape=(32, 32, 3), latent_dim=64):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        
    def build_model(self):
        """构建去噪自编码器模型"""
        
        # 编码器
        input_img = keras.layers.Input(shape=self.input_shape)
        
        # 第一层卷积块
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 16x16
        
        # 第二层卷积块
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 8x8
        
        # 第三层卷积块
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 4x4
        
        # 展平并降维到潜在空间
        x = keras.layers.Flatten()(x)
        encoded = keras.layers.Dense(self.latent_dim, activation='relu')(x)
        
        # 解码器
        x = keras.layers.Dense(4 * 4 * 128, activation='relu')(encoded)
        x = keras.layers.Reshape((4, 4, 128))(x)
        
        # 上采样层
        x = keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.UpSampling2D((2, 2))(x)  # 8x8
        
        x = keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.UpSampling2D((2, 2))(x)  # 16x16
        
        x = keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.UpSampling2D((2, 2))(x)  # 32x32
        
        # 输出层
        decoded = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        # 创建模型
        self.model = keras.Model(input_img, decoded)
        self.encoder = keras.Model(input_img, encoded)
        
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """编译模型"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
    def add_noise(self, images, noise_factor=0.1):
        """添加高斯噪声用于去噪训练"""
        noise = np.random.normal(0, noise_factor, images.shape)
        noisy_images = np.clip(images + noise, 0., 1.)
        return noisy_images
    
    def train(self, train_data, epochs=50, batch_size=32, noise_factor=0.1, validation_split=0.2):
        """训练模型"""
        
        # 数据归一化
        train_data = train_data.astype('float32') / 255.0
        
        # 添加噪声
        noisy_train = self.add_noise(train_data, noise_factor)
        
        # 训练回调
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # 训练
        history = self.model.fit(
            noisy_train, train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def calculate_reconstruction_error(self, images):
        """计算重建误差作为异常分数"""
        images = images.astype('float32') / 255.0
        reconstructed = self.model.predict(images, verbose=0)
        
        # 计算L2距离
        errors = np.sqrt(np.sum((images - reconstructed) ** 2, axis=(1, 2, 3)))
        
        return errors, reconstructed
    
    def save_model(self, save_path):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, 'cell_dae_model.h5'))
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        self.model = keras.models.load_model(model_path)
        print(f"模型已加载: {model_path}")


def load_tiles_dataset(tiles_dir, max_samples=None, sample_fraction=0.1):
    """
    加载tiles数据集
    Args:
        tiles_dir: tiles目录路径
        max_samples: 最大样本数（如果None则使用sample_fraction）
        sample_fraction: 采样比例（当数据量太大时）
    """
    
    # 获取所有PNG文件
    tile_files = glob.glob(os.path.join(tiles_dir, "*.png"))
    print(f"发现 {len(tile_files)} 个tile文件")
    
    # 如果数据量太大，进行采样
    if max_samples is None:
        max_samples = int(len(tile_files) * sample_fraction)
    
    if len(tile_files) > max_samples:
        tile_files = np.random.choice(tile_files, max_samples, replace=False)
        print(f"采样 {len(tile_files)} 个文件用于训练")
    
    # 加载图像
    images = []
    valid_files = []
    
    for i, file_path in enumerate(tile_files):
        try:
            img = io.imread(file_path)
            if img.shape == (32, 32, 3):  # 确保是RGB格式
                images.append(img)
                valid_files.append(file_path)
        except Exception as e:
            print(f"跳过损坏的文件: {file_path}")
        
        if i % 1000 == 0:
            print(f"已加载 {i}/{len(tile_files)} 个文件")
    
    images = np.array(images)
    print(f"成功加载 {len(images)} 个有效tiles，形状: {images.shape}")
    
    return images, valid_files


def filter_artifacts(errors, images, file_paths, top_k=1000):
    """
    简单的伪影过滤器
    - 过滤过亮或过暗的图像
    - 过滤颜色单一的图像
    """
    
    # 计算每个图像的统计特征
    mean_intensities = np.mean(images, axis=(1, 2, 3))
    std_intensities = np.std(images, axis=(1, 2, 3))
    
    # 更宽松的过滤条件
    valid_mask = (
        (mean_intensities > 0.05) &  # 不要太暗（放宽）
        (mean_intensities < 0.95) &  # 不要太亮（放宽）
        (std_intensities > 0.02)     # 要有一定的变化（放宽）
    )
    
    # 应用过滤
    filtered_errors = errors[valid_mask]
    filtered_images = images[valid_mask]
    filtered_paths = [file_paths[i] for i in range(len(file_paths)) if valid_mask[i]]
    
    print(f"过滤后保留 {len(filtered_errors)} / {len(errors)} 个样本")
    
    # 如果过滤后样本太少，直接使用原始数据
    if len(filtered_errors) < 50:
        print("⚠️ 过滤后样本过少，使用原始数据")
        filtered_errors = errors
        filtered_images = images
        filtered_paths = file_paths
    
    # 返回top_k最高误差的样本
    if len(filtered_errors) > top_k:
        top_indices = np.argsort(filtered_errors)[-top_k:]
        return (filtered_errors[top_indices], 
                filtered_images[top_indices], 
                [filtered_paths[i] for i in top_indices])
    else:
        # 如果样本数少于top_k，返回所有样本（按误差降序）
        sorted_indices = np.argsort(filtered_errors)[::-1]
        return (filtered_errors[sorted_indices], 
                filtered_images[sorted_indices], 
                [filtered_paths[i] for i in sorted_indices])


def visualize_results(original_images, reconstructed_images, errors, save_path, n_display=20):
    """可视化结果"""
    
    # 选择误差最高的n_display个样本
    top_indices = np.argsort(errors)[-n_display:]
    
    fig, axes = plt.subplots(3, n_display, figsize=(n_display*2, 6))
    
    for i, idx in enumerate(top_indices):
        # 原图
        axes[0, i].imshow(original_images[idx])
        axes[0, i].set_title(f'Original\nError: {errors[idx]:.3f}')
        axes[0, i].axis('off')
        
        # 重建图
        axes[1, i].imshow(reconstructed_images[idx])
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
        
        # 误差图
        diff = np.abs(original_images[idx] - reconstructed_images[idx])
        axes[2, i].imshow(diff)
        axes[2, i].set_title('Difference')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='细胞异常检测')
    parser.add_argument('--tiles_dir', type=str, default='tiles_output', 
                       help='tiles目录路径')
    parser.add_argument('--do_train', action='store_true', 
                       help='是否训练模型')
    parser.add_argument('--do_detect', action='store_true', 
                       help='是否进行异常检测')
    parser.add_argument('--model_path', type=str, default='cell_dae_model', 
                       help='模型保存/加载路径')
    parser.add_argument('--output_dir', type=str, default='anomaly_results', 
                       help='结果输出目录')
    parser.add_argument('--sample_size', type=int, default=10000, 
                       help='训练样本数量')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='训练轮数')
    parser.add_argument('--latent_dim', type=int, default=64, 
                       help='潜在空间维度')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建模型
    dae = CellDAE(latent_dim=args.latent_dim)
    dae.build_model()
    dae.compile_model()
    
    if args.do_train:
        print("=== 开始训练 ===")
        
        # 加载训练数据
        train_images, train_files = load_tiles_dataset(
            args.tiles_dir, 
            max_samples=args.sample_size
        )
        
        # 训练模型
        history = dae.train(
            train_images, 
            epochs=args.epochs,
            batch_size=32,
            noise_factor=0.1
        )
        
        # 保存模型
        dae.save_model(args.model_path)
        
        # 保存训练历史
        np.save(os.path.join(args.output_dir, 'training_history.npy'), history.history)
        
        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
        plt.close()
    
    if args.do_detect:
        print("=== 开始异常检测 ===")
        
        # 加载模型
        model_file = os.path.join(args.model_path, 'cell_dae_model.h5')
        if os.path.exists(model_file):
            dae.load_model(model_file)
        else:
            print("错误：未找到训练好的模型，请先训练模型")
            return
        
        # 加载所有数据进行检测（分批处理）
        all_files = glob.glob(os.path.join(args.tiles_dir, "*.png"))
        print(f"开始检测 {len(all_files)} 个tiles...")
        
        batch_size = 1000
        all_errors = []
        all_images = []
        all_file_names = []
        
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i+batch_size]
            
            # 加载批次数据
            batch_images = []
            batch_file_names = []
            
            for file_path in batch_files:
                try:
                    img = io.imread(file_path)
                    if img.shape == (32, 32, 3):
                        batch_images.append(img)
                        batch_file_names.append(os.path.basename(file_path))
                except:
                    continue
            
            if len(batch_images) == 0:
                continue
                
            batch_images = np.array(batch_images)
            
            # 计算重建误差
            errors, reconstructed = dae.calculate_reconstruction_error(batch_images)
            
            all_errors.extend(errors)
            all_images.extend(batch_images)
            all_file_names.extend(batch_file_names)
            
            print(f"已处理 {i+len(batch_images)}/{len(all_files)} 个文件")
        
        all_errors = np.array(all_errors)
        all_images = np.array(all_images)
        
        print(f"检测完成，总共处理 {len(all_errors)} 个有效样本")
        
        # 直接使用所有样本，获取误差最高的500个
        top_k = 500
        if len(all_errors) > top_k:
            top_indices = np.argsort(all_errors)[-top_k:]
            filtered_errors = all_errors[top_indices]
            filtered_images = all_images[top_indices]
            filtered_files = [all_file_names[i] for i in top_indices]
        else:
            # 如果样本数少于top_k，返回所有样本（按误差降序）
            sorted_indices = np.argsort(all_errors)[::-1]
            filtered_errors = all_errors[sorted_indices]
            filtered_images = all_images[sorted_indices]
            filtered_files = [all_file_names[i] for i in sorted_indices]
        
        # 保存结果
        results = {
            'errors': filtered_errors,
            'file_names': filtered_files,
            'statistics': {
                'mean_error': np.mean(all_errors),
                'std_error': np.std(all_errors),
                'max_error': np.max(all_errors),
                'min_error': np.min(all_errors),
                'percentile_95': np.percentile(all_errors, 95),
                'percentile_99': np.percentile(all_errors, 99)
            }
        }
        
        np.save(os.path.join(args.output_dir, 'anomaly_results.npy'), results)
        
        # 保存排序后的异常样本列表
        with open(os.path.join(args.output_dir, 'top_anomalies.txt'), 'w') as f:
            f.write("Rank\tFilename\tAnomaly_Score\n")
            for i, (fname, error) in enumerate(zip(filtered_files, filtered_errors)):
                f.write(f"{i+1}\t{fname}\t{error:.6f}\n")
        
        # 重建最异常的样本用于可视化
        top_indices = np.argsort(filtered_errors)[-50:]
        top_images = filtered_images[top_indices]
        top_images_norm = top_images.astype('float32') / 255.0
        top_reconstructed = dae.model.predict(top_images_norm, verbose=0)
        
        # 可视化结果
        visualize_results(
            top_images_norm, 
            top_reconstructed, 
            filtered_errors[top_indices],
            os.path.join(args.output_dir, 'top_anomalies_visualization.png'),
            n_display=20
        )
        
        # 保存异常分数分布图
        plt.figure(figsize=(10, 6))
        plt.hist(all_errors, bins=100, alpha=0.7, label='All Samples')
        plt.hist(filtered_errors, bins=50, alpha=0.7, label='Filtered Anomalies')
        plt.axvline(np.percentile(all_errors, 95), color='red', linestyle='--', 
                   label='95% Percentile')
        plt.axvline(np.percentile(all_errors, 99), color='orange', linestyle='--', 
                   label='99% Percentile')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.yscale('log')
        plt.savefig(os.path.join(args.output_dir, 'error_distribution.png'))
        plt.close()
        
        print(f"异常检测完成！结果已保存到: {args.output_dir}")
        print(f"发现 {len(filtered_files)} 个潜在异常样本")
        print(f"平均重建误差: {np.mean(all_errors):.4f}")
        print(f"异常阈值(95%): {np.percentile(all_errors, 95):.4f}")
        print(f"异常阈值(99%): {np.percentile(all_errors, 99):.4f}")


if __name__ == "__main__":
    main()
