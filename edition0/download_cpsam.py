"""
CellPose SAM (CPSAM) 模型下载脚本
"""

import os
import urllib.request
from pathlib import Path

def download_cpsam_model():
    """下载CPSAM模型文件"""
    
    # 设置模型保存路径
    home_dir = Path.home()
    cellpose_dir = home_dir / ".cellpose"
    models_dir = cellpose_dir / "models"
    
    # 创建目录
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 CellPose模型目录: {models_dir}")
    
    # CPSAM模型下载URL（这些是官方模型文件）
    model_urls = {
        "cpsam": "https://www.cellpose.org/models/cpsam",
        # 备用下载地址
        "cpsam_backup": "https://github.com/MouseLand/cellpose/releases/download/3.0/cpsam"
    }
    
    model_path = models_dir / "cpsam"
    
    if model_path.exists():
        print(f"✅ CPSAM模型已存在: {model_path}")
        return model_path
    
    print("🔄 开始下载CPSAM模型...")
    
    for name, url in model_urls.items():
        try:
            print(f"📥 尝试从 {name} 下载...")
            
            # 添加User-Agent避免被服务器拒绝
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            with urllib.request.urlopen(req) as response:
                with open(model_path, 'wb') as f:
                    # 显示下载进度
                    total_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r📥 下载进度: {progress:.1f}%", end="", flush=True)
            
            print(f"\n✅ CPSAM模型下载完成: {model_path}")
            print(f"📊 文件大小: {model_path.stat().st_size / (1024*1024):.1f} MB")
            return model_path
            
        except Exception as e:
            print(f"❌ 从 {name} 下载失败: {str(e)}")
            if model_path.exists():
                model_path.unlink()  # 删除不完整的文件
            continue
    
    print("❌ 所有下载源都失败了")
    return None

def download_via_cellpose():
    """通过CellPose官方接口下载"""
    try:
        print("🔄 尝试通过CellPose官方接口下载...")
        
        from cellpose import models
        
        # 尝试创建模型（这会自动下载）
        model = models.CellposeModel(gpu=True, model_type='cyto3')
        print("✅ 通过CellPose接口下载成功")
        return True
        
    except Exception as e:
        print(f"❌ CellPose接口下载失败: {str(e)}")
        return False

def manual_download_github():
    """从GitHub手动下载"""
    try:
        print("🔄 尝试从GitHub下载CPSAM...")
        
        # GitHub releases URL
        github_urls = [
            "https://github.com/MouseLand/cellpose/releases/download/3.0.10/cpsam",
            "https://github.com/MouseLand/cellpose/releases/download/3.0.8/cpsam",
            "https://github.com/MouseLand/cellpose/releases/download/3.0.5/cpsam"
        ]
        
        home_dir = Path.home()
        models_dir = home_dir / ".cellpose" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "cpsam"
        
        for url in github_urls:
            try:
                print(f"📥 尝试下载: {url}")
                
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/octet-stream'
                })
                
                urllib.request.urlretrieve(url, model_path)
                
                # 检查文件大小
                if model_path.stat().st_size > 1024 * 1024:  # 至少1MB
                    print(f"✅ GitHub下载成功: {model_path}")
                    print(f"📊 文件大小: {model_path.stat().st_size / (1024*1024):.1f} MB")
                    return model_path
                else:
                    print("❌ 下载的文件太小，可能有问题")
                    model_path.unlink()
                    
            except Exception as e:
                print(f"❌ GitHub下载失败: {str(e)}")
                if model_path.exists():
                    model_path.unlink()
                continue
        
        return None
        
    except Exception as e:
        print(f"❌ GitHub下载过程出错: {str(e)}")
        return None

def main():
    print("🚀 开始下载CellPose SAM (CPSAM) 模型")
    
    # 方法1: 官方下载
    result = download_cpsam_model()
    if result:
        return result
    
    # 方法2: GitHub下载
    result = manual_download_github()
    if result:
        return result
    
    # 方法3: CellPose接口
    if download_via_cellpose():
        return "通过CellPose接口下载"
    
    print("❌ 所有下载方法都失败了")
    print("💡 建议:")
    print("   1. 检查网络连接")
    print("   2. 手动访问 https://github.com/MouseLand/cellpose/releases")
    print("   3. 下载cpsam文件到 ~/.cellpose/models/ 目录")
    
    return None

if __name__ == "__main__":
    main()
