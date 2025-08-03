#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音乐推荐系统流程检查器 (新版本 - 适应重组后的项目结构)
检查每个步骤的执行状态和数据完整性
"""

import os
import json
import numpy as np
from datetime import datetime

class WorkflowChecker:
    def __init__(self):
        self.status = {}
        self.errors = []
        
    def print_header(self, title):
        print(f"\n{'='*50}")
        print(f"🔍 {title}")
        print(f"{'='*50}")
    
    def check_step(self, step_name, condition, details=""):
        status = "✅ 通过" if condition else "❌ 失败"
        self.status[step_name] = condition
        print(f"{status} {step_name}")
        if details:
            print(f"   {details}")
        if not condition:
            self.errors.append(step_name)
    
    def check_environment(self):
        """检查运行环境"""
        self.print_header("环境检查")
        
        # 检查Python模块
        try:
            import numpy
            numpy_version = numpy.__version__
            self.check_step("NumPy", True, f"版本: {numpy_version}")
        except ImportError:
            self.check_step("NumPy", False, "未安装numpy")
        
        try:
            import tensorflow as tf
            tf_version = tf.__version__
            self.check_step("TensorFlow", True, f"版本: {tf_version}")
        except ImportError:
            self.check_step("TensorFlow", False, "未安装tensorflow")
        
        # 检查配置文件
        config_exists = os.path.exists('src/config.py')
        self.check_step("配置文件", config_exists, "src/config.py")
        
        if config_exists:
            try:
                import sys
                sys.path.append('src')
                from config import Config
                cfg = Config()
                self.check_step("配置文件格式", True, f"用户数: {cfg.NUM_USERS}, 歌曲数: {cfg.NUM_SONGS}")
            except Exception as e:
                self.check_step("配置文件格式", False, f"导入错误: {str(e)}")
    
    def check_data_generation(self):
        """检查数据生成步骤"""
        self.print_header("数据生成检查")
        
        # 检查原始数据
        raw_data_exists = os.path.exists('data/raw/user_plays.json')
        self.check_step("原始数据文件", raw_data_exists, "data/raw/user_plays.json")
        
        if raw_data_exists:
            try:
                with open('data/raw/user_plays.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.check_step("原始数据格式", True, f"包含 {len(data)} 条记录")
                    
                    # 检查数据字段
                    if data and len(data) > 0:
                        first_record = data[0]
                        expected_fields = ['user_id', 'song_id', 'duration']
                        has_all_fields = all(field in first_record for field in expected_fields)
                        self.check_step("数据字段完整", has_all_fields, f"字段: {list(first_record.keys())}")
            except Exception as e:
                self.check_step("原始数据格式", False, f"读取错误: {str(e)}")
        
        # 检查测试数据
        test_files = ['data/test/users.json', 'data/test/songs.json', 'data/test/user_plays.json']
        for test_file in test_files:
            exists = os.path.exists(test_file)
            name = os.path.basename(test_file)
            self.check_step(f"测试数据 {name}", exists, test_file)
        
        # 检查SQL脚本目录
        sql_dir = 'sql/data'
        sql_dir_exists = os.path.exists(sql_dir)
        self.check_step("SQL脚本目录", sql_dir_exists, sql_dir)
        
        if sql_dir_exists:
            sql_files = [f for f in os.listdir(sql_dir) if f.endswith('.sql')]
            self.check_step("SQL文件数量", len(sql_files) > 0, f"找到 {len(sql_files)} 个SQL文件")
    
    def check_preprocessing(self):
        """检查数据预处理步骤"""
        self.print_header("数据预处理检查")
        
        # 检查处理后的数据目录
        processed_dir = 'data/processed'
        processed_exists = os.path.exists(processed_dir)
        self.check_step("处理数据目录", processed_exists, processed_dir)
        
        # 检查交互矩阵
        interactions_file = 'data/processed/interactions.npy'
        interactions_exists = os.path.exists(interactions_file)
        self.check_step("交互矩阵文件", interactions_exists, interactions_file)
        
        if interactions_exists:
            try:
                interactions = np.load(interactions_file)
                self.check_step("交互矩阵格式", True, f"矩阵形状: {interactions.shape}")
                
                # 检查值域
                min_val, max_val = interactions.min(), interactions.max()
                self.check_step("交互矩阵值域", True, f"值域: [{min_val:.3f}, {max_val:.3f}]")
            except Exception as e:
                self.check_step("交互矩阵格式", False, f"加载错误: {str(e)}")
        
        # 检查映射文件
        mappings_file = 'data/processed/mappings.npy'
        mappings_exists = os.path.exists(mappings_file)
        self.check_step("映射文件", mappings_exists, mappings_file)
        
        if mappings_exists:
            try:
                mappings = np.load(mappings_file, allow_pickle=True)
                if isinstance(mappings, np.ndarray) and mappings.ndim == 0:
                    # 单个映射字典
                    mapping_dict = mappings.item()
                    self.check_step("用户映射", len(mapping_dict) > 0, f"用户数: {len(mapping_dict)}")
                    self.check_step("歌曲映射", True, "数据存储在交互矩阵中")
                else:
                    # 多个映射的字典格式
                    user_mapping = mappings.get('user_to_idx', {})
                    song_mapping = mappings.get('song_to_idx', {})
                    self.check_step("用户映射", len(user_mapping) > 0, f"用户数: {len(user_mapping)}")
                    self.check_step("歌曲映射", len(song_mapping) > 0, f"歌曲数: {len(song_mapping)}")
            except Exception as e:
                self.check_step("映射文件格式", False, f"加载错误: {str(e)}")
    
    def check_training(self):
        """检查模型训练步骤"""
        self.print_header("模型训练检查")
        
        # 检查模型目录
        models_dir = 'models'
        models_exists = os.path.exists(models_dir)
        self.check_step("模型目录", models_exists, models_dir)
        
        # 检查模型文件
        model_file = 'models/twin_tower.keras'
        model_exists = os.path.exists(model_file)
        self.check_step("模型文件", model_exists, model_file)
        
        if model_exists:
            # 检查文件大小
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            self.check_step("模型文件大小", file_size > 0.1, f"{file_size:.2f} MB")
            
            # 检查创建时间
            create_time = datetime.fromtimestamp(os.path.getctime(model_file))
            self.check_step("模型创建时间", True, create_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # 检查训练脚本
        train_script = 'src/train.py'
        train_exists = os.path.exists(train_script)
        self.check_step("训练脚本", train_exists, train_script)
    
    def check_prediction(self):
        """检查预测功能"""
        self.print_header("预测功能检查")
        
        # 检查预测脚本
        predict_script = 'src/predict.py'
        predict_exists = os.path.exists(predict_script)
        self.check_step("预测脚本", predict_exists, predict_script)
        
        # 检查工具目录
        utils_dir = 'src/utils'
        utils_exists = os.path.exists(utils_dir)
        self.check_step("工具目录", utils_exists, utils_dir)
        
        # 尝试导入预测函数
        if predict_exists:
            try:
                import sys
                sys.path.append('src')
                from predict import predict_for_user
                self.check_step("预测函数导入", True, "成功导入 predict_for_user")
            except ImportError as e:
                self.check_step("预测函数导入", False, f"导入错误: {str(e)}")
    
    def check_deployment(self):
        """检查部署就绪状态"""
        self.print_header("部署就绪检查")
        
        # 核心文件
        required_files = [
            'src/config.py', 'src/train.py', 'src/predict.py', 'requirements.txt',
            'scripts/data_generation/data_generator.py', 'sql/schema/database_schema.sql'
        ]
        
        for file_path in required_files:
            exists = os.path.exists(file_path)
            filename = os.path.basename(file_path)
            self.check_step(f"必需文件 {filename}", exists, file_path)
        
        # 文档文件
        doc_files = ['docs/DEPLOYMENT_GUIDE.md', 'docs/DATABASE_INIT_GUIDE.md', 'docs/DETAILED_WORKFLOW.md']
        for doc_file in doc_files:
            exists = os.path.exists(doc_file)
            filename = os.path.basename(doc_file)
            self.check_step(f"文档文件 {filename}", exists, doc_file)
        
        # 自动化脚本
        auto_scripts = ['scripts/database/init_database.bat', 'scripts/database/run_all.bat']
        for script_file in auto_scripts:
            exists = os.path.exists(script_file)
            filename = os.path.basename(script_file)
            self.check_step(f"自动化脚本 {filename}", exists, script_file)
    
    def print_summary(self):
        """打印检查总结"""
        self.print_header("检查总结")
        
        total_checks = len(self.status)
        passed_checks = sum(self.status.values())
        failed_checks = total_checks - passed_checks
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        print(f"📊 总检查项: {total_checks}")
        print(f"✅ 通过项目: {passed_checks}")
        print(f"❌ 失败项目: {failed_checks}")
        print(f"🎯 成功率: {success_rate:.1f}%")
        
        if self.errors:
            print("\n⚠️  需要关注的问题:")
            for error in self.errors:
                print(f"   - {error}")
        
        print("\n💡 建议:")
        if failed_checks > 0:
            print("   - 初始化数据库: 运行 scripts/database/init_database.bat 或相应SQL脚本")
            print("   - 生成测试数据: 运行 python scripts/data_generation/data_generator.py")
            print("   - 完整运行流程请参考: docs/DETAILED_WORKFLOW.md")
        else:
            print("   - 🎉 系统状态良好，所有检查项目都通过！")
    
    def run_all_checks(self):
        """运行所有检查"""
        print("🎵 音乐推荐系统流程检查器")
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.check_environment()
        self.check_data_generation()
        self.check_preprocessing()
        self.check_training()
        self.check_prediction()
        self.check_deployment()
        self.print_summary()

if __name__ == "__main__":
    checker = WorkflowChecker()
    checker.run_all_checks()
