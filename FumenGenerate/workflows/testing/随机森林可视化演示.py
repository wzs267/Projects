#!/usr/bin/env python3
"""
随机森林决策过程可视化演示
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def 演示单棵决策树的学习过程():
    """详细展示一棵决策树是如何学习的"""
    print("🌳 单棵决策树的学习过程演示")
    print("=" * 40)
    
    # 训练数据
    训练特征 = np.array([
        [0.8, 2500],  # [RMS能量, 频谱质心] → 有音符
        [0.2, 1800],  # [RMS能量, 频谱质心] → 无音符
        [0.9, 3200],  # [RMS能量, 频谱质心] → 有音符
        [0.1, 1000],  # [RMS能量, 频谱质心] → 无音符
        [0.6, 2200]   # [RMS能量, 频谱质心] → 有音符
    ])
    训练标签 = np.array([1, 0, 1, 0, 1])
    特征名 = ['RMS能量', '频谱质心']
    
    print("📚 训练数据:")
    for i, (特征, 标签) in enumerate(zip(训练特征, 训练标签)):
        结果 = "有音符" if 标签 == 1 else "无音符"
        print(f"   样本{i+1}: RMS={特征[0]:.1f}, 频谱质心={特征[1]:.0f} → {结果}")
    
    # 创建单棵决策树
    决策树 = DecisionTreeClassifier(max_depth=2, random_state=42)
    决策树.fit(训练特征, 训练标签)
    
    print(f"\n🌳 决策树的学习结果:")
    
    # 模拟决策树的决策过程
    def 模拟决策过程(样本特征):
        RMS = 样本特征[0]
        频谱质心 = 样本特征[1]
        
        print(f"   📥 输入: RMS={RMS:.1f}, 频谱质心={频谱质心:.0f}")
        
        # 根据实际决策树的分割逻辑
        if RMS <= 0.5:  # 第一层分割
            print(f"   🤔 第1层判断: RMS={RMS:.1f} <= 0.5? 是的")
            print(f"   🎯 决策: 无音符 (因为能量太低)")
            return 0
        else:
            print(f"   🤔 第1层判断: RMS={RMS:.1f} > 0.5? 是的")
            if 频谱质心 <= 2400:  # 第二层分割
                print(f"   🤔 第2层判断: 频谱质心={频谱质心:.0f} <= 2400? 是的")
                print(f"   🎯 决策: 有音符 (中等频率)")
                return 1
            else:
                print(f"   🤔 第2层判断: 频谱质心={频谱质心:.0f} > 2400? 是的")
                print(f"   🎯 决策: 有音符 (高频率)")
                return 1
    
    # 测试新样本
    新样本 = np.array([0.7, 2800])
    print(f"\n🧪 测试新样本:")
    预测结果 = 模拟决策过程(新样本)
    
    # 验证实际结果
    实际预测 = 决策树.predict([新样本])[0]
    print(f"   ✅ 实际API结果: {实际预测}")
    
    return 决策树

def 演示随机森林的集体决策():
    """展示3棵树的集体决策过程"""
    print(f"\n🌲 随机森林的集体决策演示")
    print("=" * 35)
    
    # 使用相同的训练数据
    训练特征 = np.array([
        [0.8, 2500], [0.2, 1800], [0.9, 3200], 
        [0.1, 1000], [0.6, 2200]
    ])
    训练标签 = np.array([1, 0, 1, 0, 1])
    
    # 创建随机森林
    随机森林 = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
    随机森林.fit(训练特征, 训练标签)
    
    # 测试样本
    测试样本 = np.array([[0.7, 2800]])
    
    print(f"📥 测试样本: RMS=0.7, 频谱质心=2800")
    print(f"\n👥 询问每棵树的意见:")
    
    # 获取每棵树的预测
    个别预测 = []
    for i, 单棵树 in enumerate(随机森林.estimators_):
        预测 = 单棵树.predict(测试样本)[0]
        概率 = 单棵树.predict_proba(测试样本)[0]
        个别预测.append(预测)
        
        结果文字 = "有音符" if 预测 == 1 else "无音符"
        print(f"   🌳 决策树{i+1}: {结果文字} (概率: 无音符{概率[0]:.2f}, 有音符{概率[1]:.2f})")
    
    # 统计投票
    有音符票数 = sum(个别预测)
    总票数 = len(个别预测)
    
    print(f"\n🗳️  投票统计:")
    print(f"   有音符: {有音符票数}票")
    print(f"   无音符: {总票数 - 有音符票数}票")
    print(f"   总计: {总票数}票")
    
    # 最终决策
    森林概率 = 有音符票数 / 总票数
    最终决策 = "有音符" if 森林概率 > 0.5 else "无音符"
    
    print(f"\n🎯 随机森林最终决策:")
    print(f"   概率: {森林概率:.3f}")
    print(f"   决策: {最终决策}")
    
    # 验证API结果
    api概率 = 随机森林.predict_proba(测试样本)[0][1]
    api预测 = 随机森林.predict(测试样本)[0]
    api结果 = "有音符" if api预测 == 1 else "无音符"
    
    print(f"\n✅ API验证:")
    print(f"   predict_proba: {api概率:.3f}")
    print(f"   predict: {api结果}")

def 演示随机性的作用():
    """展示随机性如何产生不同的树"""
    print(f"\n🎲 随机性的作用演示")
    print("=" * 25)
    
    训练特征 = np.array([
        [0.8, 2500], [0.2, 1800], [0.9, 3200], 
        [0.1, 1000], [0.6, 2200], [0.7, 2000],
        [0.3, 1500], [0.5, 2800]
    ])
    训练标签 = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    
    print("🎯 演示Bootstrap采样的效果:")
    
    # 模拟3次Bootstrap采样
    np.random.seed(42)
    for i in range(3):
        # Bootstrap采样：有放回地抽取样本
        样本索引 = np.random.choice(len(训练特征), size=len(训练特征), replace=True)
        bootstrap样本 = 训练特征[样本索引]
        bootstrap标签 = 训练标签[样本索引]
        
        print(f"\n   🌳 决策树{i+1}的训练数据 (Bootstrap采样):")
        print(f"   抽取的样本索引: {样本索引}")
        
        # 统计每个原始样本被选中的次数
        选中次数 = {}
        for idx in 样本索引:
            选中次数[idx] = 选中次数.get(idx, 0) + 1
        
        print(f"   样本分布: ", end="")
        for idx in range(len(训练特征)):
            次数 = 选中次数.get(idx, 0)
            print(f"样本{idx}×{次数} ", end="")
        print()
        
        # 训练这棵树
        单棵树 = DecisionTreeClassifier(max_depth=2, random_state=i)
        单棵树.fit(bootstrap样本, bootstrap标签)
        
        # 测试预测差异
        测试样本 = np.array([[0.7, 2800]])
        预测 = 单棵树.predict(测试样本)[0]
        结果 = "有音符" if 预测 == 1 else "无音符"
        print(f"   对测试样本的预测: {结果}")

def 比较单树vs随机森林():
    """比较单棵树和随机森林的稳定性"""
    print(f"\n⚖️  单棵树 vs 随机森林稳定性对比")
    print("=" * 35)
    
    # 原始训练数据
    原始特征 = np.array([
        [0.8, 2500], [0.2, 1800], [0.9, 3200], 
        [0.1, 1000], [0.6, 2200]
    ])
    原始标签 = np.array([1, 0, 1, 0, 1])
    
    # 轻微修改的数据（模拟噪声）
    修改特征 = 原始特征.copy()
    修改特征[0, 0] = 0.75  # 把0.8改成0.75
    修改标签 = 原始标签.copy()
    
    测试样本 = np.array([[0.65, 2300]])
    
    print("🧪 测试数据稳定性:")
    print(f"   测试样本: RMS=0.65, 频谱质心=2300")
    
    # 单棵决策树的表现
    print(f"\n🌳 单棵决策树:")
    树1 = DecisionTreeClassifier(max_depth=2, random_state=42)
    树1.fit(原始特征, 原始标签)
    预测1 = 树1.predict(测试样本)[0]
    
    树2 = DecisionTreeClassifier(max_depth=2, random_state=42)
    树2.fit(修改特征, 修改标签)
    预测2 = 树2.predict(测试样本)[0]
    
    结果1 = "有音符" if 预测1 == 1 else "无音符"
    结果2 = "有音符" if 预测2 == 1 else "无音符"
    
    print(f"   原始数据训练: {结果1}")
    print(f"   修改数据训练: {结果2}")
    print(f"   结果一致性: {'✅ 一致' if 预测1 == 预测2 else '❌ 不一致'}")
    
    # 随机森林的表现
    print(f"\n🌲 随机森林:")
    森林1 = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
    森林1.fit(原始特征, 原始标签)
    概率1 = 森林1.predict_proba(测试样本)[0][1]
    
    森林2 = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
    森林2.fit(修改特征, 修改标签)
    概率2 = 森林2.predict_proba(测试样本)[0][1]
    
    print(f"   原始数据训练: {概率1:.3f}概率有音符")
    print(f"   修改数据训练: {概率2:.3f}概率有音符")
    print(f"   概率差异: {abs(概率1 - 概率2):.3f}")
    print(f"   稳定性: {'✅ 更稳定' if abs(概率1 - 概率2) < 0.1 else '⚠️  有波动'}")

if __name__ == "__main__":
    # 运行所有演示
    演示单棵决策树的学习过程()
    演示随机森林的集体决策()
    演示随机性的作用()
    比较单树vs随机森林()
    
    print(f"\n🎉 总结:")
    print(f"   随机森林 = 多个不同的决策树 + 民主投票")
    print(f"   每棵树都有自己的'经验'，但集体智慧更可靠！")
