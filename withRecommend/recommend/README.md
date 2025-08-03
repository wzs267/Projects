# MyMusic推荐系统

基于双塔神经网络的音乐推荐系统，支持个性化推荐和实时预测。

## 功能特性

- 🎵 双塔神经网络架构（用户塔LSTM + 歌曲塔Dense）
- 📊 支持mymusic数据库集成
- 🔮 实时推荐预测
- 🌐 JSON API接口
- 💾 模型持久化存储

## 环境要求

- Python 3.8+
- MySQL 5.7+
- 操作系统：Windows/Linux/macOS

## 快速开始

### 1. 项目完整性检查

首先检查项目文件完整性：

```bash
# 检查系统状态
python main.py check
```

预期输出应该显示：
- ✅ 数据目录、SQL脚本目录、模型目录存在
- ✅ 消费者数据和播放记录数据完整
- ✅ 训练模型和映射文件存在
- 📊 数据库连接状态

### 2. 环境配置

安装Python依赖：

```bash
pip install -r requirements.txt
```

依赖包列表：
- tensorflow==2.15.0
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- pymysql==1.1.0

### 3. 数据库初始化

⚠️ **重要：如果你的本地数据库还是未修改的原始版本，需要执行以下步骤**

#### 3.1 确认数据库状态
```bash
# 检查当前数据库状态
python main.py check
```

如果显示数据库连接失败或表不存在，需要初始化数据库。

#### 3.2 创建数据库（如果不存在）
```sql
-- 连接MySQL并创建数据库
CREATE DATABASE IF NOT EXISTS mymusic;
USE mymusic;
```

#### 3.3 执行数据初始化脚本

**方法一：使用MySQL命令行**
```bash
# 进入MySQL命令行
mysql -u root -p

# 选择数据库
USE mymusic;

# 执行消费者数据脚本
SOURCE sql_scripts/mymusic/insert_consumers.sql;

# 执行播放记录脚本（包含表创建）
SOURCE sql_scripts/mymusic/create_and_insert_user_plays.sql;
```

#### 3.4 验证数据库初始化
```bash
python main.py check
```

应该显示：
- ✅ consumers: 102 条记录
- ✅ songs: 10 条记录  
- ✅ singers: 6 条记录
- ✅ user_plays: 5000 条记录

### 4. 模型训练（可选）

如果需要重新训练模型：

```bash
python main.py train
```

训练完成后会显示：
- 模型准确率（预期：~81%）
- 模型保存路径：models/mymusic_twin_tower.keras

### 5. 生成推荐

#### 5.1 命令行推荐
```bash
# 生成推荐示例
python main.py predict
```

#### 5.2 API接口调用
```bash
# 为用户3推荐5首歌曲
python scripts/api/simple_recommend.py 3 5

# 为用户10推荐3首歌曲  
python scripts/api/simple_recommend.py 10 3
```

## 项目结构

```
recommend/
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖包
├── README.md              # 项目文档
├── models/                # 训练好的模型
│   └── mymusic_twin_tower.keras
├── data/                  # 数据文件
│   ├── mymusic_generated/ # 生成的数据
│   └── mymusic_processed/ # 处理后的数据
├── src/                   # 核心源码
│   ├── mymusic_config.py  # 配置文件
│   ├── train_mymusic.py   # 模型训练
│   └── predict_mymusic.py # 推荐预测
├── scripts/               # 工具脚本
│   ├── api/              # API接口
│   │   └── simple_recommend.py
│   ├── data_generation/  # 数据生成
│   └── tools/           # 工具脚本
└── sql_scripts/          # SQL脚本
    └── mymusic/         # mymusic数据库脚本
```

## 数据库结构

### 主要相关表结构：

1. **consumers** - 用户表
   - 字段：id, username, sex, pic, birth, introduction, location, avator
   - 数据：102个用户（ID: 3-102）

2. **songs** - 歌曲表  
   - 字段：id, singer_id, name, introduction, create_time, update_time, pic, lyric, url
   - 数据：10首歌曲

3. **singers** - 歌手表
   - 字段：id, name, sex, pic, birth, location, introduction
   - 数据：6位歌手

4. **user_plays** - 播放记录表（新增）
   - 字段：user_id, song_id, play_count, last_play_time
   - 数据：5000条播放记录

## API接口

### 推荐API

**调用方式：**
```bash
python scripts/api/simple_recommend.py [用户ID] [推荐数量]
```

**返回格式：**
```json
{
    "success": true,
    "user_id": "3",
    "recommendations": [
        {
            "song_id": 5,
            "score": 0.513,
            "song_info": null
        },
        {
            "song_id": 2,
            "score": 0.457,
            "song_info": null
        }
    ],
    "count": 5
}
```

## 集成到现有项目

### 后端集成（Koa.js）

在你的Koa.js项目中添加推荐接口：

```javascript
// 在controller/api.js中添加
async getRecommendations(ctx) {
    const { userId, count = 5 } = ctx.request.query;
    
    try {
        const { exec } = require('child_process');
        const result = await new Promise((resolve, reject) => {
            exec(`python ../recommend/scripts/api/simple_recommend.py ${userId} ${count}`, 
                (error, stdout, stderr) => {
                if (error) reject(error);
                else resolve(JSON.parse(stdout));
            });
        });
        
        ctx.body = {
            code: 200,
            message: '获取推荐成功',
            data: result
        };
    } catch (error) {
        ctx.body = {
            code: 500,
            message: '推荐系统错误',
            error: error.message
        };
    }
}
```

### 前端集成（HarmonyOS）

在HarmonyOS应用中调用推荐API：

```typescript
// 推荐服务类
class RecommendationService {
    async getRecommendations(userId: number, count: number = 5): Promise<any> {
        try {
            const response = await this.httpUtil.request({
                method: 'GET',
                url: `${this.serverConfig.baseUrl}/api/recommendations`,
                params: { userId, count }
            });
            return response.data;
        } catch (error) {
            console.error('获取推荐失败:', error);
            return [];
        }
    }
}
```

## 性能指标

- **训练准确率**：81.26%
- **嵌入维度**：64
- **推荐延迟**：<3秒
- **数据规模**：100用户 + 10歌曲 + 5000播放记录
- **模型大小**：~2MB

## 故障排除

### 常见问题

1. **数据库连接失败**
   ```
   ⚠️ 数据库连接检查失败: No module named 'mysql'
   ```
   **解决方案**：安装PyMySQL
   ```bash
   pip install pymysql
   ```

2. **模型文件不存在**
   ```
   模型文件不存在: models/mymusic_twin_tower.keras
   ```
   **解决方案**：重新训练模型
   ```bash
   python main.py train
   ```

3. **映射文件缺失**
   ```
   映射文件不存在: data/mymusic_processed/mappings.npy
   ```
   **解决方案**：重新训练模型会自动生成映射文件

4. **用户ID不存在**
   ```
   用户 user_999 不在映射中
   ```
   **解决方案**：使用有效的用户ID范围（3-102）

## 许可证

MIT License
