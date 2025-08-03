# MyMusic 智能音乐推荐系统

一个基于AI的全栈音乐推荐平台，集成了移动端应用、后端API服务和智能推荐引擎。

## 🎵 项目简介

MyMusic是一个现代化的音乐推荐系统，采用微服务架构，提供个性化音乐推荐服务。系统包含三个核心模块：

- **移动端应用** (HarmonyOS/Android)
- **后端API服务** (Node.js + Koa)  
- **AI推荐引擎** (Python + TensorFlow)

## 🏗️ 项目架构

```
withRecommend/
├── myMusic/           # 鸿蒙移动端应用
├── openKoa/          # Node.js后端API服务
├── recommend/        # Python AI推荐系统
├── music.sql         # MySQL数据库初始化脚本
└── README.md         # 项目文档
```

## ✨ 核心功能

### 移动端应用 (myMusic)
- 🎧 音乐播放和控制
- 👤 用户注册和登录
- 📱 现代化移动UI界面
- 🔄 实时推荐更新
- ❤️ 收藏和播放列表管理

### 后端API服务 (openKoa)
- 🔐 用户认证和授权 (JWT)
- 🎵 音乐资源管理
- 👥 用户数据管理
- 📊 播放记录统计
- 🖼️ 文件上传和管理

### AI推荐引擎 (recommend)
- 🧠 双塔神经网络架构
- 📈 实时个性化推荐
- 🔮 协同过滤算法
- 📊 用户行为分析
- 🎯 智能推荐API

## 🛠️ 技术栈

### 前端技术
- **开发框架**: HarmonyOS/ArkTS
- **UI组件**: ArkUI组件库
- **状态管理**: 本地状态管理
- **网络请求**: HTTP客户端

### 后端技术
- **运行环境**: Node.js 16+
- **Web框架**: Koa.js 2.x
- **数据库**: MySQL 8.0
- **ORM**: Sequelize
- **认证**: JWT (JSON Web Token)
- **文件处理**: Multer

### AI推荐技术
- **机器学习**: TensorFlow 2.15
- **数据处理**: Pandas, NumPy
- **神经网络**: 双塔架构 (LSTM + Dense)
- **数据库连接**: PyMySQL

### 数据库设计
- **用户管理**: 用户表、管理员表
- **音乐数据**: 歌曲表、歌手表、歌单表
- **行为数据**: 收藏表、评分表、播放记录
- **系统数据**: 轮播图表

## 🚀 快速开始

### 环境要求

- **Node.js**: 16.0.0+
- **Python**: 3.8+
- **MySQL**: 8.0+
- **DevEco Studio**: 4.0+ (鸿蒙开发)

### 1. 克隆项目

```bash
git clone <repository-url>
cd withRecommend
```

### 2. 数据库初始化

```bash
# 创建MySQL数据库
mysql -u root -p

# 在MySQL命令行中执行
CREATE DATABASE music;
USE music;
SOURCE music.sql;
```

### 3. 后端服务部署

```bash
cd openKoa

# 安装依赖
npm install

# 配置数据库连接
# 编辑 config/db.js 文件

# 启动开发服务器
npm run dev

# 生产环境启动
npm start
```

### 4. AI推荐系统部署

```bash
cd recommend

# 安装Python依赖
pip install -r requirements.txt

# 检查系统状态
python main.py check

# 生成测试数据（首次运行）
python main.py generate

# 训练模型
python main.py train

# 生成推荐
python main.py predict
```

### 5. 移动端应用开发

```bash
cd myMusic

# 使用DevEco Studio打开项目
# 1. 打开DevEco Studio
# 2. 选择"Open Project"
# 3. 选择myMusic目录
# 4. 配置签名和构建
# 5. 运行到设备或模拟器
```

## 📊 数据库结构

### 核心数据表

| 表名 | 描述 | 主要字段 |
|------|------|----------|
| consumers | 用户信息 | id, username, password, email, phone |
| singers | 歌手信息 | id, name, sex, pic, location |
| songs | 歌曲信息 | id, singer_id, name, pic, lyric, url |
| song_lists | 歌单信息 | id, title, pic, introduction, style |
| collects | 用户收藏 | id, user_id, song_id |
| ranks | 歌单评分 | id, song_list_id, consumer_id, score |
| swipers | 轮播图 | id, title, url, imgurl |
| admins | 管理员 | id, username, password |

### 推荐系统数据表

| 表名 | 描述 | 主要字段 |
|------|------|----------|
| user_plays | 播放记录 | user_id, song_id, play_count, timestamp |

## 🔧 配置说明

### 后端配置 (openKoa/config/db.js)

```javascript
module.exports = {
  host: 'localhost',
  port: 3306,
  database: 'music',
  username: 'root',
  password: 'your_password',
  dialect: 'mysql'
}
```

### 推荐系统配置 (recommend/src/config.py)

```python
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'music',
    'user': 'root',
    'password': 'your_password'
}
```

## 📱 API接口文档

### 用户认证接口

- `POST /api/login` - 用户登录
- `POST /api/register` - 用户注册
- `GET /api/user/info` - 获取用户信息

### 音乐管理接口

- `GET /api/songs` - 获取歌曲列表
- `GET /api/singers` - 获取歌手列表
- `GET /api/songlists` - 获取歌单列表
- `POST /api/collect` - 添加收藏
- `DELETE /api/collect/:id` - 取消收藏

### 推荐系统接口

- `GET /api/recommend/:userId` - 获取用户推荐
- `POST /api/feedback` - 用户反馈

## 🧠 AI推荐算法

### 双塔神经网络架构

```
用户塔 (User Tower)              歌曲塔 (Item Tower)
┌─────────────────┐              ┌─────────────────┐
│   用户特征输入    │              │   歌曲特征输入    │
│                 │              │                 │
│ ┌─────────────┐ │              │ ┌─────────────┐ │
│ │ Embedding   │ │              │ │ Embedding   │ │
│ └─────────────┘ │              │ └─────────────┘ │
│        │        │              │        │        │
│ ┌─────────────┐ │              │ ┌─────────────┐ │
│ │   LSTM      │ │              │ │   Dense     │ │
│ └─────────────┘ │              │ └─────────────┘ │
│        │        │              │        │        │
│ ┌─────────────┐ │              │ ┌─────────────┐ │
│ │   Dense     │ │              │ │   Dense     │ │
│ └─────────────┘ │              │ └─────────────┘ │
└─────────────────┘              └─────────────────┘
         │                                │
         └────────────┬───────────────────┘
                      │
              ┌─────────────┐
              │ 相似度计算   │
              │(Cosine)     │
              └─────────────┘
```

### 推荐流程

1. **特征提取**: 提取用户行为特征和歌曲内容特征
2. **向量编码**: 通过双塔网络生成用户和歌曲的向量表示
3. **相似度计算**: 计算用户向量与歌曲向量的余弦相似度
4. **排序推荐**: 根据相似度分数排序，返回Top-N推荐

## 🔒 安全说明

### 用户认证
- 使用JWT令牌进行用户认证
- 密码采用MD5加密存储
- API接口支持CORS跨域访问

### 数据安全
- 敏感信息加密存储
- SQL注入防护
- 文件上传类型限制

## 📈 性能优化

### 后端优化
- 数据库连接池
- Redis缓存 (可选)
- 静态资源CDN

### 推荐系统优化
- 模型缓存机制
- 批量预测优化
- 增量训练支持

## 🧪 测试

### 后端测试

```bash
cd openKoa
npm test
```

### 推荐系统测试

```bash
cd recommend
python -m pytest tests/
```

## 📦 部署

### Docker部署 (推荐)

```bash
# 构建和启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

### 传统部署

1. **数据库部署**: 安装MySQL并导入数据
2. **后端部署**: 使用PM2部署Node.js应用
3. **推荐系统部署**: 使用Gunicorn部署Python服务
4. **移动端部署**: 打包并发布到应用商店

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目链接: [GitHub Repository]

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户！

---

**注意**: 这是一个学习和演示项目，如需商业使用请确保遵循相关法律法规和版权要求。
