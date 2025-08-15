# 投标平台项目结构

## 📁 项目目录结构

```
bidding-platform/
├── 📁 config/              # 配置文件
│   └── database.js         # 数据库连接配置
├── 📁 docs/                # 项目文档
│   ├── deployment.md       # 部署指南
│   └── user-guide.md      # 用户使用指南
├── 📁 middleware/          # Express中间件
│   └── auth.js            # JWT认证中间件
├── 📁 models/              # 数据库模型
│   ├── Bid.js             # 投标模型
│   ├── Project.js         # 项目模型
│   └── User.js            # 用户模型
├── 📁 public/              # 静态文件目录
│   ├── 📁 css/            # 样式文件
│   │   └── style.css      # 主要样式
│   ├── 📁 js/             # JavaScript文件
│   │   └── app.js         # 前端主应用逻辑
│   └── index.html         # 主页面
├── 📁 routes/              # 路由文件
│   ├── admin.js           # 管理员路由
│   ├── auth.js            # 认证路由（旧版）
│   ├── newAuth.js         # 认证路由（新版）
│   ├── bids.js            # 投标路由
│   ├── projects.js        # 项目路由
│   └── users.js           # 用户路由
├── 📁 scripts/             # 工具脚本
│   ├── init-db.js         # 数据库初始化脚本
│   └── reset-data.js      # 数据重置脚本
├── 📁 .vscode/            # VS Code配置
├── .env                   # 环境变量
├── .env.example          # 环境变量示例
├── init-db.bat           # Windows数据库初始化批处理
├── package.json          # 项目配置和依赖
├── README.md             # 项目说明
├── server.js             # 服务器入口文件
├── start.bat             # Windows启动脚本
└── SUCCESS.md            # 项目完成说明
```

## 🚀 核心文件说明

### 后端核心文件
- **server.js** - Express服务器入口，路由配置
- **config/database.js** - MySQL数据库连接池配置
- **middleware/auth.js** - JWT认证中间件

### 数据模型
- **models/User.js** - 用户管理（注册、登录、权限）
- **models/Project.js** - 项目管理（发布、编辑、状态控制）
- **models/Bid.js** - 投标管理（投标、接受、状态更新）

### API路由
- **routes/newAuth.js** - 用户认证接口（登录、注册、token验证）
- **routes/projects.js** - 项目管理接口（CRUD、状态管理）
- **routes/bids.js** - 投标管理接口（投标、接受、拒绝）
- **routes/users.js** - 用户信息管理接口
- **routes/admin.js** - 管理员功能接口

### 前端文件
- **public/index.html** - 单页面应用主页
- **public/js/app.js** - 前端核心逻辑（约2700行）
- **public/css/style.css** - 响应式UI样式

### 工具脚本
- **scripts/init-db.js** - 创建数据库表结构
- **scripts/reset-data.js** - 重置测试数据

## 🛡️ 配置文件
- **.env** - 环境变量（数据库连接、JWT密钥等）
- **package.json** - 项目依赖和脚本配置

## 📋 已清理的文件
以下临时和测试文件已被清理：
- 所有 test-*.js 测试脚本
- 临时调试文件（DEBUG_*.md、check_password.js等）
- 数据迁移脚本（migrate-*.js）
- 一次性设置脚本（setup-test-data.js等）
- 测试HTML页面（test-*.html）

## 🎯 项目特点
- **清洁的代码结构** - 分离关注点，模块化设计
- **完整的功能实现** - 用户管理、项目发布、投标流程
- **响应式设计** - 支持PC和移动端
- **安全性** - JWT认证、参数验证、SQL注入防护
- **易于维护** - 清晰的文件组织和注释