# 招投标平台

基于Node.js + Express + MySQL的全栈Web应用，支持用户注册登录、需求发布管理、投标系统等功能。

## 功能特性

### 用户系统
- 用户注册与登录
- 密码加密存储
- JWT Token认证
- 用户角色管理（发布方/投标方）

### 需求管理
- 发布需求
- 需求列表展示
- 需求详情查看
- 需求编辑和删除

### 投标系统
- 投标功能
- 投标记录管理
- 投标状态跟踪
- 中标通知

### 管理功能
- 用户管理
- 需求审核
- 投标监管
- 数据统计

## 技术栈

- **后端**: Node.js + Express
- **数据库**: MySQL
- **前端**: HTML + CSS + JavaScript
- **认证**: JWT Token
- **安全**: bcrypt密码加密、CORS、Helmet安全中间件

## 部署环境

- 腾讯云轻量应用服务器
- Node.js 16+
- MySQL 8.0+
- Nginx (反向代理)

## 快速开始

1. 安装依赖
```bash
npm install
```

2. 配置环境变量
复制 `.env.example` 到 `.env` 并配置数据库连接

3. 初始化数据库
```bash
npm run init-db
```

4. 启动服务
```bash
npm run dev
```

## 项目结构

```
├── config/          # 配置文件
├── controllers/     # 控制器
├── middleware/      # 中间件
├── models/          # 数据模型
├── routes/          # 路由定义
├── public/          # 静态文件
├── views/           # 前端页面
├── uploads/         # 上传文件
└── server.js        # 入口文件
```

## API文档

详细的API文档请参考 `docs/api.md`

## 部署说明

部署到腾讯云轻量服务器的详细步骤请参考 `docs/deployment.md`
