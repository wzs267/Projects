# 招投标平台使用说明

## 快速开始

### 1. 环境要求
- Node.js 16.0 或更高版本
- MySQL 8.0 或更高版本
- 现代浏览器（Chrome、Firefox、Safari、Edge）

### 2. 本地开发

#### Windows系统
1. 双击运行 `init-db.bat` 初始化数据库
2. 双击运行 `start.bat` 启动开发服务器
3. 浏览器访问 `http://localhost:3000`

#### Linux/macOS系统
```bash
# 安装依赖
npm install

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置数据库连接

# 初始化数据库
npm run init-db

# 启动开发服务器
npm run dev
```

### 3. 生产部署
详细部署说明请参考 [部署文档](docs/deployment.md)

## 功能说明

### 用户角色

#### 发布方（Publisher）
- 注册/登录账户
- 发布项目需求
- 管理发布的项目
- 查看项目投标
- 选择中标者
- 项目状态管理

#### 投标方（Bidder）
- 注册/登录账户
- 浏览项目列表
- 查看项目详情
- 提交投标方案
- 管理投标记录
- 查看投标状态

#### 管理员（Admin）
- 用户管理
- 项目监管
- 数据统计
- 系统设置

### 主要功能模块

#### 1. 用户认证系统
- **注册**: 用户可选择发布方或投标方身份注册
- **登录**: 邮箱密码登录，JWT Token认证
- **个人资料**: 修改基本信息、更改密码
- **权限控制**: 基于角色的访问控制

#### 2. 项目管理系统
- **发布项目**: 发布方可发布项目需求
  - 项目标题
  - 项目描述
  - 预算范围
  - 截止时间
  - 技术要求
  - 联系方式
- **项目列表**: 公开展示所有活跃项目
- **搜索筛选**: 按类别、关键词搜索项目
- **项目详情**: 查看项目完整信息

#### 3. 投标系统
- **提交投标**: 投标方对感兴趣的项目提交方案
  - 投标金额
  - 完成时间
  - 技术方案
  - 项目经验
- **投标管理**: 查看、修改、撤回投标
- **中标通知**: 自动通知中标结果

#### 4. 项目状态流程
```
发布中 → 投标中 → 评标中 → 已中标 → 进行中 → 已完成
                          ↓
                        流标/取消
```

## API接口说明

### 认证接口
- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录
- `GET /api/auth/me` - 获取当前用户信息
- `PUT /api/auth/password` - 修改密码

### 项目接口
- `GET /api/projects` - 获取项目列表
- `GET /api/projects/:id` - 获取项目详情
- `POST /api/projects` - 发布新项目
- `PUT /api/projects/:id` - 更新项目
- `DELETE /api/projects/:id` - 删除项目
- `GET /api/projects/my/published` - 获取我发布的项目

### 投标接口
- `POST /api/bids` - 提交投标
- `GET /api/bids/my` - 获取我的投标记录
- `GET /api/bids/project/:projectId` - 获取项目投标列表
- `PATCH /api/bids/:id/select` - 选择中标者
- `DELETE /api/bids/:id` - 撤回投标

### 用户接口
- `PUT /api/users/profile` - 更新个人资料

### 管理员接口
- `GET /api/admin/stats` - 获取统计数据
- `GET /api/admin/users` - 获取用户列表
- `PATCH /api/admin/users/:id/status` - 更新用户状态

## 数据库设计

### 用户表 (users)
```sql
- id: 用户ID
- username: 用户名
- email: 邮箱
- password: 密码（加密存储）
- phone: 手机号
- user_type: 用户类型（publisher/bidder/admin）
- status: 账户状态（active/suspended）
- created_at: 创建时间
- updated_at: 更新时间
```

### 项目表 (projects)
```sql
- id: 项目ID
- title: 项目标题
- description: 项目描述
- budget: 预算
- deadline: 截止时间
- requirements: 技术要求
- contact_info: 联系方式
- category: 项目类别
- status: 项目状态
- user_id: 发布者ID
- created_at: 创建时间
- updated_at: 更新时间
```

### 投标表 (bids)
```sql
- id: 投标ID
- project_id: 项目ID
- user_id: 投标者ID
- amount: 投标金额
- proposal: 投标方案
- deadline: 承诺完成时间
- status: 投标状态（pending/won/lost/rejected）
- created_at: 创建时间
- updated_at: 更新时间
```

## 安全特性

### 1. 密码安全
- bcrypt加密存储密码
- 密码强度要求（最少6位）
- 密码修改需要验证原密码

### 2. 接口安全
- JWT Token认证
- 请求频率限制
- 输入验证和过滤
- SQL注入防护

### 3. 前端安全
- XSS防护（HTML转义）
- CSRF保护
- HTTPS支持（生产环境）

### 4. 数据安全
- 数据库备份
- 敏感信息加密
- 访问日志记录

## 系统配置

### 环境变量说明
```env
# 服务器配置
PORT=3000                    # 服务端口
NODE_ENV=development         # 运行环境

# 数据库配置
DB_HOST=localhost           # 数据库主机
DB_PORT=3306               # 数据库端口
DB_NAME=bidding_platform   # 数据库名
DB_USER=root               # 数据库用户
DB_PASSWORD=               # 数据库密码

# JWT配置
JWT_SECRET=your_secret_key  # JWT签名密钥
JWT_EXPIRES_IN=7d          # Token过期时间

# 文件上传
UPLOAD_PATH=./uploads      # 上传文件路径
MAX_FILE_SIZE=5242880      # 最大文件大小(5MB)
```

## 常见问题

### Q: 忘记密码怎么办？
A: 目前系统暂未提供密码重置功能，请联系管理员重置密码。

### Q: 如何修改投标？
A: 在投标状态为"待审核"时，可以撤回投标后重新提交。

### Q: 项目发布后还能修改吗？
A: 可以修改，但建议在有投标前进行修改，避免影响投标者。

### Q: 如何选择中标者？
A: 发布方在项目详情页面可以查看所有投标，选择合适的投标者。

### Q: 数据如何备份？
A: 生产环境建议定期备份MySQL数据库，具体方法参考部署文档。

## 技术支持

如遇到技术问题，请：
1. 查看服务器日志：`pm2 logs`
2. 检查数据库连接
3. 确认环境变量配置
4. 查看浏览器控制台错误信息

## 后续版本计划

- [ ] 邮件通知功能
- [ ] 文件上传功能
- [ ] 在线聊天功能
- [ ] 支付集成
- [ ] 移动端适配
- [ ] 项目评价系统
- [ ] 更丰富的统计报表
- [ ] 第三方登录集成
