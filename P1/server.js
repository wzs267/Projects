const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const path = require('path');
require('dotenv').config();

const authRoutes = require('./routes/auth');
const newAuthRoutes = require('./routes/newAuth');
const userRoutes = require('./routes/users');
const projectRoutes = require('./routes/projects');
const bidRoutes = require('./routes/bids');
const adminRoutes = require('./routes/admin');

const app = express();
const PORT = process.env.PORT || 3000;

// 安全中间件
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      "script-src": ["'self'", "'unsafe-inline'"],
      "script-src-attr": ["'unsafe-inline'"]
    }
  }
}));
app.use(cors());

// 请求限制
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15分钟
  max: 100 // 限制每个IP 15分钟内最多100个请求
});
app.use(limiter);

// 解析JSON
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// 静态文件服务
app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// 路由
app.use('/api/auth', newAuthRoutes); // 使用新的认证路由
app.use('/api/auth/old', authRoutes); // 保留旧的认证路由作为备份
app.use('/api/users', userRoutes);
app.use('/api/projects', projectRoutes);
app.use('/api/bids', bidRoutes);
app.use('/api/admin', adminRoutes);

// 根路径返回首页
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 404处理
app.use('*', (req, res) => {
  res.status(404).json({ error: '页面未找到' });
});

// 错误处理中间件
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: '服务器内部错误' });
});

app.listen(PORT, () => {
  console.log(`服务器运行在端口 ${PORT}`);
  console.log(`访问地址: http://localhost:${PORT}`);
  
  // 启动定时任务，每5分钟检查一次过期的项目投标
  const Project = require('./models/Project');
  
  const checkExpiredProjects = async () => {
    try {
      const updatedCount = await Project.checkAndUpdateExpiredBidding();
      if (updatedCount > 0) {
        console.log(`已更新 ${updatedCount} 个过期项目的投标状态为 bid_closed`);
      }
    } catch (error) {
      console.error('检查过期项目错误:', error);
    }
  };
  
  // 立即执行一次
  checkExpiredProjects();
  
  // 每5分钟执行一次
  setInterval(checkExpiredProjects, 5 * 60 * 1000);
});
