# 腾讯云轻量服务器部署指南

## 服务器环境准备

### 1. 连接服务器
```bash
ssh root@your_server_ip
```

### 2. 更新系统
```bash
apt update && apt upgrade -y
```

### 3. 安装Node.js
```bash
# 安装NodeSource存储库
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -

# 安装Node.js
apt-get install -y nodejs

# 验证安装
node --version
npm --version
```

### 4. 安装MySQL
```bash
# 安装MySQL服务器
apt install mysql-server -y

# 安全配置
mysql_secure_installation

# 启动MySQL服务
systemctl start mysql
systemctl enable mysql
```

### 5. 安装Nginx
```bash
apt install nginx -y
systemctl start nginx
systemctl enable nginx
```

### 6. 安装PM2（进程管理器）
```bash
npm install -g pm2
```

## 应用部署

### 1. 上传代码
```bash
# 在服务器上创建项目目录
mkdir -p /var/www/bidding-platform
cd /var/www/bidding-platform

# 从Git仓库克隆代码或直接上传文件
# git clone your_repository_url .
```

### 2. 安装依赖
```bash
npm install --production
```

### 3. 配置环境变量
```bash
# 复制环境变量文件
cp .env.example .env

# 编辑环境变量
nano .env
```

环境变量配置示例：
```env
PORT=3000
NODE_ENV=production

# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_NAME=bidding_platform
DB_USER=bidding_user
DB_PASSWORD=your_secure_password

# JWT配置
JWT_SECRET=your_very_secure_jwt_secret_key
JWT_EXPIRES_IN=7d

# 文件上传配置
UPLOAD_PATH=./uploads
MAX_FILE_SIZE=5242880
```

### 4. 数据库设置
```bash
# 登录MySQL
mysql -u root -p

# 创建数据库用户
CREATE USER 'bidding_user'@'localhost' IDENTIFIED BY 'your_secure_password';
CREATE DATABASE bidding_platform CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
GRANT ALL PRIVILEGES ON bidding_platform.* TO 'bidding_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;

# 初始化数据库
node scripts/init-db.js
```

### 5. 创建必要目录
```bash
mkdir -p uploads
chmod 755 uploads
```

### 6. 配置Nginx
```bash
# 创建Nginx配置文件
nano /etc/nginx/sites-available/bidding-platform
```

Nginx配置内容：
```nginx
server {
    listen 80;
    server_name your_domain.com;

    # 静态文件
    location / {
        root /var/www/bidding-platform/public;
        try_files $uri $uri/ /index.html;
    }

    # API代理
    location /api {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # 上传文件
    location /uploads {
        root /var/www/bidding-platform;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
}
```

启用站点：
```bash
# 创建符号链接
ln -s /etc/nginx/sites-available/bidding-platform /etc/nginx/sites-enabled/

# 删除默认站点
rm /etc/nginx/sites-enabled/default

# 测试配置
nginx -t

# 重新加载Nginx
systemctl reload nginx
```

### 7. 启动应用
```bash
# 使用PM2启动应用
pm2 start server.js --name "bidding-platform"

# 设置开机启动
pm2 startup
pm2 save
```

## SSL证书配置（推荐）

### 1. 安装Certbot
```bash
apt install certbot python3-certbot-nginx -y
```

### 2. 获取SSL证书
```bash
certbot --nginx -d your_domain.com
```

### 3. 自动续期
```bash
# 测试自动续期
certbot renew --dry-run

# 添加定时任务
crontab -e
```

添加以下行到crontab：
```
0 12 * * * /usr/bin/certbot renew --quiet
```

## 防火墙配置

```bash
# 启用UFW防火墙
ufw enable

# 允许SSH
ufw allow ssh

# 允许HTTP和HTTPS
ufw allow 'Nginx Full'

# 检查状态
ufw status
```

## 监控和维护

### 1. PM2监控
```bash
# 查看应用状态
pm2 status

# 查看日志
pm2 logs bidding-platform

# 重启应用
pm2 restart bidding-platform

# 查看详细信息
pm2 show bidding-platform
```

### 2. 数据库备份
创建备份脚本 `/root/backup.sh`：
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/root/backups"
mkdir -p $BACKUP_DIR

# 数据库备份
mysqldump -u bidding_user -p'your_secure_password' bidding_platform > $BACKUP_DIR/bidding_platform_$DATE.sql

# 压缩备份文件
gzip $BACKUP_DIR/bidding_platform_$DATE.sql

# 删除7天前的备份
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "备份完成: bidding_platform_$DATE.sql.gz"
```

设置定时备份：
```bash
chmod +x /root/backup.sh
crontab -e
```

添加每日备份任务：
```
0 2 * * * /root/backup.sh
```

### 3. 日志轮转
```bash
# 创建日志轮转配置
nano /etc/logrotate.d/bidding-platform
```

内容：
```
/var/www/bidding-platform/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    notifempty
    create 644 www-data www-data
    postrotate
        pm2 reload bidding-platform
    endscript
}
```

## 性能优化

### 1. Node.js应用优化
```bash
# 在package.json中添加生产环境脚本
"scripts": {
    "start": "NODE_ENV=production node server.js",
    "prod": "pm2 start ecosystem.config.js --env production"
}
```

### 2. 创建PM2配置文件
创建 `ecosystem.config.js`：
```javascript
module.exports = {
  apps: [{
    name: 'bidding-platform',
    script: 'server.js',
    instances: 'max',
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'development'
    },
    env_production: {
      NODE_ENV: 'production',
      PORT: 3000
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true
  }]
};
```

## 安全建议

1. **定期更新系统和软件包**
2. **使用强密码**
3. **配置防火墙**
4. **启用SSL/TLS**
5. **定期备份数据**
6. **监控系统日志**
7. **限制SSH访问**
8. **使用非root用户运行应用**

## 故障排除

### 常见问题

1. **应用无法启动**
   ```bash
   pm2 logs bidding-platform
   ```

2. **数据库连接失败**
   ```bash
   mysql -u bidding_user -p
   ```

3. **Nginx配置错误**
   ```bash
   nginx -t
   ```

4. **端口被占用**
   ```bash
   lsof -i :3000
   ```

### 有用的命令

```bash
# 查看系统资源使用
htop

# 查看磁盘使用
df -h

# 查看网络连接
netstat -tlnp

# 查看进程
ps aux | grep node

# 查看系统日志
journalctl -u nginx
journalctl -u mysql
```
