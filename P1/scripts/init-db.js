const mysql = require('mysql2/promise');
require('dotenv').config();

async function initDatabase() {
  let connection;
  
  try {
    // 首先连接到MySQL服务器（不指定数据库）
    connection = await mysql.createConnection({
      host: process.env.DB_HOST || 'localhost',
      port: process.env.DB_PORT || 3306,
      user: process.env.DB_USER || 'root',
      password: process.env.DB_PASSWORD || '',
      charset: 'utf8mb4'
    });

    // 设置字符集
    await connection.execute('SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci');
    
    // 创建数据库
    await connection.execute(`CREATE DATABASE IF NOT EXISTS ${process.env.DB_NAME || 'bidding_platform'} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci`);
    console.log('数据库创建成功');

    // 关闭连接
    await connection.end();

    // 重新连接到指定数据库
    connection = await mysql.createConnection({
      host: process.env.DB_HOST || 'localhost',
      port: process.env.DB_PORT || 3306,
      user: process.env.DB_USER || 'root',
      password: process.env.DB_PASSWORD || '',
      charset: 'utf8mb4',
      database: process.env.DB_NAME || 'bidding_platform'
    });

    // 设置字符集
    await connection.execute('SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci');

    // 创建用户表
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) NOT NULL,
        email VARCHAR(100) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL,
        phone VARCHAR(20),
        user_type ENUM('publisher', 'bidder', 'admin') DEFAULT 'bidder',
        status ENUM('active', 'suspended') DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    `);
    console.log('用户表创建成功');

    // 创建项目表
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS projects (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(200) NOT NULL,
        description TEXT NOT NULL,
        budget DECIMAL(12,2) NOT NULL,
        deadline DATETIME NOT NULL,
        requirements TEXT NOT NULL,
        contact_info VARCHAR(500) NOT NULL,
        category VARCHAR(50) DEFAULT '其他',
        status ENUM('active', 'in_progress', 'completed', 'cancelled') DEFAULT 'active',
        user_id INT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        INDEX idx_status (status),
        INDEX idx_category (category),
        INDEX idx_deadline (deadline)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    `);
    console.log('项目表创建成功');

    // 创建投标表
    await connection.execute(`
      CREATE TABLE IF NOT EXISTS bids (
        id INT AUTO_INCREMENT PRIMARY KEY,
        project_id INT NOT NULL,
        user_id INT NOT NULL,
        amount DECIMAL(12,2) NOT NULL,
        proposal TEXT NOT NULL,
        deadline DATETIME NOT NULL,
        status ENUM('pending', 'won', 'lost', 'rejected') DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE KEY unique_bid (project_id, user_id),
        INDEX idx_status (status),
        INDEX idx_project (project_id),
        INDEX idx_user (user_id)
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    `);
    console.log('投标表创建成功');

    // 插入管理员用户（可选）
    const bcrypt = require('bcrypt');
    const adminPassword = await bcrypt.hash('admin123456', 10);
    
    await connection.execute(`
      INSERT IGNORE INTO users (username, email, password, user_type)
      VALUES ('管理员', 'admin@example.com', ?, 'admin')
    `, [adminPassword]);
    console.log('管理员用户创建成功');

    console.log('数据库初始化完成！');
    
  } catch (error) {
    console.error('数据库初始化失败:', error);
  } finally {
    if (connection) {
      await connection.end();
    }
  }
}

// 如果直接运行此文件，则执行初始化
if (require.main === module) {
  initDatabase();
}

module.exports = initDatabase;
