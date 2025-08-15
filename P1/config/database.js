const mysql = require('mysql2/promise');

const dbConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 3306,
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'bidding_platform',
  charset: 'utf8mb4',
  timezone: '+08:00',
  multipleStatements: false
};

// 创建连接池
const pool = mysql.createPool({
  ...dbConfig,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  idleTimeout: 300000,  // 5分钟空闲超时
  enableKeepAlive: true,
  keepAliveInitialDelay: 0
});

// 设置连接事件监听
pool.on('connection', function (connection) {
  console.log('数据库连接建立，连接ID:', connection.threadId);
  // 为每个新连接设置字符集
  connection.query('SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci', function (error) {
    if (error) console.error('设置字符集失败:', error);
  });
});

// 测试数据库连接
async function testConnection() {
  try {
    const connection = await pool.getConnection();
    console.log('数据库连接成功');
    connection.release();
  } catch (error) {
    console.error('数据库连接失败:', error.message);
  }
}

testConnection();

module.exports = pool;
