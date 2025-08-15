const mysql = require('mysql2/promise');
const bcrypt = require('bcrypt');
require('dotenv').config();

async function cleanAndResetData() {
  let connection;
  
  try {
    // 连接到数据库
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
    console.log('连接数据库成功，字符集设置完成');

    // 清理现有的测试数据（保留管理员）
    await connection.execute('DELETE FROM bids WHERE id > 0');
    console.log('清理投标数据完成');
    
    await connection.execute('DELETE FROM projects WHERE id > 0');
    console.log('清理项目数据完成');
    
    await connection.execute("DELETE FROM users WHERE email != 'admin@example.com'");
    console.log('清理用户数据完成（保留管理员）');

    // 重置自增ID
    await connection.execute('ALTER TABLE users AUTO_INCREMENT = 2');
    await connection.execute('ALTER TABLE projects AUTO_INCREMENT = 1');
    await connection.execute('ALTER TABLE bids AUTO_INCREMENT = 1');
    console.log('重置自增ID完成');

    // 插入正确的测试用户
    const publisherPassword = await bcrypt.hash('123456', 10);
    const bidderPassword = await bcrypt.hash('123456', 10);

    // 插入发布方
    const [publisherResult] = await connection.execute(`
      INSERT INTO users (username, email, password, phone, user_type, status)
      VALUES (?, ?, ?, ?, ?, ?)
    `, ['测试发布方', 'publisher@test.com', publisherPassword, '13800138001', 'publisher', 'active']);
    
    const publisherId = publisherResult.insertId;
    console.log('插入发布方用户成功，ID:', publisherId);

    // 插入投标方
    const [bidderResult] = await connection.execute(`
      INSERT INTO users (username, email, password, phone, user_type, status)
      VALUES (?, ?, ?, ?, ?, ?)
    `, ['测试投标方', 'bidder@test.com', bidderPassword, '13800138002', 'bidder', 'active']);
    
    const bidderId = bidderResult.insertId;
    console.log('插入投标方用户成功，ID:', bidderId);

    // 插入测试项目
    const deadline = new Date();
    deadline.setDate(deadline.getDate() + 30); // 30天后截止
    const formattedDeadline = deadline.toISOString().slice(0, 19).replace('T', ' ');

    const [projectResult] = await connection.execute(`
      INSERT INTO projects (title, description, budget, deadline, requirements, contact_info, user_id, category, status)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, [
      '企业网站开发项目',
      '需要开发一个现代化的企业官网，包含产品展示、新闻发布、在线咨询等功能。要求响应式设计，兼容各种设备。',
      50000,
      formattedDeadline,
      '使用现代Web技术栈（React/Vue + Node.js），响应式设计，SEO优化，后台管理系统',
      '联系人：张经理，电话：13800138001，邮箱：zhang@company.com',
      publisherId,
      '网站开发',
      'active'
    ]);
    
    const projectId = projectResult.insertId;
    console.log('插入测试项目成功，ID:', projectId);

    // 插入第二个测试项目
    const [project2Result] = await connection.execute(`
      INSERT INTO projects (title, description, budget, deadline, requirements, contact_info, user_id, category, status)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, [
      '移动应用开发项目',
      '开发一款电商类移动应用，支持商品浏览、购物车、在线支付、订单管理等功能。需要iOS和Android双平台。',
      80000,
      formattedDeadline,
      '原生开发或React Native，支持iOS 12+和Android 8+，集成支付宝、微信支付',
      '联系人：李经理，电话：13800138003，邮箱：li@company.com',
      publisherId,
      '移动应用',
      'active'
    ]);
    
    const project2Id = project2Result.insertId;
    console.log('插入第二个测试项目成功，ID:', project2Id);

    // 插入测试投标
    const bidDeadline = new Date();
    bidDeadline.setDate(bidDeadline.getDate() + 20); // 20天后完成
    const formattedBidDeadline = bidDeadline.toISOString().slice(0, 19).replace('T', ' ');

    await connection.execute(`
      INSERT INTO bids (project_id, user_id, amount, proposal, deadline, status)
      VALUES (?, ?, ?, ?, ?, ?)
    `, [
      projectId,
      bidderId,
      45000,
      '我们是一家专业的Web开发团队，具有5年以上的企业网站开发经验。我们将使用React + Node.js技术栈来开发这个项目，确保网站的性能和用户体验。我们承诺按时交付，并提供一年的免费维护服务。',
      formattedBidDeadline,
      'pending'
    ]);
    
    console.log('插入测试投标成功');

    console.log('\n==========================================');
    console.log('            数据重置完成');
    console.log('==========================================');
    console.log('\n✅ 测试账户信息:');
    console.log('发布方: publisher@test.com / 123456');
    console.log('投标方: bidder@test.com / 123456');
    console.log('管理员: admin@example.com / admin123456');
    console.log('\n✅ 测试数据:');
    console.log('- 2个测试项目（网站开发、移动应用）');
    console.log('- 1个测试投标');
    console.log('\n🌐 请刷新网页查看正确的中文显示');

  } catch (error) {
    console.error('数据重置失败:', error);
  } finally {
    if (connection) {
      await connection.end();
    }
  }
}

// 如果直接运行此文件，则执行数据重置
if (require.main === module) {
  cleanAndResetData();
}

module.exports = cleanAndResetData;
