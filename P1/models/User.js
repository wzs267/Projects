const db = require('../config/database');

class User {
  static async create(userData) {
    const { username, email, password, phone, userType = 'bidder' } = userData;
    const sql = `
      INSERT INTO users (username, email, password, phone, user_type, created_at)
      VALUES (?, ?, ?, ?, ?, NOW())
    `;
    const [result] = await db.execute(sql, [username, email, password, phone, userType]);
    return result.insertId;
  }

  static async findByEmail(email) {
    const sql = 'SELECT * FROM users WHERE email = ?';
    const [rows] = await db.execute(sql, [email]);
    return rows[0];
  }

  static async findByPhone(phone) {
    const sql = 'SELECT * FROM users WHERE phone = ?';
    const [rows] = await db.execute(sql, [phone]);
    return rows[0];
  }

  static async findById(id) {
    const sql = 'SELECT * FROM users WHERE id = ?';
    const [rows] = await db.execute(sql, [id]);
    return rows[0];
  }

  static async updateProfile(id, userData) {
    const { username, phone } = userData;
    const sql = 'UPDATE users SET username = ?, phone = ?, updated_at = NOW() WHERE id = ?';
    const [result] = await db.execute(sql, [username, phone, id]);
    return result.affectedRows > 0;
  }

  static async updatePassword(id, newPassword) {
    const sql = 'UPDATE users SET password = ?, updated_at = NOW() WHERE id = ?';
    const [result] = await db.execute(sql, [newPassword, id]);
    return result.affectedRows > 0;
  }

  static async updateStatus(id, status) {
    const sql = 'UPDATE users SET status = ?, updated_at = NOW() WHERE id = ?';
    const [result] = await db.execute(sql, [status, id]);
    return result.affectedRows > 0;
  }

  static async getAll(page = 1, limit = 10) {
    const offset = (page - 1) * limit;
    const sql = `
      SELECT id, username, email, phone, user_type, status, created_at
      FROM users
      ORDER BY created_at DESC
      LIMIT ? OFFSET ?
    `;
    const [rows] = await db.execute(sql, [limit, offset]);
    
    // 获取总数
    const countSql = 'SELECT COUNT(*) as total FROM users';
    const [countRows] = await db.execute(countSql);
    
    return {
      users: rows,
      total: countRows[0].total,
      page,
      limit
    };
  }
}

module.exports = User;
