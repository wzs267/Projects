const db = require('../config/database');

class Bid {
  static async create(bidData) {
    const { project_id, user_id, amount, proposal, deadline } = bidData;
    
    const sql = `
      INSERT INTO bids (project_id, user_id, amount, proposal, deadline, status, created_at)
      VALUES (?, ?, ?, ?, ?, 'pending', NOW())
    `;
    const [result] = await db.execute(sql, [project_id, user_id, amount, proposal, deadline]);
    return result.insertId;
  }

  static async findById(id) {
    const sql = `
      SELECT b.*, u.username as bidder_name, u.email as bidder_email,
             p.title as project_title
      FROM bids b
      LEFT JOIN users u ON b.user_id = u.id
      LEFT JOIN projects p ON b.project_id = p.id
      WHERE b.id = ?
    `;
    const [rows] = await db.execute(sql, [id]);
    return rows[0];
  }

  static async getByProject(projectId) {
    const sql = `
      SELECT b.*, u.username as bidder_name, u.email as bidder_email
      FROM bids b
      LEFT JOIN users u ON b.user_id = u.id
      WHERE b.project_id = ?
      ORDER BY b.created_at DESC
    `;
    const [rows] = await db.execute(sql, [projectId]);
    return rows;
  }

  static async getByUser(userId, filters = {}) {
    let sql = `
      SELECT b.*, p.title as project_title, p.status as project_status,
             u.username as publisher_name
      FROM bids b
      LEFT JOIN projects p ON b.project_id = p.id
      LEFT JOIN users u ON p.user_id = u.id
      WHERE b.user_id = ?
    `;
    const params = [userId];

    if (filters.status) {
      sql += ' AND b.status = ?';
      params.push(filters.status);
    }

    sql += ' ORDER BY b.created_at DESC';

    if (filters.limit) {
      const offset = ((filters.page || 1) - 1) * filters.limit;
      sql += ' LIMIT ? OFFSET ?';
      params.push(filters.limit, offset);
    }

    const [rows] = await db.execute(sql, params);
    return rows;
  }

  static async updateStatus(id, status) {
    const sql = 'UPDATE bids SET status = ?, updated_at = NOW() WHERE id = ?';
    const [result] = await db.execute(sql, [status, id]);
    
    // 如果是中标，执行相关的自动处理逻辑
    if (status === 'won') {
      await this.handleBidWon(id);
    }
    
    return result.affectedRows > 0;
  }

  // 处理中标后的自动逻辑
  static async handleBidWon(winnerId) {
    try {
      // 获取中标投标的信息
      const winnerBid = await this.findById(winnerId);
      if (!winnerBid) {
        throw new Error('中标投标不存在');
      }

      // 1. 将同一项目的其他投标设为失败
      const updateOtherBidsSql = `
        UPDATE bids 
        SET status = 'lost', updated_at = NOW() 
        WHERE project_id = ? AND id != ? AND status = 'pending'
      `;
      const [result1] = await db.execute(updateOtherBidsSql, [winnerBid.project_id, winnerId]);

      // 2. 将该投标方在其他项目的所有投标撤销（设为withdrawn状态）
      const withdrawOtherBidsSql = `
        UPDATE bids 
        SET status = 'withdrawn', updated_at = NOW() 
        WHERE user_id = ? AND id != ? AND status = 'pending'
      `;
      const [result2] = await db.execute(withdrawOtherBidsSql, [winnerBid.user_id, winnerId]);

      // 3. 更新项目状态为已分配
      const Project = require('./Project');
      await Project.assignProject(winnerBid.project_id, winnerBid.user_id);

      console.log(`投标 ${winnerId} 中标处理完成：`);
      console.log(`- 同项目其他 ${result1.affectedRows} 个投标设为失败`);
      console.log(`- 投标方的其他 ${result2.affectedRows} 个投标已撤销`);
      console.log(`- 项目 ${winnerBid.project_id} 已分配给用户 ${winnerBid.user_id}`);
      
    } catch (error) {
      console.error('处理中标逻辑错误:', error);
      throw error;
    }
  }

  static async delete(id) {
    const sql = 'DELETE FROM bids WHERE id = ?';
    const [result] = await db.execute(sql, [id]);
    return result.affectedRows > 0;
  }

  static async checkExistingBid(projectId, userId) {
    const sql = 'SELECT id FROM bids WHERE project_id = ? AND user_id = ?';
    const [rows] = await db.execute(sql, [projectId, userId]);
    return rows.length > 0;
  }

  static async getStats() {
    const sql = `
      SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
        SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as won,
        SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as lost
      FROM bids
    `;
    const [rows] = await db.execute(sql);
    return rows[0];
  }

  static async delete(id) {
    const sql = 'DELETE FROM bids WHERE id = ?';
    const [result] = await db.execute(sql, [id]);
    return result.affectedRows > 0;
  }

  // 获取投标方的所有待处理投标
  static async getPendingBidsByUser(userId) {
    const sql = `
      SELECT b.*, p.title as project_title, p.status as project_status
      FROM bids b
      LEFT JOIN projects p ON b.project_id = p.id
      WHERE b.user_id = ? AND b.status = 'pending'
      ORDER BY b.created_at DESC
    `;
    const [rows] = await db.execute(sql, [userId]);
    return rows;
  }

  // 批量撤销投标
  static async withdrawBidsByUser(userId, excludeBidId = null) {
    let sql = `
      UPDATE bids 
      SET status = 'withdrawn', updated_at = NOW() 
      WHERE user_id = ? AND status = 'pending'
    `;
    const params = [userId];
    
    if (excludeBidId) {
      sql += ' AND id != ?';
      params.push(excludeBidId);
    }
    
    const [result] = await db.execute(sql, params);
    return result.affectedRows;
  }

  // 批量设置同项目其他投标为失败
  static async setOtherBidsAsLost(projectId, winnerId) {
    const sql = `
      UPDATE bids 
      SET status = 'lost', updated_at = NOW() 
      WHERE project_id = ? AND id != ? AND status = 'pending'
    `;
    const [result] = await db.execute(sql, [projectId, winnerId]);
    return result.affectedRows;
  }
}

module.exports = Bid;
