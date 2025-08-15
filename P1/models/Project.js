const db = require('../config/database');

class Project {
  static async create(projectData) {
    const {
      title, description, budget, deadline, requirements,
      contact_info, user_id, category = '其他',
      work_deadline, bid_deadline, is_public = true, images = []
    } = projectData;
    
    // 格式化日期为MySQL DATETIME格式
    const formattedDeadline = new Date(deadline).toISOString().slice(0, 19).replace('T', ' ');
    const formattedWorkDeadline = work_deadline ? new Date(work_deadline).toISOString().slice(0, 19).replace('T', ' ') : null;
    const formattedBidDeadline = bid_deadline ? new Date(bid_deadline).toISOString().slice(0, 19).replace('T', ' ') : formattedDeadline;
    
    const sql = `
      INSERT INTO projects (title, description, budget, deadline, work_deadline, bid_deadline,
                           requirements, contact_info, user_id, category, is_public, images, status, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NOW())
    `;
    const [result] = await db.execute(sql, [
      title, description, budget, formattedDeadline, formattedWorkDeadline, formattedBidDeadline,
      requirements, contact_info, user_id, category, is_public, JSON.stringify(images)
    ]);
    return result.insertId;
  }

  static async findById(id) {
    const sql = `
      SELECT p.*, u.username as publisher_name, u.email as publisher_email,
             (SELECT COUNT(*) FROM bids WHERE project_id = p.id) as bid_count
      FROM projects p
      LEFT JOIN users u ON p.user_id = u.id
      WHERE p.id = ?
    `;
    const [rows] = await db.execute(sql, [id]);
    return rows[0];
  }

  static async getAll(filters = {}) {
    let sql = `
      SELECT p.*, u.username as publisher_name,
             (SELECT COUNT(*) FROM bids WHERE project_id = p.id) as bid_count
      FROM projects p
      LEFT JOIN users u ON p.user_id = u.id
      WHERE 1=1
    `;
    const params = [];

    if (filters.status) {
      sql += ' AND p.status = ?';
      params.push(filters.status);
    }

    if (filters.category) {
      sql += ' AND p.category = ?';
      params.push(filters.category);
    }

    if (filters.search) {
      sql += ' AND (p.title LIKE ? OR p.description LIKE ?)';
      params.push(`%${filters.search}%`, `%${filters.search}%`);
    }

    if (filters.user_id) {
      sql += ' AND p.user_id = ?';
      params.push(filters.user_id);
    }

    sql += ' ORDER BY p.created_at DESC';

    if (filters.limit) {
      const offset = ((filters.page || 1) - 1) * filters.limit;
      sql += ' LIMIT ? OFFSET ?';
      params.push(filters.limit, offset);
    }

    const [rows] = await db.execute(sql, params);
    return rows;
  }

  static async update(id, projectData) {
    const {
      title, description, budget, deadline, requirements,
      contact_info, category, status
    } = projectData;
    
    const sql = `
      UPDATE projects 
      SET title = ?, description = ?, budget = ?, deadline = ?, 
          requirements = ?, contact_info = ?, category = ?, status = ?,
          updated_at = NOW()
      WHERE id = ?
    `;
    const [result] = await db.execute(sql, [
      title, description, budget, deadline, requirements,
      contact_info, category, status, id
    ]);
    return result.affectedRows > 0;
  }

  static async delete(id) {
    const sql = 'DELETE FROM projects WHERE id = ?';
    const [result] = await db.execute(sql, [id]);
    return result.affectedRows > 0;
  }

  static async updateStatus(id, status) {
    const sql = 'UPDATE projects SET status = ?, updated_at = NOW() WHERE id = ?';
    const [result] = await db.execute(sql, [status, id]);
    return result.affectedRows > 0;
  }

  // 设置项目为已分配状态，并记录分配的用户
  static async assignProject(id, userId) {
    const sql = 'UPDATE projects SET status = ?, assigned_to = ?, updated_at = NOW() WHERE id = ?';
    const [result] = await db.execute(sql, ['assigned', userId, id]);
    return result.affectedRows > 0;
  }

  // 检查并更新过期的项目投标状态
  static async checkAndUpdateExpiredBidding() {
    const sql = `
      UPDATE projects 
      SET status = 'bid_closed', updated_at = NOW()
      WHERE status = 'active' 
      AND bid_deadline < NOW()
    `;
    const [result] = await db.execute(sql);
    return result.affectedRows;
  }

  // 检查项目是否可以投标
  static async canAcceptBids(id) {
    const project = await this.findById(id);
    if (!project) {
      return { canBid: false, reason: '项目不存在' };
    }
    
    if (project.status !== 'active') {
      return { canBid: false, reason: '项目不在活跃状态' };
    }
    
    const bidDeadline = new Date(project.bid_deadline || project.deadline);
    const now = new Date();
    
    if (now > bidDeadline) {
      // 自动更新项目状态为投标已截止
      await this.updateStatus(id, 'bid_closed');
      return { canBid: false, reason: '投标截止时间已过' };
    }
    
    return { canBid: true };
  }

  static async getStats() {
    const sql = `
      SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled
      FROM projects
    `;
    const [rows] = await db.execute(sql);
    return rows[0];
  }

  // 更新项目信息
  static async update(id, projectData) {
    const {
      title, description, budget, deadline, requirements,
      contact_info, category, status, work_deadline, bid_deadline,
      is_public, accepted_bid_id, freelancer_id
    } = projectData;
    
    // 构建动态SQL
    const updates = [];
    const params = [];
    
    if (title !== undefined) {
      updates.push('title = ?');
      params.push(title);
    }
    if (description !== undefined) {
      updates.push('description = ?');
      params.push(description);
    }
    if (budget !== undefined) {
      updates.push('budget = ?');
      params.push(budget);
    }
    if (deadline !== undefined) {
      updates.push('deadline = ?');
      params.push(new Date(deadline).toISOString().slice(0, 19).replace('T', ' '));
    }
    if (work_deadline !== undefined) {
      updates.push('work_deadline = ?');
      params.push(work_deadline ? new Date(work_deadline).toISOString().slice(0, 19).replace('T', ' ') : null);
    }
    if (bid_deadline !== undefined) {
      updates.push('bid_deadline = ?');
      params.push(bid_deadline ? new Date(bid_deadline).toISOString().slice(0, 19).replace('T', ' ') : null);
    }
    if (requirements !== undefined) {
      updates.push('requirements = ?');
      params.push(requirements);
    }
    if (contact_info !== undefined) {
      updates.push('contact_info = ?');
      params.push(contact_info);
    }
    if (category !== undefined) {
      updates.push('category = ?');
      params.push(category);
    }
    if (status !== undefined) {
      updates.push('status = ?');
      params.push(status);
    }
    if (is_public !== undefined) {
      updates.push('is_public = ?');
      params.push(is_public);
    }
    if (accepted_bid_id !== undefined) {
      updates.push('accepted_bid_id = ?');
      params.push(accepted_bid_id);
    }
    if (freelancer_id !== undefined) {
      updates.push('freelancer_id = ?');
      params.push(freelancer_id);
    }
    
    if (updates.length === 0) {
      return true; // 没有需要更新的字段
    }
    
    updates.push('updated_at = NOW()');
    params.push(id);
    
    const sql = `UPDATE projects SET ${updates.join(', ')} WHERE id = ?`;
    const [result] = await db.execute(sql, params);
    return result.affectedRows > 0;
  }

  static async delete(id) {
    // 软删除：将状态设置为cancelled
    const sql = `UPDATE projects SET status = 'cancelled', updated_at = NOW() WHERE id = ?`;
    const [result] = await db.execute(sql, [id]);
    return result.affectedRows > 0;
  }
}

module.exports = Project;
