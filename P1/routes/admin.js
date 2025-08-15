const express = require('express');
const User = require('../models/User');
const Project = require('../models/Project');
const Bid = require('../models/Bid');
const { authenticateToken, requireRole } = require('../middleware/auth');

const router = express.Router();

// 管理员统计数据
router.get('/stats', authenticateToken, requireRole(['admin']), async (req, res) => {
  try {
    const userStats = await User.getAll(1, 1);
    const projectStats = await Project.getStats();
    const bidStats = await Bid.getStats();

    res.json({
      users: {
        total: userStats.total
      },
      projects: projectStats,
      bids: bidStats
    });
  } catch (error) {
    console.error('获取统计数据错误:', error);
    res.status(500).json({ error: '获取统计数据失败' });
  }
});

// 用户管理
router.get('/users', authenticateToken, requireRole(['admin']), async (req, res) => {
  try {
    const { page = 1, limit = 10 } = req.query;
    const result = await User.getAll(parseInt(page), parseInt(limit));
    res.json(result);
  } catch (error) {
    console.error('获取用户列表错误:', error);
    res.status(500).json({ error: '获取用户列表失败' });
  }
});

// 更新用户状态
router.patch('/users/:id/status', authenticateToken, requireRole(['admin']), async (req, res) => {
  try {
    const { status } = req.body;
    if (!['active', 'suspended'].includes(status)) {
      return res.status(400).json({ error: '状态值无效' });
    }

    const updated = await User.updateStatus(req.params.id, status);
    if (!updated) {
      return res.status(404).json({ error: '用户不存在' });
    }

    res.json({ message: '用户状态更新成功' });
  } catch (error) {
    console.error('更新用户状态错误:', error);
    res.status(500).json({ error: '更新用户状态失败' });
  }
});

module.exports = router;
