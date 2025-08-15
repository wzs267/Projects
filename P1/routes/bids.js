const express = require('express');
const { body, validationResult } = require('express-validator');
const Bid = require('../models/Bid');
const Project = require('../models/Project');
const { authenticateToken, requireRole } = require('../middleware/auth');

const router = express.Router();

// 提交投标
router.post('/', authenticateToken, requireRole(['bidder']), [
  body('project_id').isNumeric().withMessage('项目ID无效'),
  body('amount').isNumeric().withMessage('投标金额必须是数字'),
  body('proposal').isLength({ min: 20 }).withMessage('投标方案至少20字符'),
  body('deadline').isISO8601().withMessage('完成时间格式无效')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: errors.array()[0].msg });
    }

    const { project_id, amount, proposal, deadline } = req.body;

    // 检查项目是否存在且可以投标
    const project = await Project.findById(project_id);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }

    // 使用新的投标资格检查方法
    const bidCheck = await Project.canAcceptBids(project_id);
    if (!bidCheck.canBid) {
      return res.status(400).json({ error: bidCheck.reason });
    }

    // 检查是否已经投标过
    const existingBid = await Bid.checkExistingBid(project_id, req.user.id);
    if (existingBid) {
      return res.status(400).json({ error: '您已经对此项目投标过了' });
    }

    const bidData = {
      project_id,
      user_id: req.user.id,
      amount,
      proposal,
      deadline
    };

    const bidId = await Bid.create(bidData);
    const bid = await Bid.findById(bidId);

    res.status(201).json({
      message: '投标成功',
      bid
    });
  } catch (error) {
    console.error('提交投标错误:', error);
    res.status(500).json({ error: '投标失败' });
  }
});

// 获取项目的所有投标
router.get('/project/:projectId', authenticateToken, async (req, res) => {
  try {
    const { projectId } = req.params;

    // 检查项目是否存在
    const project = await Project.findById(projectId);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }

    // 只有项目发布者可以查看所有投标
    if (project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只有项目发布者可以查看投标' });
    }

    const bids = await Bid.getByProject(projectId);
    res.json({ bids });
  } catch (error) {
    console.error('获取项目投标错误:', error);
    res.status(500).json({ error: '获取投标信息失败' });
  }
});

// 获取我的投标记录
router.get('/my', authenticateToken, requireRole(['bidder']), async (req, res) => {
  try {
    const { page = 1, limit = 10, status } = req.query;
    
    const filters = {
      page: parseInt(page),
      limit: parseInt(limit),
      status
    };

    const bids = await Bid.getByUser(req.user.id, filters);
    res.json({ bids });
  } catch (error) {
    console.error('获取我的投标错误:', error);
    res.status(500).json({ error: '获取投标记录失败' });
  }
});

// 获取单个投标详情
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const bid = await Bid.findById(req.params.id);
    if (!bid) {
      return res.status(404).json({ error: '投标不存在' });
    }

    // 检查权限：投标者或项目发布者可以查看
    const project = await Project.findById(bid.project_id);
    if (bid.user_id !== req.user.id && project.user_id !== req.user.id) {
      return res.status(403).json({ error: '无权查看此投标' });
    }

    res.json({ bid });
  } catch (error) {
    console.error('获取投标详情错误:', error);
    res.status(500).json({ error: '获取投标详情失败' });
  }
});

// 选择中标者（项目发布者操作）
router.patch('/:id/select', authenticateToken, requireRole(['publisher']), async (req, res) => {
  try {
    const bid = await Bid.findById(req.params.id);
    if (!bid) {
      return res.status(404).json({ error: '投标不存在' });
    }

    // 检查项目是否存在
    const project = await Project.findById(bid.project_id);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }

    // 检查是否是项目发布者
    if (project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只有项目发布者可以选择中标者' });
    }

    // 检查项目状态
    if (project.status !== 'active') {
      return res.status(400).json({ error: '项目不在活跃状态' });
    }

    // 更新投标状态为中标
    const updated = await Bid.updateStatus(req.params.id, 'won', req.params.id);
    if (!updated) {
      return res.status(500).json({ error: '选择中标者失败' });
    }

    // 更新项目状态为进行中
    await Project.updateStatus(bid.project_id, 'in_progress');

    res.json({ message: '选择中标者成功' });
  } catch (error) {
    console.error('选择中标者错误:', error);
    res.status(500).json({ error: '选择中标者失败' });
  }
});

// 拒绝投标（项目发布者操作）
router.patch('/:id/reject', authenticateToken, requireRole(['publisher']), async (req, res) => {
  try {
    const bid = await Bid.findById(req.params.id);
    if (!bid) {
      return res.status(404).json({ error: '投标不存在' });
    }

    // 检查项目是否存在
    const project = await Project.findById(bid.project_id);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }

    // 检查是否是项目发布者
    if (project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只有项目发布者可以拒绝投标' });
    }

    // 更新投标状态为拒绝
    const updated = await Bid.updateStatus(req.params.id, 'rejected');
    if (!updated) {
      return res.status(500).json({ error: '拒绝投标失败' });
    }

    res.json({ message: '拒绝投标成功' });
  } catch (error) {
    console.error('拒绝投标错误:', error);
    res.status(500).json({ error: '拒绝投标失败' });
  }
});

// 撤回投标（投标者操作）
router.delete('/:id', authenticateToken, requireRole(['bidder']), async (req, res) => {
  try {
    const bid = await Bid.findById(req.params.id);
    if (!bid) {
      return res.status(404).json({ error: '投标不存在' });
    }

    // 检查是否是投标者
    if (bid.user_id !== req.user.id) {
      return res.status(403).json({ error: '只有投标者可以撤回投标' });
    }

    // 检查投标状态
    if (bid.status !== 'pending') {
      return res.status(400).json({ error: '只能撤回待审核的投标' });
    }

    const deleted = await Bid.delete(req.params.id);
    if (!deleted) {
      return res.status(500).json({ error: '撤回投标失败' });
    }

    res.json({ message: '撤回投标成功' });
  } catch (error) {
    console.error('撤回投标错误:', error);
    res.status(500).json({ error: '撤回投标失败' });
  }
});

// 接受投标
router.post('/:id/accept', authenticateToken, requireRole(['publisher']), async (req, res) => {
  try {
    const bidId = req.params.id;
    
    // 获取投标信息
    const bid = await Bid.findById(bidId);
    if (!bid) {
      return res.status(404).json({ error: '投标不存在' });
    }
    
    // 检查是否是项目发布者
    const project = await Project.findById(bid.project_id);
    if (!project || project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只有项目发布者可以接受投标' });
    }
    
    // 检查项目状态 - 允许在 active 或 bid_closed 状态下接受投标
    if (project.status !== 'active' && project.status !== 'bid_closed') {
      return res.status(400).json({ error: '项目不在可接受投标状态' });
    }
    
    // 更新投标状态为已接受
    const updated = await Bid.updateStatus(bidId, 'won');
    if (!updated) {
      return res.status(500).json({ error: '接受投标失败' });
    }
    
    // 可以选择将项目状态改为已分配或关闭
    // await Project.updateStatus(bid.project_id, 'assigned');
    
    res.json({ message: '投标已接受' });
  } catch (error) {
    console.error('接受投标错误:', error);
    res.status(500).json({ error: '接受投标失败' });
  }
});

// 拒绝投标
router.post('/:id/reject', authenticateToken, requireRole(['publisher']), async (req, res) => {
  try {
    const bidId = req.params.id;
    
    // 获取投标信息
    const bid = await Bid.findById(bidId);
    if (!bid) {
      return res.status(404).json({ error: '投标不存在' });
    }
    
    // 检查是否是项目发布者
    const project = await Project.findById(bid.project_id);
    if (!project || project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只有项目发布者可以拒绝投标' });
    }
    
    // 更新投标状态为已拒绝
    const updated = await Bid.updateStatus(bidId, 'rejected');
    if (!updated) {
      return res.status(500).json({ error: '拒绝投标失败' });
    }
    
    res.json({ message: '投标已拒绝' });
  } catch (error) {
    console.error('拒绝投标错误:', error);
    res.status(500).json({ error: '拒绝投标失败' });
  }
});

module.exports = router;
