const express = require('express');
const { body, validationResult } = require('express-validator');
const Project = require('../models/Project');
const { authenticateToken, requireRole } = require('../middleware/auth');

const router = express.Router();

// 获取项目列表
router.get('/', async (req, res) => {
  try {
    const { page = 1, limit = 10, status, category, search } = req.query;
    
    const filters = {
      page: parseInt(page),
      limit: parseInt(limit),
      status,
      category,
      search
    };

    const projects = await Project.getAll(filters);
    res.json({ projects });
  } catch (error) {
    console.error('获取项目列表错误:', error);
    res.status(500).json({ error: '获取项目列表失败' });
  }
});

// 获取单个项目详情
router.get('/:id', async (req, res) => {
  try {
    const project = await Project.findById(req.params.id);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }

    res.json({ project });
  } catch (error) {
    console.error('获取项目详情错误:', error);
    res.status(500).json({ error: '获取项目详情失败' });
  }
});

// 发布项目（需要发布方权限）
router.post('/', authenticateToken, requireRole(['publisher']), [
  body('title').isLength({ min: 5, max: 200 }).withMessage('项目标题长度应在5-200字符之间'),
  body('description').isLength({ min: 20, max: 800 }).withMessage('项目描述长度应在20-800字符之间'),
  body('budget').optional().isNumeric().withMessage('预算必须是数字'),
  body('work_deadline').isISO8601().withMessage('工期要求格式无效'),
  body('bid_deadline').isISO8601().withMessage('投标截止时间格式无效'),
  body('contact_info').notEmpty().withMessage('联系方式不能为空'),
  body('category').notEmpty().withMessage('项目类别不能为空'),
  body('is_public').optional().isBoolean().withMessage('公开状态必须是布尔值')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: errors.array()[0].msg });
    }

    // 验证时间逻辑
    const now = new Date();
    const bidDeadline = new Date(req.body.bid_deadline);
    const workDeadline = new Date(req.body.work_deadline);
    
    if (bidDeadline <= now) {
      return res.status(400).json({ error: '投标截止时间必须大于当前时间' });
    }
    
    if (workDeadline <= bidDeadline) {
      return res.status(400).json({ error: '工期要求必须大于投标截止时间' });
    }

    const projectData = {
      ...req.body,
      user_id: req.user.id,
      deadline: req.body.work_deadline // 使用工期作为主要截止时间
    };

    const projectId = await Project.create(projectData);
    const project = await Project.findById(projectId);

    res.status(201).json({
      message: '项目发布成功',
      project
    });
  } catch (error) {
    console.error('发布项目错误:', error);
    res.status(500).json({ error: '发布项目失败' });
  }
});

// 更新项目
router.put('/:id', authenticateToken, requireRole(['publisher']), [
  body('title').optional().isLength({ min: 5, max: 200 }).withMessage('项目标题长度应在5-200字符之间'),
  body('description').optional().isLength({ min: 20 }).withMessage('项目描述至少20字符'),
  body('budget').optional().isNumeric().withMessage('预算必须是数字'),
  body('deadline').optional().isISO8601().withMessage('截止日期格式无效')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: errors.array()[0].msg });
    }

    const project = await Project.findById(req.params.id);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }

    // 检查是否是项目发布者
    if (project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只有项目发布者可以修改项目' });
    }

    const updated = await Project.update(req.params.id, req.body);
    if (!updated) {
      return res.status(500).json({ error: '项目更新失败' });
    }

    const updatedProject = await Project.findById(req.params.id);
    res.json({
      message: '项目更新成功',
      project: updatedProject
    });
  } catch (error) {
    console.error('更新项目错误:', error);
    res.status(500).json({ error: '更新项目失败' });
  }
});

// 删除项目 - 已移动到下方，避免重复路由

// 获取我发布的项目
router.get('/my/published', authenticateToken, requireRole(['publisher']), async (req, res) => {
  try {
    const { page = 1, limit = 10, status } = req.query;
    
    const filters = {
      page: parseInt(page),
      limit: parseInt(limit),
      status,
      user_id: req.user.id
    };

    const projects = await Project.getAll(filters);
    res.json({ projects });
  } catch (error) {
    console.error('获取我的项目错误:', error);
    res.status(500).json({ error: '获取我的项目失败' });
  }
});

// 更新项目状态
router.patch('/:id/status', authenticateToken, requireRole(['publisher']), [
  body('status').isIn(['active', 'completed', 'cancelled']).withMessage('状态值无效')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: errors.array()[0].msg });
    }

    const project = await Project.findById(req.params.id);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }

    // 检查是否是项目发布者
    if (project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只有项目发布者可以修改项目状态' });
    }

    const updated = await Project.updateStatus(req.params.id, req.body.status);
    if (!updated) {
      return res.status(500).json({ error: '状态更新失败' });
    }

    res.json({ message: '状态更新成功' });
  } catch (error) {
    console.error('更新项目状态错误:', error);
    res.status(500).json({ error: '更新项目状态失败' });
  }
});

// 更新项目信息
router.put('/:id', authenticateToken, requireRole(['publisher']), [
  body('title').isLength({ min: 5 }).withMessage('项目标题至少5个字符'),
  body('description').isLength({ min: 20 }).withMessage('项目描述至少20个字符'),
  body('budget').isNumeric().withMessage('预算必须是数字'),
  body('deadline').isISO8601().withMessage('截止时间格式无效'),
  body('requirements').isLength({ min: 10 }).withMessage('项目需求至少10个字符'),
  body('contact_info').isLength({ min: 5 }).withMessage('联系方式至少5个字符'),
  body('category').notEmpty().withMessage('请选择项目类别')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: errors.array()[0].msg });
    }

    const projectId = req.params.id;
    
    // 检查项目是否存在并且是当前用户发布的
    const existingProject = await Project.findById(projectId);
    if (!existingProject) {
      return res.status(404).json({ error: '项目不存在' });
    }
    
    if (existingProject.user_id !== req.user.id) {
      return res.status(403).json({ error: '只能修改自己发布的项目' });
    }

    const projectData = {
      title: req.body.title,
      description: req.body.description,
      budget: req.body.budget,
      deadline: req.body.deadline,
      requirements: req.body.requirements,
      contact_info: req.body.contact_info,
      category: req.body.category
    };

    const updated = await Project.update(projectId, projectData);
    if (!updated) {
      return res.status(500).json({ error: '更新项目失败' });
    }

    res.json({ message: '项目更新成功' });
  } catch (error) {
    console.error('更新项目错误:', error);
    res.status(500).json({ error: '更新项目失败' });
  }
});

// 关闭项目
router.post('/:id/close', authenticateToken, requireRole(['publisher']), async (req, res) => {
  try {
    const projectId = req.params.id;
    
    // 检查项目是否存在并且是当前用户发布的
    const project = await Project.findById(projectId);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }
    
    if (project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只能关闭自己发布的项目' });
    }
    
    if (project.status !== 'active') {
      return res.status(400).json({ error: '只能关闭活跃状态的项目' });
    }

    const updated = await Project.updateStatus(projectId, 'closed');
    if (!updated) {
      return res.status(500).json({ error: '关闭项目失败' });
    }

    res.json({ message: '项目已关闭' });
  } catch (error) {
    console.error('关闭项目错误:', error);
    res.status(500).json({ error: '关闭项目失败' });
  }
});

// 获取项目的投标列表
router.get('/:id/bids', authenticateToken, requireRole(['publisher', 'admin']), async (req, res) => {
  try {
    const projectId = req.params.id;
    
    // 检查项目是否存在
    const project = await Project.findById(projectId);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }
    
    // 检查权限：只有项目发布者和管理员可以查看投标
    if (req.user.userType === 'publisher' && project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只能查看自己项目的投标' });
    }
    
    // 获取投标列表
    const Bid = require('../models/Bid');
    const bids = await Bid.getByProject(projectId);
    
    res.json({ bids });
  } catch (error) {
    console.error('获取项目投标列表错误:', error);
    res.status(500).json({ error: '获取投标列表失败' });
  }
});

// 复制项目
router.post('/:id/copy', authenticateToken, requireRole(['publisher']), async (req, res) => {
  try {
    const projectId = req.params.id;
    const project = await Project.findById(projectId);
    
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }
    
    // 检查权限：只有项目发布者可以复制自己的项目
    if (project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只能复制自己的项目' });
    }
    
    // 创建项目副本
    const copyData = {
      title: `${project.title} (副本)`,
      description: project.description,
      requirements: project.requirements,
      category: project.category,
      budget: project.budget,
      work_deadline: project.work_deadline,
      bid_deadline: project.bid_deadline,
      status: 'draft', // 复制的项目设为草稿状态
      user_id: req.user.id
    };
    
    const newProjectId = await Project.create(copyData);
    
    res.json({ 
      message: '项目复制成功',
      projectId: newProjectId 
    });
  } catch (error) {
    console.error('复制项目错误:', error);
    res.status(500).json({ error: '复制项目失败' });
  }
});

// 删除项目
router.delete('/:id', authenticateToken, requireRole(['publisher']), async (req, res) => {
  try {
    const projectId = req.params.id;
    const project = await Project.findById(projectId);
    
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }
    
    // 检查权限：只有项目发布者可以删除自己的项目
    if (project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只能删除自己的项目' });
    }
    
    // 检查是否可以删除：已分配、进行中、已完成或已取消的项目不能删除
    const restrictedStatuses = ['assigned', 'in_progress', 'completed', 'cancelled'];
    if (restrictedStatuses.includes(project.status)) {
      return res.status(400).json({ error: '该项目状态不允许删除' });
    }
    
    // 如果项目有投标，先自动拒绝所有投标
    const Bid = require('../models/Bid');
    const bids = await Bid.getByProject(projectId);
    
    if (bids.length > 0) {
      // 拒绝所有未处理的投标
      for (const bid of bids) {
        if (bid.status === 'pending') {
          await Bid.updateStatus(bid.id, 'rejected');
        }
      }
    }
    
    // 删除项目
    await Project.delete(projectId);
    
    res.json({ message: '项目已删除' });
  } catch (error) {
    console.error('删除项目错误:', error);
    res.status(500).json({ error: '删除项目失败' });
  }
});

// 接受投标
router.post('/:id/accept-bid', authenticateToken, requireRole(['publisher']), async (req, res) => {
  try {
    const projectId = req.params.id;
    const { bidId } = req.body;
    
    console.log('接受投标请求 - projectId:', projectId, 'bidId:', bidId, 'req.body:', req.body);
    
    if (!bidId) {
      return res.status(400).json({ error: '投标ID不能为空' });
    }
    
    const project = await Project.findById(projectId);
    if (!project) {
      return res.status(404).json({ error: '项目不存在' });
    }
    
    // 检查权限：只有项目发布者可以接受投标
    if (project.user_id !== req.user.id) {
      return res.status(403).json({ error: '只能接受自己项目的投标' });
    }
    
    // 检查项目状态
    if (project.status !== 'active') {
      return res.status(400).json({ error: '只有活跃状态的项目可以接受投标' });
    }
    
    const Bid = require('../models/Bid');
    const bid = await Bid.findById(bidId);
    
    console.log('Found bid:', bid);
    
    if (!bid || bid.project_id !== parseInt(projectId)) {
      return res.status(404).json({ error: '投标不存在' });
    }
    
    // 更新投标状态为中标（这会自动处理其他投标和项目状态）
    await Bid.updateStatus(bidId, 'won');
    
    res.json({ message: '投标已接受，项目进入进行中状态' });
  } catch (error) {
    console.error('接受投标错误:', error);
    res.status(500).json({ error: '接受投标失败' });
  }
});

module.exports = router;
