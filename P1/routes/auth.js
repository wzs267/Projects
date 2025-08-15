const express = require('express');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { body, validationResult } = require('express-validator');
const User = require('../models/User');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();

// 用户注册
router.post('/register', [
  body('username').isLength({ min: 2, max: 50 }).withMessage('用户名长度应在2-50字符之间'),
  body('email').isEmail().withMessage('请输入有效的邮箱地址'),
  body('password').isLength({ min: 6 }).withMessage('密码长度至少6位'),
  body('phone').isMobilePhone('zh-CN').withMessage('请输入有效的手机号码'),
  body('userType').isIn(['publisher', 'bidder']).withMessage('用户类型无效')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: errors.array()[0].msg });
    }

    const { username, email, password, phone, userType } = req.body;

    // 检查邮箱是否已存在
    const existingUser = await User.findByEmail(email);
    if (existingUser) {
      return res.status(400).json({ error: '邮箱已被注册' });
    }

    // 检查手机号是否已存在
    const existingPhoneUser = await User.findByPhone(phone);
    if (existingPhoneUser) {
      return res.status(400).json({ error: '手机号已被注册' });
    }

    // 加密密码
    const hashedPassword = await bcrypt.hash(password, 10);

    // 创建用户
    const userId = await User.create({
      username,
      email,
      password: hashedPassword,
      phone,
      userType
    });

    // 生成JWT令牌
    const token = jwt.sign(
      { userId, email, userType },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_EXPIRES_IN }
    );

    res.status(201).json({
      message: '注册成功',
      token,
      user: {
        id: userId,
        username,
        email,
        userType
      }
    });
  } catch (error) {
    console.error('注册错误:', error);
    res.status(500).json({ error: '注册失败，请稍后重试' });
  }
});

// 用户登录
router.post('/login', [
  body('email').isEmail().withMessage('请输入有效的邮箱地址'),
  body('password').notEmpty().withMessage('密码不能为空')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: errors.array()[0].msg });
    }

    const { email, password } = req.body;

    // 查找用户
    const user = await User.findByEmail(email);
    if (!user) {
      return res.status(400).json({ error: '邮箱或密码错误' });
    }

    // 检查用户状态
    if (user.status !== 'active') {
      return res.status(400).json({ error: '账户已被禁用' });
    }

    // 验证密码
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(400).json({ error: '邮箱或密码错误' });
    }

    // 生成JWT令牌
    const token = jwt.sign(
      { userId: user.id, email: user.email, userType: user.user_type },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_EXPIRES_IN }
    );

    res.json({
      message: '登录成功',
      token,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        userType: user.user_type
      }
    });
  } catch (error) {
    console.error('登录错误:', error);
    res.status(500).json({ error: '登录失败，请稍后重试' });
  }
});

// 获取当前用户信息
router.get('/me', authenticateToken, async (req, res) => {
  try {
    const user = await User.findById(req.user.id);
    if (!user) {
      return res.status(404).json({ error: '用户不存在' });
    }

    res.json({
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        phone: user.phone,
        userType: user.user_type,
        status: user.status,
        createdAt: user.created_at
      }
    });
  } catch (error) {
    console.error('获取用户信息错误:', error);
    res.status(500).json({ error: '获取用户信息失败' });
  }
});

// 修改密码
router.put('/password', authenticateToken, [
  body('currentPassword').notEmpty().withMessage('当前密码不能为空'),
  body('newPassword').isLength({ min: 6 }).withMessage('新密码长度至少6位')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: errors.array()[0].msg });
    }

    const { currentPassword, newPassword } = req.body;

    // 获取用户当前密码
    const user = await User.findByEmail(req.user.email);
    const isCurrentPasswordValid = await bcrypt.compare(currentPassword, user.password);
    
    if (!isCurrentPasswordValid) {
      return res.status(400).json({ error: '当前密码错误' });
    }

    // 加密新密码
    const hashedNewPassword = await bcrypt.hash(newPassword, 10);

    // 更新密码
    const updated = await User.updatePassword(req.user.id, hashedNewPassword);
    if (!updated) {
      return res.status(500).json({ error: '密码更新失败' });
    }

    res.json({ message: '密码修改成功' });
  } catch (error) {
    console.error('修改密码错误:', error);
    res.status(500).json({ error: '修改密码失败' });
  }
});

module.exports = router;
