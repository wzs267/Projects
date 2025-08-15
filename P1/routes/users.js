const express = require('express');
const { body, validationResult } = require('express-validator');
const User = require('../models/User');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();

// 更新用户资料
router.put('/profile', authenticateToken, [
  body('username').optional().isLength({ min: 2, max: 50 }).withMessage('用户名长度应在2-50字符之间'),
  body('phone').optional().isMobilePhone('zh-CN').withMessage('请输入有效的手机号码')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ error: errors.array()[0].msg });
    }

    const { username, phone } = req.body;
    const updated = await User.updateProfile(req.user.id, { username, phone });
    
    if (!updated) {
      return res.status(500).json({ error: '更新失败' });
    }

    const user = await User.findById(req.user.id);
    res.json({
      message: '资料更新成功',
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        phone: user.phone,
        userType: user.user_type
      }
    });
  } catch (error) {
    console.error('更新用户资料错误:', error);
    res.status(500).json({ error: '更新失败' });
  }
});

module.exports = router;
