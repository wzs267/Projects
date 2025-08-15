const jwt = require('jsonwebtoken');
const User = require('../models/User');

const authenticateToken = async (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: '访问令牌缺失' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await User.findById(decoded.id);
    
    if (!user) {
      return res.status(401).json({ error: '用户不存在' });
    }

    if (user.status && user.status !== 'active') {
      return res.status(401).json({ error: '用户账户已被禁用' });
    }

    req.user = user;
    next();
  } catch (error) {
    return res.status(403).json({ error: '无效的访问令牌' });
  }
};

const requireRole = (roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: '未认证的用户' });
    }

    if (!roles.includes(req.user.user_type)) {
      return res.status(403).json({ error: '权限不足' });
    }

    next();
  };
};

module.exports = {
  authenticateToken,
  requireRole
};
