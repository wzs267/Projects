const express = require('express');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { authenticateToken } = require('../middleware/auth');
const db = require('../config/database');
const router = express.Router();

// 生成JWT token
function generateToken(user) {
    return jwt.sign(
        { 
            id: user.id, 
            username: user.username, 
            userType: user.user_type,
            phone: user.phone 
        },
        process.env.JWT_SECRET || 'your-secret-key',
        { expiresIn: '7d' }
    );
}

// 生成随机验证码
function generateCode(length = 6) {
    const chars = '0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
}

// 发送短信验证码
router.post('/send-sms', async (req, res) => {
    try {
        const { phone } = req.body;
        
        if (!phone) {
            return res.status(400).json({ message: '手机号不能为空' });
        }
        
        // 验证手机号格式
        const phoneRegex = /^1[3-9]\d{9}$/;
        if (!phoneRegex.test(phone)) {
            return res.status(400).json({ message: '手机号格式不正确' });
        }
        
        // 检查是否频繁发送
        const recentCode = await db.execute(
            'SELECT * FROM verification_codes WHERE phone = ? AND type = "sms" AND created_at > DATE_SUB(NOW(), INTERVAL 1 MINUTE)',
            [phone]
        );
        
        if (recentCode[0].length > 0) {
            return res.status(429).json({ message: '发送太频繁，请稍后再试' });
        }
        
        // 生成验证码
        const code = generateCode(6);
        const expiresAt = new Date(Date.now() + 5 * 60 * 1000); // 5分钟后过期
        
        // 保存验证码到数据库
        await db.execute(
            'INSERT INTO verification_codes (phone, code, type, expires_at) VALUES (?, ?, "sms", ?)',
            [phone, code, expiresAt]
        );
        
        // 这里应该调用真实的短信服务，现在只是模拟
        console.log(`发送短信验证码到 ${phone}: ${code}`);
        
        res.json({ message: '验证码已发送' });
    } catch (error) {
        console.error('发送短信验证码错误:', error);
        res.status(500).json({ message: '发送失败' });
    }
});

// 发送重置密码短信验证码
router.post('/send-reset-sms', async (req, res) => {
    try {
        const { phone } = req.body;
        
        if (!phone) {
            return res.status(400).json({ message: '手机号不能为空' });
        }
        
        // 检查用户是否存在
        const [users] = await db.execute(
            'SELECT id FROM users WHERE phone = ?',
            [phone]
        );
        
        if (users.length === 0) {
            return res.status(404).json({ message: '该手机号未注册' });
        }
        
        // 检查是否频繁发送
        const recentCode = await db.execute(
            'SELECT * FROM verification_codes WHERE phone = ? AND type = "sms" AND created_at > DATE_SUB(NOW(), INTERVAL 1 MINUTE)',
            [phone]
        );
        
        if (recentCode[0].length > 0) {
            return res.status(429).json({ message: '发送太频繁，请稍后再试' });
        }
        
        // 生成验证码
        const code = generateCode(6);
        const expiresAt = new Date(Date.now() + 5 * 60 * 1000); // 5分钟后过期
        
        // 保存验证码到数据库
        await db.execute(
            'INSERT INTO verification_codes (phone, code, type, expires_at) VALUES (?, ?, "sms", ?)',
            [phone, code, expiresAt]
        );
        
        // 这里应该调用真实的短信服务，现在只是模拟
        console.log(`发送重置密码验证码到 ${phone}: ${code}`);
        
        res.json({ message: '验证码已发送' });
    } catch (error) {
        console.error('发送重置密码短信验证码错误:', error);
        res.status(500).json({ message: '发送失败' });
    }
});

// 验证短信验证码
async function verifySMSCode(phone, code) {
    const [codes] = await db.execute(
        'SELECT * FROM verification_codes WHERE phone = ? AND code = ? AND type = "sms" AND used = 0 AND expires_at > NOW() ORDER BY created_at DESC LIMIT 1',
        [phone, code]
    );
    
    if (codes.length > 0) {
        // 标记验证码为已使用
        await db.execute(
            'UPDATE verification_codes SET used = 1 WHERE id = ?',
            [codes[0].id]
        );
        return true;
    }
    
    return false;
}

// 简单登录（只需手机号和密码）
router.post('/simple-login', async (req, res) => {
    try {
        const { phone, password, userType } = req.body;
        
        if (!phone || !password || !userType) {
            return res.status(400).json({ message: '手机号、密码和用户类型不能为空' });
        }
        
        // 验证手机号格式
        const phoneRegex = /^1[3-9]\d{9}$/;
        if (!phoneRegex.test(phone)) {
            return res.status(400).json({ message: '手机号格式不正确' });
        }
        
        // 查找用户
        const [users] = await db.execute(
            'SELECT * FROM users WHERE phone = ?',
            [phone]
        );
        
        if (users.length === 0) {
            return res.status(401).json({ message: '手机号未注册' });
        }
        
        const user = users[0];
        
        // 检查用户类型
        if (user.user_type !== userType) {
            return res.status(401).json({ message: '用户类型不匹配' });
        }
        
        // 检查用户是否已设置密码
        if (!user.password) {
            return res.status(401).json({ message: '该账号尚未设置密码，请使用短信验证码登录' });
        }
        
        // 验证密码
        const isPasswordValid = await bcrypt.compare(password, user.password);
        if (!isPasswordValid) {
            return res.status(401).json({ message: '密码错误' });
        }
        
        // 生成token
        const token = generateToken(user);
        
        // 返回用户信息
        const userInfo = {
            id: user.id,
            username: user.username,
            phone: user.phone,
            email: user.email,
            userType: user.user_type,
            profession: user.profession,
            avatar: user.avatar,
            real_name_verified: user.real_name_verified,
            created_at: user.created_at
        };
        
        res.json({
            message: '登录成功',
            token,
            user: userInfo
        });
        
    } catch (error) {
        console.error('简单登录错误:', error);
        res.status(500).json({ message: '服务器错误' });
    }
});

// 获取用户类型
router.post('/get-user-type', async (req, res) => {
    try {
        const { phone } = req.body;
        
        if (!phone) {
            return res.status(400).json({ message: '手机号不能为空' });
        }
        
        // 查找用户
        const [users] = await db.execute(
            'SELECT user_type FROM users WHERE phone = ?',
            [phone]
        );
        
        if (users.length === 0) {
            return res.status(404).json({ message: '用户不存在' });
        }
        
        res.json({
            userType: users[0].user_type
        });
        
    } catch (error) {
        console.error('获取用户类型错误:', error);
        res.status(500).json({ message: '服务器错误' });
    }
});

// 登录或注册
router.post('/login-or-register', async (req, res) => {
    try {
        const { phone, password, smsCode, userType } = req.body;
        
        if (!phone || !smsCode || !userType) {
            return res.status(400).json({ message: '请填写所有必填项' });
        }
        
        // 验证短信验证码
        const smsValid = await verifySMSCode(phone, smsCode);
        if (!smsValid) {
            return res.status(400).json({ message: '短信验证码错误或已过期' });
        }
        
        // 检查用户是否存在
        const [existingUsers] = await db.execute(
            'SELECT * FROM users WHERE phone = ?',
            [phone]
        );
        
        let user;
        let isNewUser = false;
        
        if (existingUsers.length > 0) {
            // 用户已存在，进行登录
            user = existingUsers[0];
            
            // 如果提供了密码，验证密码
            if (password && user.password) {
                const passwordValid = await bcrypt.compare(password, user.password);
                if (!passwordValid) {
                    return res.status(400).json({ message: '密码错误' });
                }
            }
        } else {
            // 用户不存在，自动注册
            isNewUser = true;
            
            const username = `用户${phone.slice(-4)}${Date.now().toString().slice(-4)}`;
            const email = `${phone}@temp.com`; // 临时邮箱
            
            let hashedPassword = '';
            if (password) {
                hashedPassword = await bcrypt.hash(password, 10);
            }
            
            const [result] = await db.execute(
                'INSERT INTO users (username, email, phone, password, user_type, has_password, first_login) VALUES (?, ?, ?, ?, ?, ?, 1)',
                [username, email, phone, hashedPassword, userType, password ? 1 : 0]
            );
            
            // 获取新创建的用户
            const [newUsers] = await db.execute(
                'SELECT * FROM users WHERE id = ?',
                [result.insertId]
            );
            user = newUsers[0];
        }
        
        // 生成token
        const token = generateToken(user);
        
        // 返回用户信息（不包含密码）
        const { password: _, ...userInfo } = user;
        
        res.json({
            message: isNewUser ? '注册成功' : '登录成功',
            token,
            user: userInfo,
            isNewUser
        });
        
    } catch (error) {
        console.error('登录或注册错误:', error);
        res.status(500).json({ message: '登录失败' });
    }
});

// 设置密码
router.post('/set-password', authenticateToken, async (req, res) => {
    try {
        const { password } = req.body;
        const userId = req.user.id;
        
        if (!password || password.length < 6) {
            return res.status(400).json({ message: '密码长度至少6位' });
        }
        
        const hashedPassword = await bcrypt.hash(password, 10);
        
        await db.execute(
            'UPDATE users SET password = ?, has_password = 1 WHERE id = ?',
            [hashedPassword, userId]
        );
        
        res.json({ message: '密码设置成功' });
    } catch (error) {
        console.error('设置密码错误:', error);
        res.status(500).json({ message: '设置失败' });
    }
});

// 设置职业技能
router.post('/set-skills', authenticateToken, async (req, res) => {
    try {
        const { profession, skills, experienceYears } = req.body;
        const userId = req.user.id;
        
        if (!profession || !skills || !experienceYears) {
            return res.status(400).json({ message: '请填写所有字段' });
        }
        
        await db.execute(
            'UPDATE users SET profession = ?, specialties = ?, experience_years = ?, first_login = 0 WHERE id = ?',
            [profession, skills, experienceYears, userId]
        );
        
        res.json({ message: '设置成功' });
    } catch (error) {
        console.error('设置技能错误:', error);
        res.status(500).json({ message: '设置失败' });
    }
});

// 重置密码
router.post('/reset-password', async (req, res) => {
    try {
        const { phone, smsCode, newPassword } = req.body;
        
        if (!phone || !smsCode || !newPassword) {
            return res.status(400).json({ message: '请填写所有字段' });
        }
        
        if (newPassword.length < 6) {
            return res.status(400).json({ message: '密码长度至少6位' });
        }
        
        // 验证短信验证码
        const smsValid = await verifySMSCode(phone, smsCode);
        if (!smsValid) {
            return res.status(400).json({ message: '短信验证码错误或已过期' });
        }
        
        // 检查用户是否存在
        const [users] = await db.execute(
            'SELECT id FROM users WHERE phone = ?',
            [phone]
        );
        
        if (users.length === 0) {
            return res.status(404).json({ message: '该手机号未注册' });
        }
        
        // 更新密码
        const hashedPassword = await bcrypt.hash(newPassword, 10);
        await db.execute(
            'UPDATE users SET password = ?, has_password = 1 WHERE phone = ?',
            [hashedPassword, phone]
        );
        
        res.json({ message: '密码重置成功' });
    } catch (error) {
        console.error('重置密码错误:', error);
        res.status(500).json({ message: '重置失败' });
    }
});

// 获取当前用户信息
router.get('/me', authenticateToken, async (req, res) => {
    try {
        const [users] = await db.execute(
            'SELECT id, username, email, phone, user_type, status, created_at FROM users WHERE id = ?',
            [req.user.id]
        );
        
        if (users.length === 0) {
            return res.status(404).json({ error: '用户不存在' });
        }
        
        const user = users[0];
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

module.exports = router;
