// 全局变量
let currentUser = null;
let currentPage = 1;
let currentEditingProjectId = null;
let currentUserProjects = null;
const API_BASE = '/api';

// 初始化应用
document.addEventListener('DOMContentLoaded', function() {
    // 检查登录状态
    const token = localStorage.getItem('token');
    if (token) {
        validateToken(token);
    }
    
    // 绑定表单事件
    bindFormEvents();
    
    // 加载初始数据
    loadProjects();
});

// 验证token
async function validateToken(token) {
    try {
        const response = await fetch(`${API_BASE}/auth/me`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            currentUser = data.user;
            updateUserInterface();
            // 在确认登录状态后，加载用户数据
            await initializeUserData();
        } else {
            localStorage.removeItem('token');
            showLoginForm();
        }
    } catch (error) {
        console.error('Token验证失败:', error);
        localStorage.removeItem('token');
        showLoginForm();
    }
}

// 初始化用户数据
async function initializeUserData() {
    if (!currentUser) return;
    
    // 根据用户类型加载相应的数据
    if (currentUser.userType === 'publisher') {
        const currentView = document.querySelector('.tab-content.active');
        if (currentView && currentView.id === 'my-projects') {
            await loadUserProjects();
        }
    } else if (currentUser.userType === 'bidder') {
        const currentView = document.querySelector('.tab-content.active');
        if (currentView && currentView.id === 'my-bids') {
            await loadUserBids();
        }
    }
}

// 更新用户界面
function updateUserInterface() {
    const navAuth = document.getElementById('nav-auth');
    const navUser = document.getElementById('nav-user');
    const userName = document.getElementById('user-name');
    
    if (currentUser) {
        navAuth.style.display = 'none';
        navUser.style.display = 'flex';
        userName.textContent = currentUser.username;
        
        // 如果是发布方，显示控制台链接
        if (currentUser.userType === 'publisher') {
            // 可以添加特定于发布方的界面元素
        }
    } else {
        navAuth.style.display = 'flex';
        navUser.style.display = 'none';
    }
}

// 绑定表单事件
function bindFormEvents() {
    // 发布项目表单
    const publishForm = document.getElementById('publish-form');
    if (publishForm) {
        publishForm.addEventListener('submit', handlePublishProject);
    }
    
    // 投标表单
    const bidForm = document.getElementById('bid-form');
    if (bidForm) {
        bidForm.addEventListener('submit', handleSubmitBid);
    }
    
    // 个人资料表单
    const profileForm = document.getElementById('profile-form');
    if (profileForm) {
        profileForm.addEventListener('submit', handleUpdateProfile);
    }
    
    // 初始化新的认证表单（延迟绑定，因为模态框可能还未加载）
    setTimeout(() => {
        initAuthForms();
    }, 100);
}

// 处理登录
async function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    
    try {
        const response = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('token', data.token);
            currentUser = data.user;
            updateUserInterface();
            closeModal('login-modal');
            showAlert('登录成功！', 'success');
            
            // 跳转到控制台
            showSection('dashboard');
            loadUserData();
            
            // 刷新项目列表
            loadProjects(1);
        } else {
            showAlert(data.error || '登录失败', 'error');
        }
    } catch (error) {
        console.error('登录错误:', error);
        showAlert('登录失败，请检查网络连接', 'error');
    }
}

// 处理注册
async function handleRegister(event) {
    event.preventDefault();
    
    const formData = {
        username: document.getElementById('register-username').value,
        email: document.getElementById('register-email').value,
        password: document.getElementById('register-password').value,
        phone: document.getElementById('register-phone').value,
        userType: document.getElementById('register-usertype').value
    };
    
    try {
        const response = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('token', data.token);
            currentUser = data.user;
            updateUserInterface();
            closeModal('register-modal');
            showAlert('注册成功！', 'success');
            
            // 跳转到控制台
            showSection('dashboard');
            loadUserData();
        } else {
            showAlert(data.error || '注册失败', 'error');
        }
    } catch (error) {
        console.error('注册错误:', error);
        showAlert('注册失败，请检查网络连接', 'error');
    }
}

// 处理发布项目
async function handlePublishProject(event) {
    event.preventDefault();
    
    if (!currentUser || currentUser.userType !== 'publisher') {
        showAlert('只有发布方可以发布项目', 'error');
        return;
    }
    
    // 验证必填字段
    const title = document.getElementById('project-title').value.trim();
    const description = document.getElementById('project-description').value.trim();
    const workDeadline = document.getElementById('project-work-deadline').value;
    const bidDeadline = document.getElementById('project-bid-deadline').value;
    const contact = document.getElementById('project-contact').value.trim();
    const category = document.getElementById('project-category').value;
    
    if (!title || !description || !workDeadline || !bidDeadline || !contact || !category) {
        showAlert('请填写所有必填项', 'error');
        return;
    }
    
    // 验证字数限制
    if (description.length > 800) {
        showAlert('项目描述不能超过800字', 'error');
        return;
    }
    
    // 验证时间逻辑
    const now = new Date();
    const bidDeadlineDate = new Date(bidDeadline);
    const workDeadlineDate = new Date(workDeadline);
    
    if (bidDeadlineDate <= now) {
        showAlert('投标截止时间必须大于当前时间', 'error');
        return;
    }
    
    if (workDeadlineDate <= bidDeadlineDate) {
        showAlert('工期要求必须大于投标截止时间', 'error');
        return;
    }
    
    // 处理图片上传
    const uploadedImages = [];
    const imageFiles = document.getElementById('project-images').files;
    
    if (imageFiles.length > 5) {
        showAlert('最多只能上传5张图片', 'error');
        return;
    }
    
    // 这里先存储文件信息，实际项目中需要上传到服务器
    for (let i = 0; i < imageFiles.length; i++) {
        const file = imageFiles[i];
        if (file.size > 5 * 1024 * 1024) { // 5MB限制
            showAlert(`图片 ${file.name} 大小超过5MB`, 'error');
            return;
        }
        
        // 简化处理：这里只存储文件名，实际应该上传到服务器获取URL
        uploadedImages.push({
            name: file.name,
            size: file.size,
            url: URL.createObjectURL(file) // 临时URL，实际应该是服务器URL
        });
    }
    
    const formData = {
        title: title,
        description: description,
        budget: document.getElementById('project-budget').value ? parseFloat(document.getElementById('project-budget').value) : null,
        deadline: workDeadline, // 使用工期作为主要截止时间
        work_deadline: workDeadline,
        bid_deadline: bidDeadline,
        requirements: description, // 合并到描述中
        contact_info: contact,
        category: category,
        is_public: document.getElementById('project-is-public').checked,
        images: uploadedImages
    };
    
    try {
        const url = currentEditingProjectId 
            ? `${API_BASE}/projects/${currentEditingProjectId}`
            : `${API_BASE}/projects`;
        const method = currentEditingProjectId ? 'PUT' : 'POST';
        
        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            closeModal('publish-modal');
            const message = currentEditingProjectId ? '项目更新成功！' : '项目发布成功！';
            showAlert(message, 'success');
            
            // 重置表单和编辑状态
            document.getElementById('publish-form').reset();
            document.getElementById('image-preview').innerHTML = '';
            document.getElementById('publish-btn').textContent = '发布项目';
            document.getElementById('project-form-title').textContent = '发布新项目';
            currentEditingProjectId = null;
            updateCharacterCount();
            
            // 刷新项目列表
            loadProjects();
            loadUserProjects();
        } else {
            showAlert(data.error || '操作失败', 'error');
        }
    } catch (error) {
        console.error('项目操作错误:', error);
        showAlert('操作失败，请检查网络连接', 'error');
    }
}

// 处理提交投标
async function handleSubmitBid(event) {
    event.preventDefault();
    
    if (!currentUser || currentUser.userType !== 'bidder') {
        showAlert('只有投标方可以提交投标', 'error');
        return;
    }
    
    const formData = {
        project_id: parseInt(document.getElementById('bid-project-id').value),
        amount: parseFloat(document.getElementById('bid-amount').value),
        deadline: document.getElementById('bid-deadline').value,
        proposal: document.getElementById('bid-proposal').value
    };
    
    try {
        const response = await fetch(`${API_BASE}/bids`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            closeModal('bid-modal');
            showAlert('投标提交成功！', 'success');
            
            // 刷新投标列表
            loadUserBids();
        } else {
            showAlert(data.error || '投标失败', 'error');
        }
    } catch (error) {
        console.error('提交投标错误:', error);
        showAlert('投标失败，请检查网络连接', 'error');
    }
}

// 处理更新个人资料
async function handleUpdateProfile(event) {
    event.preventDefault();
    
    const formData = {
        username: document.getElementById('profile-username').value,
        phone: document.getElementById('profile-phone').value
    };
    
    try {
        const response = await fetch(`${API_BASE}/users/profile`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentUser = { ...currentUser, ...data.user };
            updateUserInterface();
            showAlert('资料更新成功！', 'success');
        } else {
            showAlert(data.error || '更新失败', 'error');
        }
    } catch (error) {
        console.error('更新资料错误:', error);
        showAlert('更新失败，请检查网络连接', 'error');
    }
}

// 加载项目列表
async function loadProjects(page = 1) {
    const projectsList = document.getElementById('projects-list');
    projectsList.innerHTML = '<div class="loading"></div>';
    
    try {
        const category = document.getElementById('category-filter').value;
        const search = document.getElementById('search-input').value;
        
        let url = `${API_BASE}/projects?page=${page}&limit=12`;
        if (category) url += `&category=${encodeURIComponent(category)}`;
        if (search) url += `&search=${encodeURIComponent(search)}`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (response.ok) {
            await displayProjects(data.projects);
            currentPage = page;
        } else {
            projectsList.innerHTML = '<p>加载项目失败</p>';
        }
    } catch (error) {
        console.error('加载项目错误:', error);
        projectsList.innerHTML = '<p>加载项目失败</p>';
    }
}

// 显示项目列表
async function displayProjects(projects) {
    const projectsList = document.getElementById('projects-list');
    
    if (projects.length === 0) {
        projectsList.innerHTML = '<p>暂无项目</p>';
        return;
    }

    // 如果是投标方，获取用户的投标记录
    let userBids = [];
    if (currentUser && currentUser.userType === 'bidder') {
        try {
            const response = await fetch(`${API_BASE}/bids/my`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });
            if (response.ok) {
                const data = await response.json();
                userBids = data.bids || [];
            }
        } catch (error) {
            console.error('获取用户投标记录失败:', error);
        }
    }
    
    projectsList.innerHTML = projects.map(project => {
        // 检查用户是否已对该项目投标
        const hasUserBid = userBids.some(bid => bid.project_id === project.id);
        
        return `
        <div class="project-card">
            <div class="project-header">
                <h3 class="project-title">${escapeHtml(project.title)}</h3>
                <div class="project-meta">
                    <span>类别: ${escapeHtml(project.category)}</span>
                    <span>发布者: ${escapeHtml(project.publisher_name)}</span>
                </div>
            </div>
            <p class="project-description">${escapeHtml(project.description)}</p>
            <div class="project-footer">
                <span class="project-budget">¥${project.budget.toLocaleString()}</span>
                <div>
                    <button class="btn btn-secondary" onclick="viewProject(${project.id})">查看详情</button>
                    ${currentUser && currentUser.userType === 'bidder' ? 
                        (hasUserBid ? 
                            `<button class="btn btn-success" disabled>已投标</button>` :
                            (project.status === 'active' ? 
                                `<button class="btn btn-primary" onclick="showBidModal(${project.id})">投标</button>` :
                                `<button class="btn btn-secondary" disabled>投标已截止</button>`
                            )
                        ) : ''
                    }
                </div>
            </div>
        </div>
        `;
    }).join('');
}

// 加载用户数据
function loadUserData() {
    if (!currentUser) return;
    
    // 加载个人资料
    loadUserProfile();
    
    // 根据用户类型加载相应数据
    if (currentUser.userType === 'publisher') {
        loadUserProjects();
    } else if (currentUser.userType === 'bidder') {
        loadUserBids();
    }
}

// 加载用户个人资料
function loadUserProfile() {
    document.getElementById('profile-username').value = currentUser.username || '';
    document.getElementById('profile-email').value = currentUser.email || '';
    document.getElementById('profile-phone').value = currentUser.phone || '';
}

// 加载用户项目
async function loadUserProjects() {
    // 检查token是否存在
    const token = localStorage.getItem('token');
    if (!token) {
        showLoginForm();
        return;
    }
    
    // 如果用户信息未加载，先验证token
    if (!currentUser) {
        await validateToken(token);
        if (!currentUser) return;
    }
    
    if (currentUser.userType !== 'publisher') return;
    
    const userProjects = document.getElementById('user-projects');
    userProjects.innerHTML = '<div class="loading"></div>';
    
    try {
        const response = await fetch(`${API_BASE}/projects/my/published`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayUserProjects(data.projects);
        } else {
            if (response.status === 401) {
                // Token已过期，清除并显示登录界面
                localStorage.removeItem('token');
                currentUser = null;
                showLoginForm();
                return;
            }
            userProjects.innerHTML = '<p>加载项目失败</p>';
        }
    } catch (error) {
        console.error('加载用户项目错误:', error);
        userProjects.innerHTML = '<p>加载项目失败</p>';
    }
}

// 显示用户项目
function displayUserProjects(projects) {
    const userProjects = document.getElementById('user-projects');
    
    if (projects.length === 0) {
        userProjects.innerHTML = '<div class="no-data">您还没有发布任何项目</div>';
        return;
    }
    
    userProjects.innerHTML = projects.map(project => `
        <div class="project-management-card">
            <div class="project-card-header">
                <div class="project-info-section">
                    <h3 class="project-title-large">${escapeHtml(project.title)}</h3>
                    <div class="project-meta-info">
                        <span>类别: ${escapeHtml(project.category)}</span>
                        <span>预算: ¥${project.budget ? project.budget.toLocaleString() : '面议'}</span>
                        <span>投标数: ${project.bid_count || 0}</span>
                        <span>发布时间: ${formatDateTime(project.created_at)}</span>
                    </div>
                    <div class="project-deadlines">
                        <small>投标截止: ${formatDateTime(project.bid_deadline || project.deadline)}</small>
                        <small>工期要求: ${formatDateTime(project.work_deadline || project.deadline)}</small>
                    </div>
                </div>
                <div class="project-status-section">
                    <div class="project-status-large ${project.status}">${getStatusText(project.status)}</div>
                    <small>查看: ${project.view_count || 0}次</small>
                </div>
            </div>
            
            <p class="project-description">${escapeHtml(project.description.length > 200 ? project.description.substring(0, 200) + '...' : project.description)}</p>
            
            <div class="project-actions-horizontal">
                <button class="btn btn-info" onclick="viewProjectBids(${project.id})">投标管理</button>
                ${project.status === 'active' || project.status === 'bid_closed' ? `
                    <button class="btn btn-primary" onclick="editProjectFromList(${project.id})">更改</button>
                ` : ''}
                <button class="btn btn-secondary" onclick="copyProject(${project.id})">复制</button>
                ${canDeleteProject(project) ? `
                    <button class="btn btn-danger" onclick="deleteProject(${project.id})">删除</button>
                ` : ''}
            </div>
        </div>
    `).join('');
}

// 判断项目是否可以删除
function canDeleteProject(project) {
    // 项目已分配给投标方或已取消后不能删除
    if (project.status === 'assigned' || project.status === 'in_progress' || project.status === 'completed' || project.status === 'cancelled') {
        return false;
    }
    // 其他状态的项目都可以删除（包括有投标但未接受的active和bid_closed状态）
    return true;
}

// 加载用户投标
async function loadUserBids() {
    // 检查token是否存在
    const token = localStorage.getItem('token');
    if (!token) {
        showLoginForm();
        return;
    }
    
    // 如果用户信息未加载，先验证token
    if (!currentUser) {
        await validateToken(token);
        if (!currentUser) return;
    }
    
    if (currentUser.userType !== 'bidder') return;
    
    const userBids = document.getElementById('user-bids');
    userBids.innerHTML = '<div class="loading"></div>';
    
    try {
        const response = await fetch(`${API_BASE}/bids/my`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayUserBids(data.bids);
        } else {
            if (response.status === 401) {
                // Token已过期，清除并显示登录界面
                localStorage.removeItem('token');
                currentUser = null;
                showLoginForm();
                return;
            }
            userBids.innerHTML = '<p>加载投标失败</p>';
        }
    } catch (error) {
        console.error('加载用户投标错误:', error);
        userBids.innerHTML = '<p>加载投标失败</p>';
    }
}

// 显示用户投标
function displayUserBids(bids) {
    const userBids = document.getElementById('user-bids');
    
    if (bids.length === 0) {
        userBids.innerHTML = '<p>您还没有提交任何投标</p>';
        return;
    }
    
    userBids.innerHTML = bids.map(bid => `
        <div class="bid-card">
            <div class="bid-header">
                <h3>${escapeHtml(bid.project_title)}</h3>
                <span class="bid-status ${bid.status}">${getBidStatusText(bid.status)}</span>
            </div>
            <div class="bid-meta">
                <span class="bid-amount">投标金额: ¥${bid.amount.toLocaleString()}</span>
                <span>提交时间: ${formatDate(bid.created_at)}</span>
            </div>
            <p class="bid-proposal">${escapeHtml(bid.proposal.substring(0, 100))}...</p>
        </div>
    `).join('');
}

// 显示指定区域
function showSection(sectionId) {
    // 隐藏所有区域
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // 显示指定区域
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
        
        // 如果显示控制台，加载用户数据
        if (sectionId === 'dashboard' && currentUser) {
            loadUserData();
        }
    }
}

// 显示标签页
function showTab(tabId) {
    // 更新按钮状态
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // 更新内容
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');
    
    // 加载对应数据
    if (tabId === 'my-projects') {
        loadUserProjects();
    } else if (tabId === 'my-bids') {
        loadUserBids();
    } else if (tabId === 'profile') {
        loadUserProfile();
    }
}

// 模态框相关函数
function showModal(modalId) {
    document.getElementById(modalId).style.display = 'block';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function showLogin() {
    showModal('login-modal');
}

function showRegister() {
    showModal('register-modal');
}

function showPublishProject() {
    if (!currentUser) {
        showAlert('请先登录', 'error');
        showLogin();
        return;
    }
    
    if (currentUser.userType !== 'publisher') {
        showAlert('只有发布方可以发布项目', 'error');
        return;
    }
    
    // 重置编辑模式
    currentEditingProjectId = null;
    document.getElementById('publish-btn').textContent = '发布项目';
    document.getElementById('project-form-title').textContent = '发布新项目';
    document.getElementById('publish-form').reset();
    document.getElementById('image-preview').innerHTML = '';
    updateCharacterCount();
    
    showModal('publish-modal');
}

function showBidModal(projectId) {
    if (!currentUser) {
        showAlert('请先登录', 'error');
        showLogin();
        return;
    }
    
    if (currentUser.userType !== 'bidder') {
        showAlert('只有投标方可以提交投标', 'error');
        return;
    }
    
    document.getElementById('bid-project-id').value = projectId;
    showModal('bid-modal');
}

// 搜索项目
function searchProjects() {
    loadProjects(1);
}

// 退出登录
function logout() {
    localStorage.removeItem('token');
    currentUser = null;
    updateUserInterface();
    showSection('home');
    showAlert('已退出登录', 'info');
    
    // 重新加载页面数据
    loadProjects(1);
    
    // 清理可能的用户相关数据
    window.currentProjectId = null;
    window.currentProject = null;
    window.currentUserBid = null;
}

// 工具函数
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('zh-CN');
}

function getStatusText(status) {
    const statusMap = {
        'active': '进行中',
        'in_progress': '进行中',
        'completed': '已完成',
        'cancelled': '已取消',
        'assigned': '已分配',
        'bid_closed': '投标已截止'
    };
    return statusMap[status] || status;
}

function getBidStatusText(status) {
    const statusMap = {
        'pending': '待审核',
        'won': '中标',
        'lost': '未中标',
        'rejected': '已拒绝',
        'withdrawn': '已撤销'
    };
    return statusMap[status] || status;
}

function showAlert(message, type = 'info', isHtml = false) {
    // 创建提示元素
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    
    // 根据是否HTML内容来设置消息
    if (isHtml) {
        alert.innerHTML = message;
    } else {
        alert.textContent = message;
    }
    
    // 添加到页面顶部
    document.body.insertBefore(alert, document.body.firstChild);
    
    // 3秒后自动移除
    setTimeout(() => {
        if (alert.parentNode) {
            alert.parentNode.removeChild(alert);
        }
    }, 3000);
}

// 点击模态框外部关闭模态框
window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.style.display = 'none';
    }
}

// ==================== 项目详情功能 ====================

// 查看项目详情
async function viewProject(projectId) {
    try {
        // 获取项目详情
        const response = await fetch(`${API_BASE}/projects/${projectId}`);
        const data = await response.json();
        
        if (!response.ok) {
            showAlert('获取项目详情失败', 'error');
            return;
        }
        
        const project = data.project;
        displayProjectDetail(project);
        showSection('project-detail');
        
        // 加载投标列表
        if (currentUser && (currentUser.userType === 'publisher' || currentUser.userType === 'admin')) {
            loadProjectBids(projectId);
        }
        
    } catch (error) {
        console.error('获取项目详情错误:', error);
        showAlert('获取项目详情失败', 'error');
    }
}

// 显示项目详情
function displayProjectDetail(project) {
    // 填充基本信息
    document.getElementById('detail-title').textContent = project.title;
    document.getElementById('detail-publisher').textContent = project.publisher_name;
    document.getElementById('detail-budget').textContent = `¥${project.budget.toLocaleString()}`;
    document.getElementById('detail-deadline').textContent = formatDateTime(project.deadline);
    document.getElementById('detail-category').textContent = project.category;
    document.getElementById('detail-bid-count').textContent = project.bid_count || 0;
    document.getElementById('detail-description').textContent = project.description;
    document.getElementById('detail-requirements').textContent = project.requirements;
    document.getElementById('detail-contact').textContent = project.contact_info;
    
    // 设置状态
    const statusElement = document.getElementById('detail-status');
    statusElement.textContent = getStatusText(project.status);
    statusElement.className = `project-status ${project.status}`;
    
    // 根据用户身份显示不同的操作按钮
    showProjectActions(project);
    
    // 存储当前项目ID
    window.currentProjectId = project.id;
    window.currentProject = project;
}

// 根据用户身份显示项目操作
function showProjectActions(project) {
    // 隐藏所有操作组
    document.getElementById('publisher-actions').style.display = 'none';
    document.getElementById('bidder-actions').style.display = 'none';
    document.getElementById('admin-actions').style.display = 'none';
    
    if (!currentUser) {
        return;
    }
    
    // 发布方操作
    if (currentUser.userType === 'publisher' && project.user_id === currentUser.id) {
        document.getElementById('publisher-actions').style.display = 'block';
    }
    // 投标方操作
    else if (currentUser.userType === 'bidder' && project.status === 'active') {
        document.getElementById('bidder-actions').style.display = 'block';
        
        // 重置投标状态UI
        document.getElementById('user-bid-info').style.display = 'none';
        document.querySelector('#bidder-actions .btn-primary').style.display = 'block';
        window.currentUserBid = null;
        
        // 检查用户投标状态
        checkUserBidStatus(project.id);
    }
    // 管理员操作
    else if (currentUser.userType === 'admin') {
        document.getElementById('admin-actions').style.display = 'block';
    }
}

// 检查用户是否已投标
async function checkUserBidStatus(projectId) {
    try {
        // 首先重置UI状态
        document.getElementById('user-bid-info').style.display = 'none';
        document.querySelector('#bidder-actions .btn-primary').style.display = 'block';
        window.currentUserBid = null;
        
        const response = await fetch(`${API_BASE}/bids/my`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            const userBid = data.bids.find(bid => bid.project_id === projectId);
            
            if (userBid) {
                document.getElementById('user-bid-info').style.display = 'block';
                document.querySelector('#bidder-actions .btn-primary').style.display = 'none';
                window.currentUserBid = userBid;
            }
        }
    } catch (error) {
        console.error('检查投标状态错误:', error);
    }
}

// 加载项目投标列表
async function loadProjectBids(projectId) {
    try {
        const response = await fetch(`${API_BASE}/projects/${projectId}/bids`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            displayProjectBids(data.bids);
        }
    } catch (error) {
        console.error('加载投标列表错误:', error);
    }
}

// 显示项目投标列表
function displayProjectBids(bids) {
    const bidsContainer = document.getElementById('project-bids-list');
    
    if (!bids || bids.length === 0) {
        bidsContainer.innerHTML = '<p class="no-data">暂无投标</p>';
        return;
    }
    
    bidsContainer.innerHTML = bids.map(bid => `
        <div class="bid-card">
            <div class="bid-header">
                <div class="bid-bidder">${escapeHtml(bid.bidder_name)}</div>
                <div class="bid-amount">¥${bid.amount.toLocaleString()}</div>
                <div class="bid-status-badge ${bid.status}">${getBidStatusText(bid.status)}</div>
            </div>
            <div class="bid-deadline">完成时间: ${formatDateTime(bid.deadline)}</div>
            <div class="bid-proposal">${escapeHtml(bid.proposal)}</div>
            ${currentUser && currentUser.userType === 'publisher' && window.currentProject.user_id === currentUser.id ? `
                <div class="bid-actions">
                    <button class="btn btn-success btn-sm" onclick="acceptBid(${bid.id})">接受</button>
                    <button class="btn btn-danger btn-sm" onclick="rejectBid(${bid.id})">拒绝</button>
                    <button class="btn btn-info btn-sm" onclick="viewBidDetail(${bid.id})">查看详情</button>
                </div>
            ` : ''}
        </div>
    `).join('');
}

// 编辑项目
function editProject() {
    if (!window.currentProject) return;
    
    const project = window.currentProject;
    
    // 填充编辑表单
    document.getElementById('edit-project-id').value = project.id;
    document.getElementById('edit-project-title').value = project.title;
    document.getElementById('edit-project-description').value = project.description;
    document.getElementById('edit-project-budget').value = project.budget;
    document.getElementById('edit-project-deadline').value = formatDateTimeForInput(project.deadline);
    document.getElementById('edit-project-requirements').value = project.requirements;
    document.getElementById('edit-project-contact').value = project.contact_info;
    document.getElementById('edit-project-category').value = project.category;
    
    // 显示编辑模态框
    document.getElementById('edit-project-modal').style.display = 'block';
}

// 处理编辑项目表单提交
document.getElementById('edit-project-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const projectId = document.getElementById('edit-project-id').value;
    const formData = {
        title: document.getElementById('edit-project-title').value,
        description: document.getElementById('edit-project-description').value,
        budget: document.getElementById('edit-project-budget').value,
        deadline: document.getElementById('edit-project-deadline').value,
        requirements: document.getElementById('edit-project-requirements').value,
        contact_info: document.getElementById('edit-project-contact').value,
        category: document.getElementById('edit-project-category').value
    };
    
    try {
        const response = await fetch(`${API_BASE}/projects/${projectId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showAlert('项目更新成功', 'success');
            closeModal('edit-project-modal');
            // 重新加载项目详情
            viewProject(projectId);
        } else {
            showAlert(data.message || '更新失败', 'error');
        }
    } catch (error) {
        console.error('更新项目错误:', error);
        showAlert('更新失败', 'error');
    }
});

// 显示投标表单
function showBidForm() {
    if (!window.currentProjectId) return;
    
    document.getElementById('bid-project-id').value = window.currentProjectId;
    document.getElementById('bid-modal').style.display = 'block';
}

// 查看我的投标
function viewMyBid() {
    if (!window.currentUserBid) return;
    
    const bid = window.currentUserBid;
    const content = `
        <div class="bid-detail">
            <div class="detail-item">
                <strong>投标金额:</strong> ¥${bid.amount.toLocaleString()}
            </div>
            <div class="detail-item">
                <strong>完成时间:</strong> ${formatDateTime(bid.deadline)}
            </div>
            <div class="detail-item">
                <strong>投标状态:</strong> ${getBidStatusText(bid.status)}
            </div>
            <div class="detail-item">
                <strong>投标方案:</strong>
                <div class="proposal-text">${escapeHtml(bid.proposal)}</div>
            </div>
        </div>
    `;
    
    document.getElementById('bid-detail-content').innerHTML = content;
    document.getElementById('bid-detail-modal').style.display = 'block';
}

// 接受投标
async function acceptBid(bidId) {
    if (!confirm('确定要接受这个投标吗？')) return;
    
    try {
        const response = await fetch(`${API_BASE}/bids/${bidId}/accept`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showAlert('投标已接受', 'success');
            if (window.currentProjectId) {
                loadProjectBids(window.currentProjectId);
            }
            // 刷新当前视图以更新项目状态
            await refreshCurrentView();
        } else {
            if (response.status === 401) {
                localStorage.removeItem('token');
                currentUser = null;
                showLoginForm();
                return;
            }
            showAlert(data.message || '操作失败', 'error');
        }
    } catch (error) {
        console.error('接受投标错误:', error);
        showAlert('操作失败', 'error');
    }
}

// 拒绝投标
async function rejectBid(bidId) {
    if (!confirm('确定要拒绝这个投标吗？')) return;
    
    try {
        const response = await fetch(`${API_BASE}/bids/${bidId}/reject`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showAlert('投标已拒绝', 'success');
            loadProjectBids(window.currentProjectId);
        } else {
            showAlert(data.message || '操作失败', 'error');
        }
    } catch (error) {
        console.error('拒绝投标错误:', error);
        showAlert('操作失败', 'error');
    }
}

// 关闭项目
async function closeProject() {
    if (!confirm('确定要关闭这个项目吗？关闭后将不再接受新的投标。')) return;
    
    try {
        const response = await fetch(`${API_BASE}/projects/${window.currentProjectId}/close`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showAlert('项目已关闭', 'success');
            viewProject(window.currentProjectId);
        } else {
            showAlert(data.message || '操作失败', 'error');
        }
    } catch (error) {
        console.error('关闭项目错误:', error);
        showAlert('操作失败', 'error');
    }
}

// 显示项目投标管理
function showProjectBids() {
    // 滚动到投标列表区域
    document.getElementById('bids-section').scrollIntoView({ behavior: 'smooth' });
}

// 返回上一页
function goBack() {
    showSection('projects');
}

// 格式化日期时间用于输入框
function formatDateTimeForInput(dateString) {
    const date = new Date(dateString);
    return date.getFullYear() + '-' + 
           String(date.getMonth() + 1).padStart(2, '0') + '-' + 
           String(date.getDate()).padStart(2, '0') + 'T' + 
           String(date.getHours()).padStart(2, '0') + ':' + 
           String(date.getMinutes()).padStart(2, '0');
}

// 获取投标状态文本
function getBidStatusText(status) {
    const statusMap = {
        'pending': '待审核',
        'won': '中标',
        'lost': '未中标',
        'rejected': '已拒绝',
        'withdrawn': '已撤销'
    };
    return statusMap[status] || status;
}

// 格式化日期时间显示
function formatDateTime(dateString) {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    
    return `${year}-${month}-${day} ${hours}:${minutes}`;
}

// ==================== 表单辅助函数 ====================

// 字符计数更新
function updateCharacterCount() {
    const textarea = document.getElementById('project-description');
    const counter = document.getElementById('desc-char-count');
    
    if (textarea && counter) {
        textarea.addEventListener('input', function() {
            const count = this.value.length;
            counter.textContent = count;
            
            if (count > 800) {
                counter.style.color = 'red';
            } else if (count > 700) {
                counter.style.color = 'orange';
            } else {
                counter.style.color = '#6c757d';
            }
        });
    }
}

// 图片预览处理
document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('project-images');
    if (imageInput) {
        imageInput.addEventListener('change', handleImagePreview);
    }
    
    // 初始化字符计数
    updateCharacterCount();
});

function handleImagePreview(event) {
    const files = event.target.files;
    const previewContainer = document.getElementById('image-preview');
    
    // 清空现有预览
    previewContainer.innerHTML = '';
    
    if (files.length > 5) {
        showAlert('最多只能上传5张图片', 'error');
        event.target.value = '';
        return;
    }
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        if (file.size > 5 * 1024 * 1024) {
            showAlert(`图片 ${file.name} 大小超过5MB`, 'error');
            continue;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewItem = document.createElement('div');
            previewItem.className = 'image-preview-item';
            previewItem.innerHTML = `
                <img src="${e.target.result}" alt="预览图片">
                <button type="button" class="image-remove-btn" onclick="removeImage(${i})">×</button>
            `;
            previewContainer.appendChild(previewItem);
        };
        reader.readAsDataURL(file);
    }
}

function removeImage(index) {
    const imageInput = document.getElementById('project-images');
    const previewContainer = document.getElementById('image-preview');
    
    // 创建新的文件列表，排除指定索引的文件
    const dt = new DataTransfer();
    const files = imageInput.files;
    
    for (let i = 0; i < files.length; i++) {
        if (i !== index) {
            dt.items.add(files[i]);
        }
    }
    
    imageInput.files = dt.files;
    
    // 重新生成预览
    handleImagePreview({ target: imageInput });
}

// ==================== 发布方投标管理功能 ====================

// 查看项目投标（从我的项目页面）
async function viewProjectBids(projectId) {
    try {
        // 获取项目详情
        const projectResponse = await fetch(`${API_BASE}/projects/${projectId}`);
        const projectData = await projectResponse.json();
        
        if (!projectResponse.ok) {
            showAlert('获取项目信息失败', 'error');
            return;
        }
        
        // 获取投标列表
        const bidsResponse = await fetch(`${API_BASE}/projects/${projectId}/bids`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        const bidsData = await bidsResponse.json();
        
        if (!bidsResponse.ok) {
            showAlert('获取投标列表失败', 'error');
            return;
        }
        
        // 显示投标管理界面
        displayProjectBidsManagement(projectId, bidsData.bids);
        // 注意：不需要手动显示模态框，displayProjectBidsManagement会处理
        
    } catch (error) {
        console.error('获取投标信息错误:', error);
        showAlert('获取投标信息失败', 'error');
    }
}

// 显示投标管理界面
function displayBidsManagement(project, bids) {
    // 填充项目信息
    document.getElementById('bids-project-title').textContent = project.title;
    document.getElementById('bids-project-budget').textContent = `预算: ¥${project.budget.toLocaleString()}`;
    document.getElementById('bids-project-deadline').textContent = `截止: ${formatDateTime(project.deadline)}`;
    document.getElementById('bids-project-status').textContent = `状态: ${getStatusText(project.status)}`;
    
    // 计算统计信息
    const stats = calculateBidsStats(bids);
    document.getElementById('total-bids').textContent = stats.total;
    document.getElementById('pending-bids').textContent = stats.pending;
    document.getElementById('accepted-bids').textContent = stats.accepted;
    
    // 显示投标列表
    displayBidsManagementList(bids);
    
    // 存储当前项目和投标数据
    window.currentManagementProject = project;
    window.currentManagementBids = bids;
}

// 计算投标统计信息
function calculateBidsStats(bids) {
    return {
        total: bids.length,
        pending: bids.filter(bid => bid.status === 'pending').length,
        accepted: bids.filter(bid => bid.status === 'accepted').length,
        rejected: bids.filter(bid => bid.status === 'rejected').length
    };
}

// 显示投标管理列表
function displayBidsManagementList(bids) {
    const container = document.getElementById('bids-management-list');
    
    if (!bids || bids.length === 0) {
        container.innerHTML = '<div class="no-data">暂无投标</div>';
        return;
    }
    
    container.innerHTML = bids.map(bid => `
        <div class="bid-management-card">
            <div class="bid-management-header">
                <div class="bid-bidder-info">
                    <div class="bid-bidder-name">${escapeHtml(bid.bidder_name)}</div>
                    <div class="bid-submit-time">投标时间: ${formatDateTime(bid.created_at)}</div>
                </div>
                <div class="bid-amount-info">
                    <div class="bid-amount-large">¥${bid.amount.toLocaleString()}</div>
                    <div class="bid-deadline-info">完成时间: ${formatDateTime(bid.deadline)}</div>
                </div>
                <div class="bid-status-badge ${bid.status}">${getBidStatusText(bid.status)}</div>
            </div>
            
            <div class="bid-proposal-preview">
                ${escapeHtml(bid.proposal.length > 150 ? bid.proposal.substring(0, 150) + '...' : bid.proposal)}
            </div>
            
            <div class="bid-management-actions">
                <button class="btn btn-info btn-sm" onclick="viewBidDetailInManagement(${bid.id})">查看详情</button>
                ${bid.status === 'pending' ? `
                    <button class="btn btn-success btn-sm" onclick="acceptBidInManagement(${bid.id})">接受</button>
                    <button class="btn btn-danger btn-sm" onclick="rejectBidInManagement(${bid.id})">拒绝</button>
                ` : ''}
            </div>
        </div>
    `).join('');
}

// 筛选投标
function filterBids() {
    const statusFilter = document.getElementById('bids-status-filter').value;
    const bids = window.currentManagementBids || [];
    
    let filteredBids = bids;
    if (statusFilter) {
        filteredBids = bids.filter(bid => bid.status === statusFilter);
    }
    
    displayBidsManagementList(filteredBids);
}

// 在管理界面中接受投标
async function acceptBidInManagement(bidId) {
    if (!confirm('确定要接受这个投标吗？')) return;
    
    try {
        const response = await fetch(`${API_BASE}/bids/${bidId}/accept`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showAlert('投标已接受', 'success');
            // 重新加载投标管理界面
            if (window.currentManagementProject) {
                viewProjectBids(window.currentManagementProject.id);
            }
            // 同时刷新项目列表以更新状态
            await refreshCurrentView();
        } else {
            if (response.status === 401) {
                localStorage.removeItem('token');
                currentUser = null;
                showLoginForm();
                return;
            }
            showAlert(data.message || '操作失败', 'error');
        }
    } catch (error) {
        console.error('接受投标错误:', error);
        showAlert('操作失败', 'error');
    }
}

// 在管理界面中拒绝投标
async function rejectBidInManagement(bidId) {
    if (!confirm('确定要拒绝这个投标吗？')) return;
    
    try {
        const response = await fetch(`${API_BASE}/bids/${bidId}/reject`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showAlert('投标已拒绝', 'success');
            // 重新加载投标管理界面
            viewProjectBids(window.currentManagementProject.id);
        } else {
            showAlert(data.message || '操作失败', 'error');
        }
    } catch (error) {
        console.error('拒绝投标错误:', error);
        showAlert('操作失败', 'error');
    }
}

// 在管理界面中查看投标详情
async function viewBidDetailInManagement(bidId) {
    const bid = window.currentManagementBids.find(b => b.id === bidId);
    if (!bid) return;
    
    const content = `
        <div class="bid-detail">
            <div class="detail-section">
                <h4>投标方信息</h4>
                <div class="detail-item">
                    <strong>投标方:</strong> ${escapeHtml(bid.bidder_name)}
                </div>
                <div class="detail-item">
                    <strong>投标时间:</strong> ${formatDateTime(bid.created_at)}
                </div>
            </div>
            
            <div class="detail-section">
                <h4>投标信息</h4>
                <div class="detail-item">
                    <strong>投标金额:</strong> ¥${bid.amount.toLocaleString()}
                </div>
                <div class="detail-item">
                    <strong>完成时间:</strong> ${formatDateTime(bid.deadline)}
                </div>
                <div class="detail-item">
                    <strong>投标状态:</strong> ${getBidStatusText(bid.status)}
                </div>
            </div>
            
            <div class="detail-section">
                <h4>投标方案</h4>
                <div class="proposal-detail">${escapeHtml(bid.proposal)}</div>
            </div>
        </div>
        
        <style>
        .detail-section {
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .detail-section:last-child {
            border-bottom: none;
        }
        
        .detail-section h4 {
            color: #2c5aa0;
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        .detail-item {
            margin-bottom: 10px;
            display: flex;
            gap: 10px;
        }
        
        .detail-item strong {
            min-width: 100px;
            color: #6c757d;
        }
        
        .proposal-detail {
            line-height: 1.8;
            color: #333;
            white-space: pre-wrap;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }
        </style>
    `;
    
    document.getElementById('bid-detail-content').innerHTML = content;
    document.getElementById('bid-detail-modal').style.display = 'block';
}

// ==================== 项目管理功能 ====================

// 复制项目
async function copyProject(projectId) {
    if (!confirm('确定要复制这个项目吗？复制后会创建一个新的项目草稿。')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/projects/${projectId}/copy`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (response.ok) {
            showAlert('项目复制成功！新项目已创建为草稿状态。', 'success');
            if (currentUserProjects) {
                await loadUserProjects();
            }
        } else {
            const error = await response.json();
            showAlert(error.message || '复制项目失败', 'error');
        }
    } catch (error) {
        console.error('复制项目错误:', error);
        showAlert('复制项目失败', 'error');
    }
}

// 删除项目
async function deleteProject(projectId) {
    if (!confirm('确定要删除这个项目吗？此操作不可撤销！')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/projects/${projectId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (response.ok) {
            showAlert('项目已删除', 'success');
            // 自动刷新当前视图
            await refreshCurrentView();
        } else {
            if (response.status === 401) {
                localStorage.removeItem('token');
                currentUser = null;
                showLoginForm();
                return;
            }
            const error = await response.json();
            showAlert(error.message || '删除项目失败', 'error');
        }
    } catch (error) {
        console.error('删除项目错误:', error);
        showAlert('删除项目失败', 'error');
    }
}

// 从列表编辑项目
async function editProjectFromList(projectId) {
    try {
        // 获取项目详情
        const response = await fetch(`/api/projects/${projectId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (response.ok) {
            const project = await response.json();
            
            // 填充编辑表单并显示编辑项目模态框
            document.getElementById('edit-project-id').value = project.id || '';
            document.getElementById('edit-project-title').value = project.title || '';
            document.getElementById('edit-project-description').value = project.description || '';
            document.getElementById('edit-project-budget').value = project.budget || '';
            document.getElementById('edit-project-deadline').value = formatDateForInput(project.deadline || project.work_deadline);
            document.getElementById('edit-project-requirements').value = project.requirements || '';
            document.getElementById('edit-project-contact').value = project.contact_info || '';
            document.getElementById('edit-project-category').value = project.category || '';
            
            // 显示编辑模态框
            document.getElementById('edit-project-modal').style.display = 'block';
        } else {
            showAlert('获取项目信息失败', 'error');
        }
    } catch (error) {
        console.error('编辑项目错误:', error);
        showAlert('获取项目信息失败', 'error');
    }
}

// 管理项目投标
async function manageProjectBids(projectId) {
    try {
        const response = await fetch(`/api/projects/${projectId}/bids`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            displayProjectBidsManagement(projectId, data.bids || data);
        } else {
            showAlert('获取投标信息失败', 'error');
        }
    } catch (error) {
        console.error('获取投标信息错误:', error);
        showAlert('获取投标信息失败', 'error');
    }
}

// 显示投标管理界面
function displayProjectBidsManagement(projectId, bids) {
    console.log('displayProjectBidsManagement called with:', { projectId, bids });
    
    const content = `
        <div class="bids-management">
            <div class="bids-management-header">
                <h3>投标管理</h3>
                <button class="btn btn-secondary" onclick="closeModal('bids-management-modal')">关闭</button>
            </div>
            <div class="bids-list">
                ${bids.length > 0 ? bids.map(bid => {
                    console.log('Processing bid:', bid);
                    return `
                    <div class="bid-management-item">
                        <div class="bid-info">
                            <h4>${escapeHtml(bid.bidder_name || '')}</h4>
                            <p>报价: ¥${bid.amount ? bid.amount.toLocaleString() : '0'}</p>
                            <p>工期: ${bid.deadline || '未指定'}天</p>
                            <p>投标时间: ${formatDateTime(bid.created_at)}</p>
                            <p class="bid-proposal">${escapeHtml(bid.proposal || '')}</p>
                        </div>
                        <div class="bid-actions">
                            <button class="btn btn-success" onclick="acceptBid(${projectId}, ${bid.id})">接受投标</button>
                            <button class="btn btn-info" onclick="contactBidder(${bid.user_id})">联系投标人</button>
                        </div>
                    </div>
                    `;
                }).join('') : '<p class="no-data">暂无投标</p>'}
            </div>
        </div>
    `;
    
    // 创建或更新投标管理模态框
    let modal = document.getElementById('bids-management-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'bids-management-modal';
        modal.className = 'modal';
        document.body.appendChild(modal);
    }
    
    modal.innerHTML = `<div class="modal-content">${content}</div>`;
    modal.style.display = 'flex';
}

// 接受投标
async function acceptBid(projectId, bidId) {
    console.log('acceptBid called with:', { projectId, bidId, typeof_bidId: typeof bidId });
    
    if (!bidId || bidId === 'undefined') {
        showAlert('投标ID无效', 'error');
        return;
    }
    
    if (!confirm('确定要接受这个投标吗？接受后项目将进入进行中状态。')) {
        return;
    }
    
    try {
        const requestBody = { bidId: parseInt(bidId) };
        console.log('Sending request body:', requestBody);
        
        const response = await fetch(`/api/projects/${projectId}/accept-bid`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify(requestBody)
        });
        
        console.log('Response status:', response.status);
        
        if (response.ok) {
            showAlert('投标已接受！', 'success');
            closeModal('bids-management-modal');
            await loadUserProjects();
        } else {
            const error = await response.json();
            console.log('Error response:', error);
            showAlert(error.message || '接受投标失败', 'error');
        }
    } catch (error) {
        console.error('接受投标错误:', error);
        showAlert('接受投标失败', 'error');
    }
}

// 联系投标人
function contactBidder(userId) {
    // 这里可以实现私信功能或显示联系方式
    showAlert('联系功能将在后续版本中实现', 'info');
}

// 格式化日期为输入框格式
function formatDateForInput(dateString) {
    if (!dateString) {
        return ''; // 如果日期为空，返回空字符串
    }
    
    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
        return ''; // 如果日期无效，返回空字符串
    }
    
    return date.toISOString().slice(0, 16); // YYYY-MM-DDTHH:MM
}

// 关闭模态框
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

// ================ 新认证系统 ================

// 全局变量
let captchaText = '';
let smsCountdown = 0;
let smsTimer = null;

// 显示登录模态框
function showLogin() {
    closeAllModals();
    document.getElementById('login-modal').style.display = 'flex';
    refreshCaptcha();
    initAuthForms();
    
    // 清空表单和错误信息
    document.getElementById('login-form').reset();
    clearFormErrors();
    hideModalError();
    
    // 设置为登录模式
    setAuthMode('login');
}

// 显示注册模态框（实际上是同一个模态框，但显示不同提示）
function showRegister() {
    closeAllModals();
    document.getElementById('login-modal').style.display = 'flex';
    refreshCaptcha();
    initAuthForms();
    
    // 清空表单和错误信息
    document.getElementById('login-form').reset();
    clearFormErrors();
    hideModalError();
    
    // 设置为注册模式
    setAuthMode('register');
}

// 设置认证模式
function setAuthMode(mode) {
    // 清除之前的错误信息
    hideModalError();
    
    const modalTitle = document.querySelector('#login-modal .modal-header h2');
    const submitButton = document.querySelector('#login-modal .btn-primary');
    const passwordField = document.getElementById('login-password');
    const passwordGroup = passwordField.parentNode;
    const passwordLabel = passwordGroup.querySelector('label');
    const captchaGroup = document.querySelector('.captcha-group');
    const smsGroup = document.querySelector('.sms-group');
    const userTypeSelection = document.querySelector('.user-type-selection');
    const agreementGroup = document.querySelector('.agreement');
    const captchaField = document.getElementById('login-captcha');
    const smsField = document.getElementById('login-sms');
    const agreeTermsField = document.getElementById('agree-terms');
    
    if (mode === 'register') {
        modalTitle.textContent = '用户注册';
        submitButton.textContent = '立即注册';
        passwordGroup.style.display = 'none'; // 注册时隐藏密码字段
        passwordField.required = false;
        captchaGroup.style.display = 'block';
        smsGroup.style.display = 'block';
        userTypeSelection.style.display = 'block';
        agreementGroup.style.display = 'block';
        // 注册时设置必填项
        captchaField.required = true;
        smsField.required = true;
        agreeTermsField.required = true;
    } else {
        modalTitle.textContent = '用户登录';
        submitButton.textContent = '登录';
        passwordGroup.style.display = 'block'; // 登录时显示密码字段
        passwordLabel.textContent = '密码';
        passwordField.placeholder = '请输入密码';
        passwordField.required = true;
        captchaGroup.style.display = 'none'; // 登录时隐藏验证码
        smsGroup.style.display = 'none'; // 登录时隐藏短信验证码
        userTypeSelection.style.display = 'block'; // 保留用户类型选择
        agreementGroup.style.display = 'none'; // 登录时隐藏协议
        // 登录时移除隐藏字段的必填属性
        captchaField.required = false;
        smsField.required = false;
        agreeTermsField.required = false;
    }
}

// 显示忘记密码模态框
function showForgotPassword() {
    closeModal('login-modal');
    document.getElementById('forgot-password-modal').style.display = 'flex';
    refreshResetCaptcha();
}

// 显示用户协议
function showUserAgreement(event) {
    if (event) {
        event.preventDefault(); // 阻止默认链接行为
    }
    
    const agreementContent = `
        <h3>用户协议</h3>
        <div style="max-height: 400px; overflow-y: auto; padding: 10px 0; line-height: 1.6;">
            <p><strong>1. 服务条款</strong></p>
            <p>欢迎使用招投标平台。本协议是您与本平台之间的法律协议。通过使用本平台服务，您同意遵守本协议的所有条款。</p>
            
            <p><strong>2. 用户责任</strong></p>
            <p>• 提供真实、准确、完整的个人信息和资质材料</p>
            <p>• 遵守平台规则和相关法律法规</p>
            <p>• 不得发布虚假、误导性或欺诈性信息</p>
            <p>• 维护良好的商业信誉和职业道德</p>
            
            <p><strong>3. 平台责任</strong></p>
            <p>• 保护用户隐私和数据安全</p>
            <p>• 提供公平、透明、公正的交易环境</p>
            <p>• 维护平台正常运行和服务质量</p>
            <p>• 处理争议和投诉</p>
            
            <p><strong>4. 禁止行为</strong></p>
            <p>• 恶意竞价或围标</p>
            <p>• 发布虚假项目信息</p>
            <p>• 泄露他人商业秘密</p>
            <p>• 其他违法违规行为</p>
            
            <p><strong>5. 其他条款</strong></p>
            <p>本协议的解释权归平台所有。平台有权根据法律法规和业务发展需要修改本协议，修改后的协议将在平台公布。</p>
        </div>
    `;
    
    showContentModal('用户协议', agreementContent);
}

// 显示隐私政策
function showPrivacyPolicy(event) {
    if (event) {
        event.preventDefault(); // 阻止默认链接行为
    }
    
    const policyContent = `
        <h3>隐私政策</h3>
        <div style="max-height: 400px; overflow-y: auto; padding: 10px 0; line-height: 1.6;">
            <p><strong>1. 信息收集</strong></p>
            <p>我们收集您主动提供的信息，包括但不限于：</p>
            <p>• 基本信息：手机号、用户名、密码</p>
            <p>• 身份信息：实名认证资料、企业资质</p>
            <p>• 业务信息：专业技能、项目经验、投标记录</p>
            <p>• 技术信息：设备信息、IP地址、浏览记录</p>
            
            <p><strong>2. 信息使用</strong></p>
            <p>• 提供招投标平台服务</p>
            <p>• 进行身份验证和账户安全保护</p>
            <p>• 改善用户体验和服务质量</p>
            <p>• 发送重要通知和服务信息</p>
            <p>• 进行数据分析和业务优化</p>
            
            <p><strong>3. 信息保护</strong></p>
            <p>• 采用行业标准的加密技术保护数据传输</p>
            <p>• 严格限制员工访问您的个人信息</p>
            <p>• 定期更新和完善安全防护措施</p>
            <p>• 建立数据备份和恢复机制</p>
            
            <p><strong>4. 信息共享</strong></p>
            <p>除以下情况外，我们不会向第三方分享您的个人信息：</p>
            <p>• 获得您的明确同意</p>
            <p>• 法律法规要求或司法机关要求</p>
            <p>• 保护平台和用户的合法权益</p>
            <p>• 经过匿名化处理的统计数据</p>
            
            <p><strong>5. 用户权利</strong></p>
            <p>• 查询和更正个人信息</p>
            <p>• 删除或注销账户</p>
            <p>• 撤回授权同意</p>
            <p>• 投诉和举报</p>
        </div>
    `;
    
    showContentModal('隐私政策', policyContent);
}

// 显示内容模态框
function showContentModal(title, content) {
    document.getElementById('content-modal-title').textContent = title;
    document.getElementById('content-modal-body').innerHTML = content;
    document.getElementById('content-modal').style.display = 'flex';
}

// 关闭内容模态框
function closeContentModal() {
    document.getElementById('content-modal').style.display = 'none';
}

// 关闭所有模态框
function closeAllModals() {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.style.display = 'none';
    });
}

// 初始化认证表单事件
// 初始化认证表单事件
function initAuthForms() {
    // 绑定新的登录表单事件
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.onsubmit = handleNewLogin;
    }
    
    // 绑定设置密码表单事件
    const setPasswordForm = document.getElementById('set-password-form');
    if (setPasswordForm) {
        setPasswordForm.onsubmit = handleSetPassword;
    }
    
    // 绑定设置技能表单事件
    const setSkillsForm = document.getElementById('set-skills-form');
    if (setSkillsForm) {
        setSkillsForm.onsubmit = handleSetSkills;
    }
    
    // 绑定忘记密码表单事件
    const forgotPasswordForm = document.getElementById('forgot-password-form');
    if (forgotPasswordForm) {
        forgotPasswordForm.onsubmit = handleForgotPassword;
    }
}

// 生成图形验证码
function generateCaptcha() {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let result = '';
    for (let i = 0; i < 4; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
}

// 绘制验证码
function drawCaptcha(canvasId, text) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn(`Canvas ${canvasId} not found`);
        return;
    }
    const ctx = canvas.getContext('2d');
    
    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 设置背景
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 添加干扰线
    for (let i = 0; i < 5; i++) {
        ctx.strokeStyle = `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255})`;
        ctx.beginPath();
        ctx.moveTo(Math.random() * canvas.width, Math.random() * canvas.height);
        ctx.lineTo(Math.random() * canvas.width, Math.random() * canvas.height);
        ctx.stroke();
    }
    
    // 绘制文字
    ctx.font = '20px Arial';
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    for (let i = 0; i < text.length; i++) {
        const x = (canvas.width / text.length) * i + (canvas.width / text.length) / 2;
        const y = canvas.height / 2 + (Math.random() - 0.5) * 10;
        const angle = (Math.random() - 0.5) * 0.3;
        
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        ctx.fillText(text[i], 0, 0);
        ctx.restore();
    }
    
    // 添加干扰点
    for (let i = 0; i < 50; i++) {
        ctx.fillStyle = `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255})`;
        ctx.fillRect(Math.random() * canvas.width, Math.random() * canvas.height, 2, 2);
    }
}

// 刷新验证码
function refreshCaptcha() {
    captchaText = generateCaptcha();
    drawCaptcha('captcha-canvas', captchaText);
}

// 刷新重置密码验证码
function refreshResetCaptcha() {
    captchaText = generateCaptcha();
    drawCaptcha('reset-captcha-canvas', captchaText);
}

// 发送短信验证码
async function sendSMS() {
    const phone = document.getElementById('login-phone').value;
    const captcha = document.getElementById('login-captcha').value;
    
    if (!phone) {
        showAlert('请输入手机号', 'error');
        return;
    }
    
    if (!captcha) {
        showAlert('请输入图形验证码', 'error');
        return;
    }
    
    if (captcha.toUpperCase() !== captchaText) {
        showAlert('图形验证码错误', 'error');
        refreshCaptcha();
        return;
    }
    
    if (smsCountdown > 0) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/auth/send-sms`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ phone })
        });
        
        if (response.ok) {
            showAlert('验证码已发送', 'success');
            startSMSCountdown('send-sms-btn');
        } else {
            const error = await response.json();
            showAlert(error.message || '发送失败', 'error');
        }
    } catch (error) {
        console.error('发送短信错误:', error);
        showAlert('发送失败', 'error');
    }
}

// 发送重置密码短信验证码
async function sendResetSMS() {
    const phone = document.getElementById('reset-phone').value;
    const captcha = document.getElementById('reset-captcha').value;
    
    if (!phone) {
        showAlert('请输入手机号', 'error');
        return;
    }
    
    if (!captcha) {
        showAlert('请输入图形验证码', 'error');
        return;
    }
    
    if (captcha.toUpperCase() !== captchaText) {
        showAlert('图形验证码错误', 'error');
        refreshResetCaptcha();
        return;
    }
    
    if (smsCountdown > 0) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/auth/send-reset-sms`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ phone })
        });
        
        if (response.ok) {
            showAlert('验证码已发送', 'success');
            startSMSCountdown('reset-send-sms-btn');
        } else {
            const error = await response.json();
            showAlert(error.message || '发送失败', 'error');
        }
    } catch (error) {
        console.error('发送短信错误:', error);
        showAlert('发送失败', 'error');
    }
}

// 短信倒计时
function startSMSCountdown(buttonId) {
    const button = document.getElementById(buttonId);
    smsCountdown = 60;
    button.disabled = true;
    
    smsTimer = setInterval(() => {
        button.textContent = `${smsCountdown}秒后重试`;
        smsCountdown--;
        
        if (smsCountdown < 0) {
            clearInterval(smsTimer);
            button.disabled = false;
            button.textContent = '获取验证码';
        }
    }, 1000);
}

// 处理新登录/注册
async function handleNewLogin(event) {
    event.preventDefault();
    
    const phone = document.getElementById('login-phone').value;
    const password = document.getElementById('login-password').value;
    const captcha = document.getElementById('login-captcha').value;
    const smsCode = document.getElementById('login-sms').value;
    const userType = document.querySelector('input[name="userType"]:checked')?.value;
    const agreeTerms = document.getElementById('agree-terms').checked;
    
    // 判断当前模式
    const modalTitle = document.querySelector('#login-modal .modal-header h2').textContent;
    const isLoginMode = modalTitle === '用户登录';
    
    // 清除之前的错误样式
    clearFormErrors();
    
    if (isLoginMode) {
        // 简化的登录模式
        return await handleSimpleLogin(phone, password, userType);
    } else {
        // 注册模式，保持原有逻辑
        return await handleRegistration(phone, password, captcha, smsCode, userType, agreeTerms);
    }
}

// 处理简化登录
async function handleSimpleLogin(phone, password, userType) {
    // 验证表单
    let hasError = false;
    
    if (!phone) {
        showFieldError('login-phone', '请输入手机号');
        hasError = true;
    }
    
    if (!password) {
        showFieldError('login-password', '请输入密码');
        hasError = true;
    }
    
    if (!userType) {
        showModalError('请选择用户类型');
        hasError = true;
    }
    
    if (hasError) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/auth/simple-login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                phone,
                password,
                userType
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('token', data.token);
            currentUser = data.user;
            
            closeModal('login-modal');
            updateUserInterface();
            showAlert('登录成功！', 'success');
            
            // 刷新项目列表
            loadProjects(1);
        } else {
            // 检查是否是用户类型不匹配错误
            if (data.message === '用户类型不匹配') {
                // 获取用户在系统中的实际类型
                getUserActualType(phone).then(actualUserType => {
                    if (actualUserType) {
                        const typeText = actualUserType === 'publisher' ? '发布方' : '投标方';
                        const selectedTypeText = userType === 'publisher' ? '发布方' : '投标方';
                        showModalError(`您注册的身份是${typeText}，但选择的是${selectedTypeText}。请选择正确的用户类型后重新登录。`);
                    } else {
                        showModalError('用户类型不匹配，请检查您的身份选择');
                    }
                }).catch(error => {
                    console.error('获取用户类型失败:', error);
                    showModalError('用户类型不匹配，请检查您的身份选择');
                });
            } else {
                // 显示其他错误信息
                showModalError(data.message || '登录失败');
            }
            console.log('登录失败详情:', data);
        }
    } catch (error) {
        console.error('登录错误:', error);
        showAlert('登录失败，请检查网络连接', 'error');
    }
}

// 在模态框内显示错误信息
function showModalError(message, type = 'error') {
    const errorDiv = document.getElementById('modal-error-message');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.className = `modal-error-message ${type}`;
        errorDiv.style.display = 'block';
        
        // 5秒后自动隐藏
        setTimeout(() => {
            hideModalError();
        }, 5000);
    }
}

// 隐藏模态框错误信息
function hideModalError() {
    const errorDiv = document.getElementById('modal-error-message');
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
}

// 获取用户的实际类型
async function getUserActualType(phone) {
    try {
        const response = await fetch(`${API_BASE}/auth/get-user-type`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ phone })
        });
        
        if (response.ok) {
            const data = await response.json();
            return data.userType;
        }
    } catch (error) {
        console.error('获取用户类型错误:', error);
    }
    return null;
}

// 处理注册
async function handleRegistration(phone, password, captcha, smsCode, userType, agreeTerms) {
    // 验证表单
    let hasError = false;
    
    if (!phone) {
        showFieldError('login-phone', '请输入手机号');
        hasError = true;
    }
    
    if (!captcha) {
        showFieldError('login-captcha', '请输入图形验证码');
        hasError = true;
    }
    
    if (!smsCode) {
        showFieldError('login-sms', '请输入短信验证码');
        hasError = true;
    }
    
    if (!userType) {
        showAlert('请选择用户类型', 'error');
        hasError = true;
    }
    
    if (!agreeTerms) {
        showFieldError('agree-terms', '请阅读并同意用户协议');
        showAlert('请阅读并同意用户协议', 'error');
        hasError = true;
    }
    
    if (hasError) {
        return;
    }
    
    if (captcha.toUpperCase() !== captchaText) {
        showFieldError('login-captcha', '图形验证码错误');
        showAlert('图形验证码错误', 'error');
        refreshCaptcha();
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/auth/login-or-register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                phone,
                password,
                smsCode,
                userType
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            localStorage.setItem('token', data.token);
            currentUser = data.user;
            
            closeModal('login-modal');
            
            // 检查是否是首次登录
            if (data.user.first_login) {
                if (!data.user.has_password) {
                    // 需要设置密码
                    document.getElementById('set-password-modal').style.display = 'flex';
                } else if (!data.user.profession) {
                    // 需要设置职业特长
                    document.getElementById('set-skills-modal').style.display = 'flex';
                } else {
                    // 完成设置
                    updateUserInterface();
                    showAlert('登录成功！', 'success');
                    // 刷新项目列表
                    loadProjects(1);
                }
            } else {
                updateUserInterface();
                showAlert(data.isNewUser ? '注册成功！' : '登录成功！', 'success');
                // 刷新项目列表
                loadProjects(1);
            }
        } else {
            showAlert(data.message || '登录失败', 'error');
        }
    } catch (error) {
        console.error('登录错误:', error);
        showAlert('登录失败', 'error');
    }
}

// 处理设置密码
async function handleSetPassword(event) {
    event.preventDefault();
    
    const password1 = document.getElementById('new-password1').value;
    const password2 = document.getElementById('new-password2').value;
    
    if (password1 !== password2) {
        showAlert('两次输入的密码不一致', 'error');
        return;
    }
    
    if (password1.length < 6) {
        showAlert('密码长度至少6位', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/auth/set-password`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({ password: password1 })
        });
        
        if (response.ok) {
            currentUser.has_password = true;
            closeModal('set-password-modal');
            
            if (!currentUser.profession) {
                // 继续设置职业特长
                document.getElementById('set-skills-modal').style.display = 'flex';
            } else {
                // 完成设置
                updateUserInterface();
                showAlert('密码设置成功！', 'success');
            }
        } else {
            const error = await response.json();
            showAlert(error.message || '设置失败', 'error');
        }
    } catch (error) {
        console.error('设置密码错误:', error);
        showAlert('设置失败', 'error');
    }
}

// 处理设置职业特长
async function handleSetSkills(event) {
    event.preventDefault();
    
    const profession = document.getElementById('profession').value;
    const skills = document.getElementById('skills').value;
    const experienceYears = document.getElementById('experience-years').value;
    
    if (!profession || !skills || !experienceYears) {
        showAlert('请填写所有字段', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/auth/set-skills`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                profession,
                skills,
                experienceYears
            })
        });
        
        if (response.ok) {
            currentUser.profession = profession;
            currentUser.specialties = skills;
            currentUser.experience_years = experienceYears;
            currentUser.first_login = false;
            
            closeModal('set-skills-modal');
            updateUserInterface();
            showAlert('设置完成！欢迎使用招投标平台', 'success');
            
            // 刷新项目列表
            loadProjects(1);
        } else {
            const error = await response.json();
            showAlert(error.message || '设置失败', 'error');
        }
    } catch (error) {
        console.error('设置技能错误:', error);
        showAlert('设置失败', 'error');
    }
}

// 处理忘记密码
async function handleForgotPassword(event) {
    event.preventDefault();
    
    const phone = document.getElementById('reset-phone').value;
    const captcha = document.getElementById('reset-captcha').value;
    const smsCode = document.getElementById('reset-sms').value;
    const password1 = document.getElementById('reset-new-password1').value;
    const password2 = document.getElementById('reset-new-password2').value;
    
    if (!phone || !captcha || !smsCode || !password1 || !password2) {
        showAlert('请填写所有字段', 'error');
        return;
    }
    
    if (captcha.toUpperCase() !== captchaText) {
        showAlert('图形验证码错误', 'error');
        refreshResetCaptcha();
        return;
    }
    
    if (password1 !== password2) {
        showAlert('两次输入的密码不一致', 'error');
        return;
    }
    
    if (password1.length < 6) {
        showAlert('密码长度至少6位', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/auth/reset-password`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                phone,
                smsCode,
                newPassword: password1
            })
        });
        
        if (response.ok) {
            closeModal('forgot-password-modal');
            showAlert('密码重置成功，请重新登录', 'success');
            showLogin();
        } else {
            const error = await response.json();
            showAlert(error.message || '重置失败', 'error');
        }
    } catch (error) {
        console.error('重置密码错误:', error);
        showAlert('重置失败', 'error');
    }
}

// 表单验证辅助函数
function showFieldError(fieldId, message) {
    const field = document.getElementById(fieldId);
    if (field) {
        field.classList.add('error');
        
        // 移除之前的错误消息
        const existingError = field.parentNode.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
        
        // 添加新的错误消息
        const errorSpan = document.createElement('span');
        errorSpan.className = 'error-message';
        errorSpan.textContent = message;
        field.parentNode.appendChild(errorSpan);
        
        // 3秒后自动清除错误样式
        setTimeout(() => {
            clearFieldError(fieldId);
        }, 3000);
    }
}

function clearFieldError(fieldId) {
    const field = document.getElementById(fieldId);
    if (field) {
        field.classList.remove('error');
        const errorMessage = field.parentNode.querySelector('.error-message');
        if (errorMessage) {
            errorMessage.remove();
        }
    }
}

function clearFormErrors() {
    const errorFields = document.querySelectorAll('.error');
    errorFields.forEach(field => {
        field.classList.remove('error');
    });
    
    const errorMessages = document.querySelectorAll('.error-message');
    errorMessages.forEach(message => {
        message.remove();
    });
}

// 刷新当前视图
async function refreshCurrentView() {
    const activeTab = document.querySelector('.tab-content.active');
    if (!activeTab) return;
    
    switch (activeTab.id) {
        case 'my-projects':
            if (currentUser && currentUser.userType === 'publisher') {
                await loadUserProjects();
            }
            break;
        case 'my-bids':
            if (currentUser && currentUser.userType === 'bidder') {
                await loadUserBids();
            }
            break;
        case 'all-projects':
            await loadProjects();
            break;
        default:
            break;
    }
}

// 显示登录表单
function showLoginForm() {
    // 清空当前用户信息
    currentUser = null;
    updateUserInterface();
    
    // 显示登录标签页
    const loginTab = document.querySelector('[data-tab="login"]');
    const registerTab = document.querySelector('[data-tab="register"]');
    const allProjectsTab = document.querySelector('[data-tab="all-projects"]');
    
    if (loginTab) {
        loginTab.click();
    } else if (allProjectsTab) {
        allProjectsTab.click();
    }
    
    // 清空敏感内容
    const myProjectsContent = document.getElementById('my-projects');
    const myBidsContent = document.getElementById('my-bids');
    
    if (myProjectsContent) {
        myProjectsContent.innerHTML = '<div class="no-data">请先登录</div>';
    }
    if (myBidsContent) {
        myBidsContent.innerHTML = '<div class="no-data">请先登录</div>';
    }
}
