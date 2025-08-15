@echo off
echo ========================================
echo    招投标平台 - 开发环境启动脚本
echo ========================================
echo.

REM 检查Node.js是否安装
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Node.js，请先安装Node.js
    echo 下载地址: https://nodejs.org/
    pause
    exit /b 1
)

REM 检查npm是否可用
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: npm不可用
    pause
    exit /b 1
)

REM 检查是否安装了依赖
if not exist "node_modules" (
    echo 正在安装依赖包...
    call npm install
    if %errorlevel% neq 0 (
        echo 错误: 依赖安装失败
        pause
        exit /b 1
    )
    echo 依赖安装完成！
    echo.
)

REM 检查环境变量文件
if not exist ".env" (
    echo 正在创建环境变量文件...
    copy ".env.example" ".env"
    echo.
    echo 请编辑 .env 文件配置数据库连接信息！
    echo 默认配置:
    echo   - 端口: 3000
    echo   - 数据库: bidding_platform
    echo   - 用户: root
    echo   - 密码: (空)
    echo.
    pause
)

REM 检查上传目录
if not exist "uploads" (
    mkdir uploads
    echo 创建上传目录: uploads
)

REM 检查日志目录
if not exist "logs" (
    mkdir logs
    echo 创建日志目录: logs
)

echo 启动开发服务器...
echo.
echo 服务器地址: http://localhost:3000
echo 按 Ctrl+C 停止服务器
echo.

REM 启动服务器
if exist "node_modules\.bin\nodemon.cmd" (
    call npm run dev
) else (
    call npm start
)

echo.
echo 服务器已停止
pause
