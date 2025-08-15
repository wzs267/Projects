@echo off
echo ========================================
echo    招投标平台 - 数据库初始化脚本
echo ========================================
echo.

REM 检查Node.js是否安装
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Node.js，请先安装Node.js
    pause
    exit /b 1
)

REM 检查环境变量文件
if not exist ".env" (
    echo 错误: 未找到 .env 文件，请先复制 .env.example 并配置数据库连接
    pause
    exit /b 1
)

echo 正在初始化数据库...
echo.
echo 请确保：
echo 1. MySQL服务已启动
echo 2. .env 文件中的数据库配置正确
echo 3. 数据库用户具有创建数据库的权限
echo.

pause

REM 执行数据库初始化
node scripts/init-db.js

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo    数据库初始化完成！
    echo ========================================
    echo.
    echo 默认管理员账户:
    echo   邮箱: admin@example.com
    echo   密码: admin123456
    echo.
    echo 请登录后立即修改管理员密码！
    echo.
) else (
    echo.
    echo 数据库初始化失败，请检查：
    echo 1. MySQL服务是否正常运行
    echo 2. .env 文件中的数据库配置是否正确
    echo 3. 数据库用户权限是否足够
    echo.
)

pause
