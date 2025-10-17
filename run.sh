#!/bin/bash

# ========================================================================
# 泛癌数据分析平台 - 启动脚本
# ========================================================================

echo "=========================================="
echo "  泛癌数据分析平台"
echo "  Pan-Cancer Analysis Platform"
echo "=========================================="
echo ""

# 检查R是否安装
if ! command -v R &> /dev/null; then
    echo "❌ 错误: 未检测到R环境"
    echo "请先运行: bash scripts/install_r_environment.sh"
    exit 1
fi

echo "✅ R环境检测通过"
echo ""

# 获取当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📂 工作目录: $SCRIPT_DIR"
echo ""

# 设置端口（默认3838）
PORT=${1:-3838}

echo "🚀 正在启动应用..."
echo "📍 访问地址: http://localhost:$PORT"
echo "⏹️  停止应用: 按 Ctrl+C"
echo ""
echo "=========================================="
echo ""

# 启动应用
R -e "shiny::runApp('src/cancer_analysis_app.R', host='0.0.0.0', port=$PORT)"

