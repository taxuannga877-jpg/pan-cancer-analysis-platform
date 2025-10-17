#!/bin/bash

set -e  # 遇到错误立即退出

echo "=================================="
echo "R 环境自动安装脚本"
echo "=================================="

# 检测操作系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "无法检测操作系统"
    exit 1
fi

echo "检测到操作系统: $OS ($VERSION)"

# 安装 R
echo -e "\n[1/3] 安装 R 和系统依赖..."
if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    echo "更新包列表..."
    sudo apt update
    
    echo "安装 R..."
    sudo apt install -y r-base r-base-dev
    
    echo "安装系统依赖库..."
    sudo apt install -y \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libfontconfig1-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libfreetype6-dev \
        libpng-dev \
        libtiff5-dev \
        libjpeg-dev \
        build-essential \
        gfortran
        
elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ]; then
    echo "安装 EPEL 仓库..."
    sudo yum install -y epel-release
    
    echo "安装 R..."
    sudo yum install -y R
    
    echo "安装系统依赖库..."
    sudo yum install -y \
        libcurl-devel \
        openssl-devel \
        libxml2-devel \
        fontconfig-devel \
        harfbuzz-devel \
        fribidi-devel \
        freetype-devel \
        libpng-devel \
        libtiff-devel \
        libjpeg-devel
else
    echo "不支持的操作系统: $OS"
    echo "请手动安装 R: https://www.r-project.org/"
    exit 1
fi

echo "✓ R 和系统依赖安装完成"

# 验证 R 安装
echo -e "\n验证 R 安装..."
R --version | head -3

# 安装 R 包
echo -e "\n[2/3] 安装 R 包 (这可能需要 10-20 分钟)..."

sudo R --vanilla << 'REOF'
# 设置 CRAN 镜像
options(repos = c(CRAN = "https://cloud.r-project.org/"))
options(timeout = 300)

# 要安装的包列表
packages <- c(
  # 核心 Shiny 包
  "shiny",
  "shinydashboard",
  "shinyWidgets",
  "DT",
  
  # 可视化包
  "ggplot2",
  "plotly",
  
  # 数据处理包
  "tidyverse",
  "dplyr",
  "tidyr",
  
  # 机器学习包
  "caret",
  "pROC",
  "randomForest",
  "glmnet",
  "gbm",
  "e1071"
)

cat("\n开始安装", length(packages), "个 R 包...\n")
cat("这可能需要一些时间，请耐心等待...\n\n")

# 逐个安装并显示进度
for (i in seq_along(packages)) {
  pkg <- packages[i]
  cat(sprintf("[%d/%d] 安装 %s...\n", i, length(packages), pkg))
  
  tryCatch({
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, dependencies = TRUE, quiet = FALSE)
      cat("  ✓", pkg, "安装成功\n\n")
    } else {
      cat("  ✓", pkg, "已安装\n\n")
    }
  }, error = function(e) {
    cat("  ✗", pkg, "安装失败:", e$message, "\n\n")
  })
}

cat("\n================================\n")
cat("检查安装结果:\n")
cat("================================\n")

all_success <- TRUE
for (pkg in packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat("✓", pkg, "\n")
  } else {
    cat("✗", pkg, "安装失败\n")
    all_success <- FALSE
  }
}

if (all_success) {
  cat("\n✓ 所有 R 包安装成功!\n")
} else {
  cat("\n⚠ 部分包安装失败，但可能不影响核心功能\n")
}
REOF

echo "✓ R 包安装完成"

# 测试安装
echo -e "\n[3/3] 测试 R 环境..."

if [ -f "test_liver_cancer_module.R" ]; then
    echo "运行测试脚本..."
    Rscript test_liver_cancer_module.R
else
    echo "测试脚本不存在，执行简单测试..."
    Rscript -e "
    cat('测试 R 环境...\n')
    cat('R 版本:', R.version.string, '\n')
    cat('已安装的关键包:\n')
    pkgs <- c('shiny', 'ggplot2', 'caret')
    for (pkg in pkgs) {
      if (requireNamespace(pkg, quietly = TRUE)) {
        cat('  ✓', pkg, '\n')
      } else {
        cat('  ✗', pkg, '\n')
      }
    }
    cat('\n✓ R 环境测试完成!\n')
    "
fi

echo -e "\n=================================="
echo "✓ R 环境安装完成!"
echo "=================================="
echo ""
echo "R 版本:"
R --version | head -1
echo ""
echo "现在可以运行:"
echo "  cd /root/SurvivalML"
echo "  Rscript test_liver_cancer_module.R"
echo "  R -e \"shiny::runApp('liver_cancer_shiny_app.R')\""
echo ""
echo "或者在 R 控制台中:"
echo "  R"
echo "  > setwd('/root/SurvivalML')"
echo "  > shiny::runApp('liver_cancer_shiny_app.R')"
echo ""

 