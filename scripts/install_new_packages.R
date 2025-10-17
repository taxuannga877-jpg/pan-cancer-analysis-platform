# ========================================================================
# 安装新增的机器学习包
# ========================================================================

cat("开始安装新的R包...\n\n")

# 需要安装的包列表
required_packages <- c(
  "xgboost",      # XGBoost算法
  "nnet",         # Neural Network
  "naivebayes",   # Naive Bayes
  "kknn"          # K-Nearest Neighbors
)

# 检查并安装包
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("安装", pkg, "...\n")
    install.packages(pkg, dependencies = TRUE, repos = "https://cloud.r-project.org/")
    
    # 验证安装
    if (require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("✓", pkg, "安装成功\n\n")
    } else {
      cat("✗", pkg, "安装失败\n\n")
    }
  } else {
    cat("✓", pkg, "已安装\n")
  }
}

cat("\n========================================\n")
cat("包安装完成！\n")
cat("========================================\n")

