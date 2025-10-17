# ========================================================================
# 肝癌分析模块测试脚本
# ========================================================================
# 用于快速测试模块是否正常工作
# ========================================================================

cat("========================================\n")
cat("肝癌分析模块测试脚本\n")
cat("========================================\n\n")

# ========================================================================
# 1. 测试数据加载
# ========================================================================

cat("[测试 1/5] 数据加载...\n")
tryCatch({
  data <- read.csv("liver_cancer2.csv", stringsAsFactors = FALSE)
  cat("  ✓ 数据加载成功\n")
  cat("    - 样本数:", nrow(data), "\n")
  cat("    - 特征数:", ncol(data) - 1, "\n")
  cat("    - Target=0:", sum(data$Target == 0), "\n")
  cat("    - Target=1:", sum(data$Target == 1), "\n")
}, error = function(e) {
  cat("  ✗ 数据加载失败:", e$message, "\n")
  stop("无法继续测试")
})

# ========================================================================
# 2. 测试包加载
# ========================================================================

cat("\n[测试 2/5] R 包检查...\n")

required_packages <- c(
  "caret", "pROC", "randomForest", "glmnet", "gbm", "e1071",
  "ggplot2", "tidyverse"
)

missing_packages <- c()

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
    cat("  ✗", pkg, "未安装\n")
  } else {
    cat("  ✓", pkg, "\n")
  }
}

if (length(missing_packages) > 0) {
  cat("\n需要安装以下包:\n")
  cat("install.packages(c('", paste(missing_packages, collapse = "', '"), "'))\n", sep = "")
} else {
  cat("  ✓ 所有必需的包都已安装\n")
}

# ========================================================================
# 3. 测试简单模型训练
# ========================================================================

cat("\n[测试 3/5] 简单模型训练...\n")

tryCatch({
  suppressPackageStartupMessages({
    library(caret)
    library(pROC)
    library(glmnet)
  })
  
  # 数据分割
  set.seed(1234)
  train_index <- createDataPartition(data$Target, p = 0.7, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  cat("  ✓ 数据分割完成\n")
  cat("    - 训练集:", nrow(train_data), "样本\n")
  cat("    - 测试集:", nrow(test_data), "样本\n")
  
  # 训练简单的 Logistic 回归模型
  cat("  → 训练 Logistic Regression...\n")
  fit_logistic <- glm(Target ~ ., data = train_data, family = binomial())
  pred_test <- predict(fit_logistic, test_data, type = "response")
  
  # 计算 AUC
  roc_obj <- roc(test_data$Target, pred_test, quiet = TRUE)
  auc_value <- auc(roc_obj)
  
  cat("  ✓ 模型训练成功\n")
  cat("    - AUC:", round(auc_value, 4), "\n")
  
  if (auc_value > 0.7) {
    cat("    - 性能评估: 良好 ✓\n")
  } else {
    cat("    - 性能评估: 需要改进\n")
  }
  
}, error = function(e) {
  cat("  ✗ 模型训练失败:", e$message, "\n")
})

# ========================================================================
# 4. 测试特征分析
# ========================================================================

cat("\n[测试 4/5] 特征分析...\n")

tryCatch({
  # 计算相关性矩阵
  cor_matrix <- cor(data[, -ncol(data)])
  
  cat("  ✓ 相关性矩阵计算成功\n")
  
  # 找出最相关的特征对
  cor_values <- abs(cor_matrix)
  diag(cor_values) <- 0
  max_cor <- max(cor_values)
  max_pos <- which(cor_values == max_cor, arr.ind = TRUE)[1, ]
  
  cat("    - 最高相关性:", round(max_cor, 4), "\n")
  cat("    - 特征对:", colnames(cor_matrix)[max_pos[1]], "vs", 
      colnames(cor_matrix)[max_pos[2]], "\n")
  
  # 计算与 Target 的相关性
  target_cor <- sapply(data[, -ncol(data)], function(x) {
    cor(x, data$Target)
  })
  
  top_feature <- names(target_cor)[which.max(abs(target_cor))]
  top_cor <- target_cor[top_feature]
  
  cat("  ✓ 与 Target 相关性最高的特征:", top_feature, 
      "(相关系数:", round(top_cor, 4), ")\n")
  
}, error = function(e) {
  cat("  ✗ 特征分析失败:", e$message, "\n")
})

# ========================================================================
# 5. 测试预测功能
# ========================================================================

cat("\n[测试 5/5] 预测功能...\n")

tryCatch({
  # 创建一个新样本（使用训练数据的平均值）
  new_sample <- data.frame(
    TEF_365nm = mean(data$TEF_365nm),
    TEF_405nm = mean(data$TEF_405nm),
    TEF_450nm = mean(data$TEF_450nm),
    LDF_Myo = mean(data$LDF_Myo),
    LDF_Card = mean(data$LDF_Card),
    LDF_Resp = mean(data$LDF_Resp),
    AFP_Level = mean(data$AFP_Level),
    Bilirubin = mean(data$Bilirubin),
    Albumin = mean(data$Albumin),
    Spectral_Skewness = mean(data$Spectral_Skewness),
    Spectral_Kurtosis = mean(data$Spectral_Kurtosis),
    Tissue_Density = mean(data$Tissue_Density)
  )
  
  # 使用之前训练的模型进行预测
  pred_prob <- predict(fit_logistic, new_sample, type = "response")
  pred_class <- ifelse(pred_prob > 0.5, 1, 0)
  pred_label <- ifelse(pred_class == 1, "恶性", "良性")
  
  cat("  ✓ 预测成功\n")
  cat("    - 输入: 所有特征的平均值\n")
  cat("    - 预测概率:", round(pred_prob * 100, 2), "%\n")
  cat("    - 预测类别:", pred_label, "\n")
  
}, error = function(e) {
  cat("  ✗ 预测失败:", e$message, "\n")
})

# ========================================================================
# 测试总结
# ========================================================================

cat("\n========================================\n")
cat("测试完成!\n")
cat("========================================\n\n")

cat("如果所有测试都通过 (✓)，可以继续使用:\n")
cat("1. 运行完整分析: source('liver_cancer_analysis.R')\n")
cat("2. 启动 Shiny 应用: shiny::runApp('liver_cancer_shiny_app.R')\n\n")

cat("如果有测试失败 (✗)，请:\n")
cat("1. 检查错误信息\n")
cat("2. 安装缺失的 R 包\n")
cat("3. 确保数据文件 'liver_cancer2.csv' 在当前目录\n\n")

