# 贡献指南

感谢您对泛癌数据分析平台的关注！我们欢迎任何形式的贡献。

## 🤝 如何贡献

### 报告Bug

如果您发现了Bug，请通过GitHub Issues报告，并包含以下信息：

1. **Bug描述**：清晰简洁地描述问题
2. **重现步骤**：详细的重现步骤
3. **期望行为**：您期望发生什么
4. **实际行为**：实际发生了什么
5. **环境信息**：
   - R版本
   - 操作系统
   - 相关R包版本
6. **截图**：如果适用，添加截图
7. **数据样例**：如果可能，提供能重现问题的最小数据集

### 提出新功能

如果您有新功能的想法：

1. 首先检查是否已有相关Issue
2. 创建新的Feature Request Issue
3. 清晰描述功能和使用场景
4. 说明为什么这个功能对用户有价值

### 提交代码

#### 开发流程

1. **Fork项目**
   ```bash
   # 点击GitHub页面右上角的Fork按钮
   ```

2. **克隆仓库**
   ```bash
   git clone https://github.com/your-username/cancer-analysis-platform.git
   cd cancer-analysis-platform
   ```

3. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

4. **进行开发**
   - 编写代码
   - 添加测试
   - 更新文档

5. **提交更改**
   ```bash
   git add .
   git commit -m "描述您的更改"
   ```

6. **推送到GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建Pull Request**
   - 访问您fork的仓库
   - 点击"New Pull Request"
   - 填写PR描述
   - 等待审核

#### 代码规范

**R代码规范**

- 遵循[tidyverse代码风格](https://style.tidyverse.org/)
- 使用有意义的变量名和函数名
- 添加必要的注释
- 保持函数简洁，单一职责

```r
# 好的示例
calculate_accuracy <- function(predictions, actual) {
  # 计算分类准确率
  correct <- sum(predictions == actual)
  total <- length(actual)
  accuracy <- correct / total
  return(accuracy)
}

# 避免
calc_acc <- function(p, a) {
  return(sum(p == a) / length(a))
}
```

**代码注释**

- 函数开头添加说明注释
- 复杂逻辑添加行内注释
- 使用中文或英文注释（保持一致）

```r
#' 训练随机森林模型
#' 
#' @param X_train 训练特征矩阵
#' @param y_train 训练标签向量
#' @param params 模型参数列表
#' @return 训练好的模型对象
train_random_forest <- function(X_train, y_train, params) {
  # 实现代码
}
```

**Git提交信息**

使用清晰的提交信息：

```
类型: 简短描述（不超过50个字符）

详细描述（如有必要）
- 要点1
- 要点2

相关Issue: #123
```

提交类型：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：
```
feat: 添加XGBoost算法支持

- 实现XGBoost训练功能
- 添加超参数优化
- 更新UI界面

相关Issue: #45
```

### 测试

在提交PR之前，请确保：

1. **功能测试**
   ```r
   # 运行应用测试
   source("tests/test_module.R")
   ```

2. **手动测试**
   - 启动应用
   - 测试新功能
   - 验证不影响现有功能

3. **数据测试**
   - 使用不同大小的数据集
   - 测试边界情况
   - 验证错误处理

### 文档

更新相关文档：

1. **代码文档**：函数注释
2. **README**：新功能说明
3. **CHANGELOG**：记录更改
4. **使用指南**：用户文档

## 📋 开发环境设置

### 必需软件

- R >= 4.0
- RStudio (推荐)
- Git

### 安装依赖

```r
# 安装所有依赖包
source("scripts/install_r_environment.sh")
```

### 运行应用

```r
shiny::runApp('src/cancer_analysis_app.R')
```

## 🎯 优先级领域

我们特别欢迎以下方面的贡献：

1. **算法增强**
   - 新的机器学习算法
   - 深度学习模型集成
   - 算法性能优化

2. **可解释性**
   - 新的可解释性方法
   - 可视化改进
   - 交互式解释

3. **数据支持**
   - 更多数据格式支持
   - 数据预处理功能
   - 数据质量检测

4. **用户体验**
   - UI/UX改进
   - 性能优化
   - 错误处理

5. **文档**
   - 教程和示例
   - API文档
   - 视频教程

## ❓ 问题？

如有疑问，可以通过以下方式联系：

- 创建GitHub Issue
- 发送邮件至：[your-email@example.com]
- 查看[文档](docs/)

## 📜 行为准则

### 我们的承诺

为建立一个开放友好的环境，我们承诺：

- 尊重不同观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表现同理心

### 不可接受的行为

- 使用性化的语言或图像
- 人身攻击或侮辱
- 公开或私下骚扰
- 未经许可发布他人信息
- 其他不专业或不受欢迎的行为

### 执行

不可接受的行为可以通过[your-email@example.com]报告，所有投诉都会被审查和调查。

## 🙏 致谢

感谢所有贡献者！

您的贡献让这个项目变得更好。

---

再次感谢您的贡献！🎉

