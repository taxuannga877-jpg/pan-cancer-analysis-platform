# 🧬 泛癌数据分析平台

[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)
[![Shiny](https://img.shields.io/badge/Shiny-1.7+-brightgreen.svg)](https://shiny.rstudio.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 项目简介

泛癌数据分析平台是一个基于R Shiny的智能诊断系统，专为癌症数据分析和机器学习建模设计。该平台集成了多种先进的机器学习算法、智能超参数优化、模型可解释性分析等功能，支持多种癌症类型的数据分析。

### 🎯 核心创新点

1. **跨轮次学习的智能贝叶斯优化系统**
   - 自动记忆历史最佳参数
   - 逐轮优化提升模型性能
   - 支持多种算法并行优化

2. **动态目标列识别与全流程自适应**
   - 支持任意命名的目标列
   - 全流程自动适配
   - 即插即用，无需数据预处理

3. **实时过拟合检测与多维度可解释性分析**
   - 智能过拟合监测与预警
   - SHAP多维度解释
   - 完整的数据质量检测

### ✨ 主要功能

- 📊 **数据概览**: 自动统计分析、数据质量检查
- 🔍 **特征分析**: 相关性热图、特征分布可视化
- 🤖 **机器学习**: **12种算法支持** (v2.1新增4种)
  - 线性模型: Logistic, Lasso, Ridge, Elastic Net
  - 集成学习: Random Forest, GBM, XGBoost
  - 支持向量机: SVM-Radial, SVM-Linear ⭐
  - 其他: Neural Network ⭐, Naive Bayes ⭐, KNN ⭐
- 📈 **模型评估**: ROC曲线、混淆矩阵、交叉验证、过拟合检测
- 💡 **可解释性**: SHAP值分析、特征重要性、依赖图
- 🔮 **预测分析**: 新样本预测、批量预测、动态模型选择
- 💾 **模型管理**: 模型保存/加载、历史记录、性能对比

## 🚀 快速开始

### 环境要求

- R >= 4.0
- RStudio (推荐)
- 8GB+ RAM
- Linux/macOS/Windows

### 安装步骤

#### 1. 自动安装（推荐）

```bash
cd 泛癌分析平台
bash scripts/install_r_environment.sh
```

#### 2. 手动安装

```r
# 安装必需的R包
install.packages(c(
  "shiny", "shinydashboard", "shinyWidgets",
  "DT", "ggplot2", "plotly", "tidyverse",
  "caret", "pROC", "randomForest", "glmnet",
  "gbm", "e1071", "xgboost",
  "rBayesianOptimization", "shapr", "DALEX"
))
```

### 启动应用

```r
# 方法1: 在R控制台中运行
setwd("泛癌分析平台")
shiny::runApp('src/cancer_analysis_app.R', host='0.0.0.0', port=3838)

# 方法2: 使用命令行
cd 泛癌分析平台
R -e "shiny::runApp('src/cancer_analysis_app.R', host='0.0.0.0', port=3838)"
```

访问 `http://localhost:3838` 即可使用平台。

## 📖 使用指南

### 数据格式要求

- **文件格式**: CSV
- **数据结构**: 
  - 每行代表一个样本
  - 每列代表一个特征或目标变量
  - 目标列可以是任意列名（系统会自动识别）
  - 目标列应为二分类（0和1）

示例数据格式：

```csv
Feature1,Feature2,Feature3,...,Target
1.23,4.56,7.89,...,0
2.34,5.67,8.90,...,1
...
```

### 基本工作流程

1. **数据加载**: 上传CSV文件或加载示例数据
2. **选择目标列**: 系统自动识别或手动选择
3. **数据探索**: 查看统计信息、相关性分析
4. **模型训练**: 选择算法、配置参数、启动训练
5. **模型评估**: 查看性能指标、ROC曲线
6. **可解释性**: 分析SHAP值、特征重要性
7. **预测应用**: 对新样本进行预测
8. **模型保存**: 保存训练好的模型

### 支持的机器学习算法 (12种)

| # | 算法 | 类型 | 特点 |
|---|------|------|------|
| 1 | Logistic Regression | 线性模型 | 简单、可解释性强 |
| 2 | Lasso | 正则化 | 特征选择 |
| 3 | Ridge | 正则化 | 防止过拟合 |
| 4 | Elastic Net | 正则化 | 综合Lasso和Ridge |
| 5 | Random Forest | 集成学习 | 高准确率、鲁棒性强 |
| 6 | GBM | 梯度提升 | 强大的预测能力 |
| 7 | XGBoost | 梯度提升 | 高效、准确 |
| 8 | SVM-Radial | 核方法 | 非线性分类 |
| 9 | **SVM-Linear** ⭐ | 线性SVM | 快速、稳定 |
| 10 | **Neural Network** ⭐ | 深度学习 | 复杂模式识别 |
| 11 | **Naive Bayes** ⭐ | 概率模型 | 快速、简单 |
| 12 | **K-Nearest Neighbors** ⭐ | 非参数 | 直观易用 |

## 📁 项目结构

```
泛癌分析平台/
├── README.md                 # 项目说明文档
├── LICENSE                   # 开源许可证
├── .gitignore               # Git忽略文件
├── src/                     # 源代码目录
│   └── cancer_analysis_app.R    # 主应用程序
├── data/                    # 数据目录
│   └── example_liver_cancer.csv # 示例数据
├── models/                  # 模型保存目录
│   └── .gitkeep
├── docs/                    # 文档目录
│   ├── 完整使用手册.md
│   ├── 使用指南.md
│   ├── 环境安装指南.md
│   ├── 数据质量检测说明.md
│   └── 项目总结.md
├── scripts/                 # 脚本目录
│   └── install_r_environment.sh
└── tests/                   # 测试目录
    └── test_module.R
```

## 🔧 高级功能

### K折交叉验证

- 支持3-10折交叉验证
- 自动计算平均性能指标
- 实时过拟合检测

### 贝叶斯超参数优化

- 自动搜索最优参数组合
- 跨训练轮次学习优化
- 支持多种算法并行优化

### SHAP可解释性分析

- 全局特征重要性
- 特征依赖图
- 部分依赖图（PDP）
- 个体样本解释

### 数据质量检测

- 测试集大小检查
- 类别平衡性检查
- 数据泄漏风险提示
- 完美分类警告

## 📊 应用场景

- 🏥 临床诊断辅助
- 🔬 生物标志物发现
- 📉 预后分析
- 🧪 药物响应预测
- 📈 风险评估

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 更新日志

### v2.1.0 (2025-10-17) 🎉 重大更新

**新功能**:
- ✨ 扩展至**12种机器学习算法**（新增4种）
- ✨ 详细的训练进度和时间显示
- ✨ 动态模型选择（只显示已训练模型）
- ✨ 改进的过拟合检测和警告系统

**Bug修复**:
- 🐛 修复ROC曲线显示问题，支持12个模型
- 🐛 修复GBM和RandomForest过拟合问题
- 🐛 修复SVM数据处理失败问题
- 🐛 修复预测页面模型选择问题

**改进**:
- 🔧 强化防过拟合机制（更严格的正则化）
- 🔧 优化数据预处理（自动标准化）
- 🔧 改进错误处理和日志输出
- 🔧 K折交叉验证和贝叶斯优化进度可视化

详见: [更新日志_v2.1.md](更新日志_v2.1.md) | [修复说明.md](修复说明.md)

### v2.0.0 (2025-10-17)

- ✨ 实现跨轮次学习的贝叶斯优化
- ✨ 添加动态目标列识别功能
- ✨ 集成SHAP可解释性分析
- ✨ 实现实时过拟合检测
- ✨ 添加智能数据质量检测
- 🐛 修复相关性矩阵计算错误
- 🐛 修复ROC曲线绘制问题
- 📝 完善文档和使用指南

### v1.0.0 (2025-10)

- 🎉 首次发布
- ✨ 实现基础机器学习功能
- ✨ 支持8种分类算法
- ✨ 添加模型评估功能

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👥 作者

泛癌数据分析平台开发团队

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 Email: [taxuannga877@gmail.com]
- 🐛 Issues: [GitHub Issues](https://github.com/taxuannga877-jpg/pan-cancer-analysis-platform/issues)

## 🙏 致谢

感谢以下开源项目和社区：

- [R Project](https://www.r-project.org/)
- [Shiny](https://shiny.rstudio.com/)
- [caret](https://topepo.github.io/caret/)
- [SHAP](https://github.com/slundberg/shap)
- 所有贡献者和使用者

---

⭐ 如果这个项目对您有帮助，请给我们一个Star！
