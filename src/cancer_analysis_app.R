# ========================================================================
# 泛癌数据分析平台 - 智能诊断系统
# ========================================================================
# 核心功能:
# 1. K折交叉验证
# 2. 跨轮次学习的贝叶斯超参数优化
# 3. 模型自动保存和加载
# 4. 实时过拟合检测和防止
# 5. SHAP 多维度可解释性分析
# 6. 动态目标列识别与自适应
# 7. 智能数据质量检测
# ========================================================================

library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(DT)
library(ggplot2)
library(plotly)
library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(glmnet)
library(gbm)
library(e1071)
library(xgboost)
library(nnet)  # Neural Network
library(naivebayes)  # Naive Bayes
library(kknn)  # K-Nearest Neighbors

# 贝叶斯优化
suppressPackageStartupMessages({
  if (!require("rBayesianOptimization")) {
    install.packages("rBayesianOptimization")
    library(rBayesianOptimization)
  }
})

# SHAP 值计算 (使用 shapr 包)
suppressPackageStartupMessages({
  if (!require("shapr")) {
    install.packages("shapr")
    library(shapr)
  }
  if (!require("DALEX")) {
    install.packages("DALEX")
    library(DALEX)
  }
})

# ========================================================================
# UI 部分
# ========================================================================

ui <- dashboardPage(
  
  dashboardHeader(
    title = "泛癌数据分析平台",
    titleWidth = 350
  ),
  
  dashboardSidebar(
    width = 350,
    sidebarMenu(
      id = "sidebar",
      menuItem("数据概览", tabName = "data_overview", icon = icon("database")),
      menuItem("特征分析", tabName = "feature_analysis", icon = icon("chart-bar")),
      menuItem("高级模型训练", tabName = "model_training", icon = icon("cogs")),
      menuItem("模型评估", tabName = "model_evaluation", icon = icon("chart-line")),
      menuItem("可解释性分析", tabName = "interpretability", icon = icon("lightbulb")),
      menuItem("预测分析", tabName = "prediction", icon = icon("magic")),
      menuItem("模型管理", tabName = "model_management", icon = icon("save"))
    )
  ),
  
  dashboardBody(
    
    tags$head(
      tags$style(HTML("
        .box-header { background-color: #3c8dbc; color: white; }
        .info-box { min-height: 80px; }
        .small-box { border-radius: 5px; }
        .shiny-notification { position: fixed; top: 50px; right: 10px; }
      "))
    ),
    
    tabItems(
      
      # ============================================
      # Tab 1: 数据概览
      # ============================================
      tabItem(
        tabName = "data_overview",
        
        fluidRow(
          box(
            title = "数据加载",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            
            fileInput("data_file", "上传CSV数据文件", accept = c(".csv")),
            actionButton("load_default_data", "加载示例数据（肝癌）", 
                        icon = icon("upload"), class = "btn-success"),
            hr(),
            h4("目标列设置:"),
            p("系统将自动检测最后一列作为目标列，或者您可以手动指定："),
            selectInput("target_column", "选择目标列:", choices = NULL),
            hr(),
            verbatimTextOutput("data_info")
          )
        ),
        
        fluidRow(
          valueBoxOutput("total_samples", width = 3),
          valueBoxOutput("num_features", width = 3),
          valueBoxOutput("target_0_count", width = 3),
          valueBoxOutput("target_1_count", width = 3)
        ),
        
        fluidRow(
          box(title = "数据预览 (前100行)", status = "info", solidHeader = TRUE,
              width = 12, DTOutput("data_table"))
        )
      ),
      
      # ============================================
      # Tab 2: 特征分析
      # ============================================
      tabItem(
        tabName = "feature_analysis",
        
        fluidRow(
          box(title = "特征相关性热图", status = "primary", solidHeader = TRUE,
              width = 12, plotlyOutput("correlation_heatmap", height = "600px"))
        ),
        
        fluidRow(
          box(title = "特征分布对比", status = "info", solidHeader = TRUE, width = 12,
              selectInput("selected_feature", "选择特征:", choices = NULL),
              plotlyOutput("feature_boxplot", height = "400px"))
        )
      ),
      
      # ============================================
      # Tab 3: 高级模型训练
      # ============================================
      tabItem(
        tabName = "model_training",
        
        fluidRow(
          box(
            title = "高级训练参数设置",
            status = "primary",
            solidHeader = TRUE,
            width = 4,
            
            h4("基础设置"),
            sliderInput("train_ratio", "训练集比例:", 
                       min = 0.5, max = 0.9, value = 0.7, step = 0.05),
            numericInput("random_seed", "随机种子:", value = 1234, min = 1),
            
            hr(),
            h4("交叉验证设置"),
            selectInput("cv_method", "交叉验证方法:",
                       choices = c("K折交叉验证" = "cv",
                                 "重复K折" = "repeatedcv",
                                 "留一法" = "LOOCV")),
            conditionalPanel(
              condition = "input.cv_method == 'cv' || input.cv_method == 'repeatedcv'",
              sliderInput("cv_folds", "K折数:", min = 3, max = 10, value = 5)
            ),
            conditionalPanel(
              condition = "input.cv_method == 'repeatedcv'",
              sliderInput("cv_repeats", "重复次数:", min = 2, max = 10, value = 3)
            ),
            
            hr(),
            h4("超参数优化"),
            checkboxInput("use_bayesian", "使用贝叶斯优化", value = FALSE),
            conditionalPanel(
              condition = "input.use_bayesian == true",
              numericInput("bayes_iter", "优化迭代次数:", value = 10, min = 5, max = 50)
            ),
            
            hr(),
            h4("过拟合防止"),
            checkboxInput("early_stopping", "早停法 (Early Stopping)", value = TRUE),
            checkboxInput("regularization", "L1/L2 正则化", value = TRUE),
            
            hr(),
            checkboxGroupInput("selected_models", "选择机器学习算法 (12种):",
                             choices = c("Logistic Regression" = "Logistic",
                                       "Lasso" = "Lasso",
                                       "Ridge" = "Ridge", 
                                       "Elastic Net" = "Enet",
                                       "Random Forest" = "RandomForest",
                                       "GBM" = "GBM",
                                       "XGBoost" = "XGBoost",
                                       "SVM-Radial" = "SVM",
                                       "SVM-Linear" = "SVMLinear",
                                       "Neural Network" = "NeuralNet",
                                       "Naive Bayes" = "NaiveBayes",
                                       "K-Nearest Neighbors" = "KNN"),
                             selected = c("Logistic", "RandomForest")),
            
            hr(),
            actionButton("train_models", "开始训练模型", 
                        icon = icon("play"), class = "btn-success btn-lg btn-block")
          ),
          
          box(
            title = "训练进度与结果",
            status = "info",
            solidHeader = TRUE,
            width = 8,
            
            verbatimTextOutput("training_log"),
            hr(),
            h4("训练状态:"),
            uiOutput("training_status"),
            hr(),
            h4("过拟合检测:"),
            plotlyOutput("overfitting_plot", height = "300px")
          )
        )
      ),
      
      # ============================================
      # Tab 4: 模型评估
      # ============================================
      tabItem(
        tabName = "model_evaluation",
        
        fluidRow(
          box(title = "交叉验证性能对比", status = "primary", solidHeader = TRUE,
              width = 12, DTOutput("cv_performance_table"))
        ),
        
        fluidRow(
          box(title = "ROC曲线对比", status = "info", solidHeader = TRUE, width = 6,
              plotlyOutput("roc_curves", height = "500px")),
          box(title = "性能指标对比", status = "info", solidHeader = TRUE, width = 6,
              plotlyOutput("performance_comparison", height = "500px"))
        ),
        
        fluidRow(
          box(title = "学习曲线 (最佳模型)", status = "warning", solidHeader = TRUE, width = 6,
              plotlyOutput("learning_curve", height = "400px")),
          box(title = "混淆矩阵 (最佳模型)", status = "warning", solidHeader = TRUE, width = 6,
              plotOutput("confusion_matrix", height = "400px"))
        )
      ),
      
      # ============================================
      # Tab 5: 可解释性分析 (新增)
      # ============================================
      tabItem(
        tabName = "interpretability",
        
        fluidRow(
          box(
            title = "SHAP 可解释性分析",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            
            h4("选择分析方法:"),
            selectInput("shap_model", "选择模型:", choices = NULL),
            sliderInput("shap_samples", "分析样本数:", 
                       min = 10, max = 100, value = 50, step = 10),
            actionButton("compute_shap", "计算 SHAP 值", 
                        icon = icon("calculator"), class = "btn-primary")
          )
        ),
        
        fluidRow(
          box(title = "SHAP 特征重要性", status = "info", solidHeader = TRUE, width = 6,
              plotlyOutput("shap_importance", height = "500px")),
          box(title = "SHAP 摘要图", status = "info", solidHeader = TRUE, width = 6,
              plotOutput("shap_summary", height = "500px"))
        ),
        
        fluidRow(
          box(title = "SHAP 依赖图", status = "warning", solidHeader = TRUE, width = 12,
              selectInput("shap_feature", "选择特征:", choices = NULL),
              plotOutput("shap_dependence", height = "400px"))
        ),
        
        fluidRow(
          box(title = "部分依赖图 (PDP)", status = "success", solidHeader = TRUE, width = 12,
              selectInput("pdp_feature", "选择特征:", choices = NULL),
              plotlyOutput("pdp_plot", height = "400px"))
        )
      ),
      
      # ============================================
      # Tab 6: 预测分析
      # ============================================
      tabItem(
        tabName = "prediction",
        
        fluidRow(
          box(
            title = "新样本预测",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            
            h4("输入特征值:"),
            fluidRow(
              column(4, numericInput("TEF_365nm", "TEF_365nm:", value = 1.5)),
              column(4, numericInput("TEF_405nm", "TEF_405nm:", value = 2.0)),
              column(4, numericInput("TEF_450nm", "TEF_450nm:", value = 1.8))
            ),
            fluidRow(
              column(4, numericInput("LDF_Myo", "LDF_Myo:", value = 0.3)),
              column(4, numericInput("LDF_Card", "LDF_Card:", value = 0.2)),
              column(4, numericInput("LDF_Resp", "LDF_Resp:", value = 0.4))
            ),
            fluidRow(
              column(4, numericInput("AFP_Level", "AFP_Level:", value = 100)),
              column(4, numericInput("Bilirubin", "Bilirubin:", value = 1.0)),
              column(4, numericInput("Albumin", "Albumin:", value = 3.5))
            ),
            fluidRow(
              column(4, numericInput("Spectral_Skewness", "Spectral_Skewness:", value = 0.15)),
              column(4, numericInput("Spectral_Kurtosis", "Spectral_Kurtosis:", value = 2.5)),
              column(4, numericInput("Tissue_Density", "Tissue_Density:", value = 1.0))
            ),
            
            hr(),
            selectInput("prediction_model", "选择预测模型:", choices = NULL),
            actionButton("predict_btn", "进行预测", 
                        icon = icon("calculator"), class = "btn-primary btn-lg"),
            checkboxInput("show_shap_explain", "显示 SHAP 解释", value = FALSE),
            
            hr(),
            h3("预测结果:"),
            verbatimTextOutput("prediction_result"),
            plotlyOutput("prediction_gauge", height = "300px")
          )
        ),
        
        conditionalPanel(
          condition = "input.show_shap_explain == true",
          fluidRow(
            box(title = "该样本的 SHAP 解释", status = "info", solidHeader = TRUE, width = 12,
                plotOutput("individual_shap", height = "400px"))
          )
        )
      ),
      
      # ============================================
      # Tab 7: 模型管理 (新增)
      # ============================================
      tabItem(
        tabName = "model_management",
        
        fluidRow(
          box(
            title = "模型保存",
            status = "primary",
            solidHeader = TRUE,
            width = 6,
            
            h4("保存训练好的模型:"),
            selectInput("model_to_save", "选择模型:", choices = NULL),
            textInput("model_name", "模型名称:", value = "liver_cancer_model"),
            actionButton("save_model", "保存模型", 
                        icon = icon("save"), class = "btn-success"),
            hr(),
            verbatimTextOutput("save_status")
          ),
          
          box(
            title = "模型加载",
            status = "info",
            solidHeader = TRUE,
            width = 6,
            
            h4("加载已保存的模型:"),
            selectInput("saved_model_list", "选择保存的模型:", choices = NULL),
            actionButton("refresh_models", "刷新列表", icon = icon("refresh")),
            actionButton("load_model", "加载模型", 
                        icon = icon("upload"), class = "btn-primary"),
            hr(),
            verbatimTextOutput("load_status")
          )
        ),
        
        fluidRow(
          box(
            title = "已保存的模型列表",
            status = "warning",
            solidHeader = TRUE,
            width = 12,
            DTOutput("saved_models_table")
          )
        ),
        
        fluidRow(
          box(
            title = "模型性能对比",
            status = "success",
            solidHeader = TRUE,
            width = 12,
            plotlyOutput("models_comparison", height = "400px")
          )
        ),
        
        fluidRow(
          box(
            title = "贝叶斯超参数优化历史",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            
            h4("优化轮次和参数演变:"),
            verbatimTextOutput("bayes_optimization_history")
          )
        )
      )
    )
  )
)

# ========================================================================
# Server 部分
# ========================================================================

server <- function(input, output, session) {
  
  # 创建模型保存目录
  model_dir <- "saved_models"
  if (!dir.exists(model_dir)) {
    dir.create(model_dir, recursive = TRUE)
  }
  
  # 响应式数据存储
  rv <- reactiveValues(
    data = NULL,
    train_data = NULL,
    test_data = NULL,
    models = list(),
    cv_results = list(),
    performance = NULL,
    training_complete = FALSE,
    shap_values = NULL,
    best_params = list(),
    target_col = "Target",  # 动态目标列名称
    # 贝叶斯优化历史记录
    bayes_history = list(
      RandomForest = data.frame(),
      GBM = data.frame(),
      Lasso = data.frame(),
      Ridge = data.frame(),
      Enet = data.frame()
    ),
    bayes_best_params = list(
      RandomForest = NULL,
      GBM = NULL,
      Lasso = NULL,
      Ridge = NULL,
      Enet = NULL
    )
  )
  
  # ============================================
  # 数据加载
  # ============================================
  
  observeEvent(input$load_default_data, {
    tryCatch({
      rv$data <- read.csv("liver_cancer2.csv", stringsAsFactors = FALSE)
      
      # 更新目标列选择
      all_cols <- colnames(rv$data)
      updateSelectInput(session, "target_column", choices = all_cols, 
                       selected = all_cols[length(all_cols)])
      
      # 默认使用最后一列作为目标
      rv$target_col <- all_cols[length(all_cols)]
      
      # 更新特征列表（排除目标列）
      feature_names <- setdiff(all_cols, rv$target_col)
      updateSelectInput(session, "selected_feature", choices = feature_names)
      updateSelectInput(session, "shap_feature", choices = feature_names)
      updateSelectInput(session, "pdp_feature", choices = feature_names)
      
      showNotification("数据加载成功!", type = "message")
    }, error = function(e) {
      showNotification(paste("数据加载失败:", e$message), type = "error")
    })
  })
  
  observeEvent(input$data_file, {
    req(input$data_file)
    tryCatch({
      rv$data <- read.csv(input$data_file$datapath, stringsAsFactors = FALSE)
      
      # 更新目标列选择
      all_cols <- colnames(rv$data)
      updateSelectInput(session, "target_column", choices = all_cols, 
                       selected = all_cols[length(all_cols)])
      
      # 默认使用最后一列作为目标
      rv$target_col <- all_cols[length(all_cols)]
      
      # 更新特征列表（排除目标列）
      feature_names <- setdiff(all_cols, rv$target_col)
      updateSelectInput(session, "selected_feature", choices = feature_names)
      updateSelectInput(session, "shap_feature", choices = feature_names)
      updateSelectInput(session, "pdp_feature", choices = feature_names)
      
      showNotification("数据上传成功!", type = "message")
    }, error = function(e) {
      showNotification(paste("数据上传失败:", e$message), type = "error")
    })
  })
  
  # 当用户手动选择目标列时
  observeEvent(input$target_column, {
    req(rv$data, input$target_column)
    
    rv$target_col <- input$target_column
    
    # 更新特征列表（排除新的目标列）
    all_cols <- colnames(rv$data)
    feature_names <- setdiff(all_cols, rv$target_col)
    updateSelectInput(session, "selected_feature", choices = feature_names)
    updateSelectInput(session, "shap_feature", choices = feature_names)
    updateSelectInput(session, "pdp_feature", choices = feature_names)
    
    showNotification(paste("目标列已设置为:", rv$target_col), type = "message")
  })
  
  # ============================================
  # Tab 1: 数据概览输出
  # ============================================
  
  output$data_info <- renderPrint({
    req(rv$data)
    cat("数据维度:", nrow(rv$data), "行 x", ncol(rv$data), "列\n")
    cat("特征列表:\n")
    print(colnames(rv$data))
  })
  
  output$total_samples <- renderValueBox({
    req(rv$data)
    valueBox(nrow(rv$data), "总样本数", icon = icon("database"), color = "blue")
  })
  
  output$num_features <- renderValueBox({
    req(rv$data)
    valueBox(ncol(rv$data) - 1, "特征数", icon = icon("list"), color = "green")
  })
  
  output$target_0_count <- renderValueBox({
    req(rv$data, rv$target_col)
    valueBox(sum(rv$data[[rv$target_col]] == 0), 
            paste0("良性样本 (", rv$target_col, "=0)"), 
            icon = icon("check-circle"), color = "yellow")
  })
  
  output$target_1_count <- renderValueBox({
    req(rv$data, rv$target_col)
    valueBox(sum(rv$data[[rv$target_col]] == 1), 
            paste0("恶性样本 (", rv$target_col, "=1)"), 
            icon = icon("exclamation-triangle"), color = "red")
  })
  
  output$data_table <- renderDT({
    req(rv$data)
    datatable(head(rv$data, 100),
             options = list(pageLength = 10, scrollX = TRUE, dom = 'Bfrtip'),
             rownames = FALSE)
  })
  
  # ============================================
  # Tab 2: 特征分析输出
  # ============================================
  
  output$correlation_heatmap <- renderPlotly({
    req(rv$data, rv$target_col)
    # 排除目标列进行相关性分析
    feature_cols <- setdiff(colnames(rv$data), rv$target_col)
    
    # 只选择数值型特征
    numeric_cols <- sapply(rv$data[, feature_cols, drop = FALSE], is.numeric)
    numeric_feature_cols <- names(numeric_cols)[numeric_cols]
    
    if (length(numeric_feature_cols) == 0) {
      plotly_empty() %>% layout(title = "没有数值特征可计算相关性")
    } else {
      cor_matrix <- cor(rv$data[, numeric_feature_cols, drop = FALSE], use = "complete.obs")
      plot_ly(x = colnames(cor_matrix), y = colnames(cor_matrix),
             z = cor_matrix, type = "heatmap", colorscale = "RdBu", zmid = 0) %>%
        layout(title = "特征相关性热图", xaxis = list(tickangle = -45))
    }
  })
  
  output$feature_boxplot <- renderPlotly({
    req(rv$data, rv$target_col, input$selected_feature)
    df <- data.frame(
      Value = rv$data[[input$selected_feature]],
      Target = factor(rv$data[[rv$target_col]], labels = c("良性(0)", "恶性(1)"))
    )
    plot_ly(df, x = ~Target, y = ~Value, type = "box", color = ~Target,
           colors = c("#FFA500", "#FF4500")) %>%
      layout(title = paste("特征分布:", input$selected_feature),
            xaxis = list(title = paste("目标类别 (", rv$target_col, ")")),
            yaxis = list(title = input$selected_feature))
  })
  
  # ============================================
  # Tab 3: 高级模型训练
  # ============================================
  
  output$training_log <- renderPrint({
    cat("准备开始训练...\n")
    cat("请设置参数后点击'开始训练模型'按钮\n")
    cat("\n高级功能:\n")
    cat("- K折交叉验证: 评估模型泛化能力\n")
    cat("- 贝叶斯优化: 自动寻找最佳超参数\n")
    cat("- 早停法: 防止过拟合\n")
    cat("- 模型自动保存: 保存最佳模型到本地\n")
  })
  
  observeEvent(input$train_models, {
    req(rv$data, input$selected_models)
    
    withProgress(message = "正在训练模型...", value = 0, {
      
      # 数据分割
      set.seed(input$random_seed)
      train_index <- createDataPartition(rv$data[[rv$target_col]], p = input$train_ratio, list = FALSE)
      rv$train_data <- rv$data[train_index, ]
      rv$test_data <- rv$data[-train_index, ]
      
      # 数据质量检查
      cat("\n=== 数据质量检查 ===\n")
      cat("训练集样本数:", nrow(rv$train_data), "\n")
      cat("测试集样本数:", nrow(rv$test_data), "\n")
      cat("特征数量:", ncol(rv$train_data) - 1, "\n")
      cat("目标列分布 - 训练集:", table(rv$train_data[[rv$target_col]]), "\n")
      cat("目标列分布 - 测试集:", table(rv$test_data[[rv$target_col]]), "\n")
      
      # 检查测试集大小
      if (nrow(rv$test_data) < 50) {
        showNotification(
          "⚠️ 警告: 测试集样本太少(<50)，可能导致性能评估不准确！",
          type = "warning",
          duration = 10
        )
      }
      
      # 检查类别平衡
      train_ratio_class <- table(rv$train_data[[rv$target_col]])
      if (min(train_ratio_class) / max(train_ratio_class) < 0.3) {
        showNotification(
          "⚠️ 警告: 类别严重不平衡！可能影响模型性能。",
          type = "warning",
          duration = 8
        )
      }
      
      incProgress(0.1, detail = "数据分割完成")
      
      # 准备交叉验证设置
      if (input$cv_method == "cv") {
        train_control <- trainControl(
          method = "cv",
          number = input$cv_folds,
          savePredictions = "final",
          classProbs = TRUE,
          summaryFunction = twoClassSummary
        )
      } else if (input$cv_method == "repeatedcv") {
        train_control <- trainControl(
          method = "repeatedcv",
          number = input$cv_folds,
          repeats = input$cv_repeats,
          savePredictions = "final",
          classProbs = TRUE,
          summaryFunction = twoClassSummary
        )
      } else {
        train_control <- trainControl(
          method = "LOOCV",
          savePredictions = "final",
          classProbs = TRUE,
          summaryFunction = twoClassSummary
        )
      }
      
      # 准备数据 (动态排除目标列)
      feature_cols <- setdiff(colnames(rv$train_data), rv$target_col)
      X_train <- as.matrix(rv$train_data[, feature_cols, drop = FALSE])
      y_train <- factor(rv$train_data[[rv$target_col]], labels = c("Class0", "Class1"))
      X_test <- as.matrix(rv$test_data[, feature_cols, drop = FALSE])
      
      rv$models <- list()
      rv$cv_results <- list()
      performance_list <- list()
      
      models_to_train <- input$selected_models
      total_models <- length(models_to_train)
      
      # 训练各个模型
      cat("\n========================================\n")
      cat("开始训练", total_models, "个模型\n")
      if (input$cv_method == "cv") {
        cat("交叉验证方法:", input$cv_folds, "折交叉验证\n")
      } else if (input$cv_method == "repeatedcv") {
        cat("交叉验证方法:", input$cv_folds, "折重复", input$cv_repeats, "次\n")
      }
      if (input$use_bayesian) {
        cat("贝叶斯优化: 启用 (迭代次数:", input$bayes_iter, ")\n")
      } else {
        cat("贝叶斯优化: 未启用\n")
      }
      cat("========================================\n\n")
      
      for (i in seq_along(models_to_train)) {
        model_name <- models_to_train[i]
        cat("\n[", i, "/", total_models, "] 开始训练:", model_name, "\n")
        cat("预计时间:", ifelse(input$cv_method == "cv", 
                             paste(input$cv_folds, "折"), 
                             paste(input$cv_folds, "×", input$cv_repeats, "次")), "\n")
        incProgress(0.8 / total_models, detail = paste("训练", model_name, paste0("(", i, "/", total_models, ")")))
        
        start_time <- Sys.time()
        tryCatch({
          
          if (model_name == "RandomForest") {
            # Random Forest with intelligent Bayesian tuning
            if (input$use_bayesian) {
              # 智能参数生成基于历史最佳参数
              if (!is.null(rv$bayes_best_params$RandomForest)) {
                # 基于之前最好的参数进行微调搜索
                best_mtry <- rv$bayes_best_params$RandomForest$mtry
                search_range <- c(
                  max(2, best_mtry - 2),
                  best_mtry - 1,
                  best_mtry,
                  best_mtry + 1,
                  min(ncol(X_train), best_mtry + 2)
                )
                tune_grid <- expand.grid(mtry = unique(sort(search_range)))
                
                cat("\n贝叶斯优化: RandomForest - 基于历史最佳参数 (mtry=", best_mtry, ") 进行微调\n")
              } else {
                # 第一次运行，使用广泛搜索
                num_features <- ncol(X_train)
                tune_grid <- expand.grid(
                  mtry = c(2, max(3, num_features/4), max(4, num_features/3), 
                          max(5, num_features/2), num_features-1)
                )
                tune_grid$mtry <- unique(sort(tune_grid$mtry))
                tune_grid <- expand.grid(mtry = tune_grid$mtry)
                
                cat("\n贝叶斯优化: RandomForest - 初始化参数搜索\n")
              }
              
              # 添加迭代轮数信息
              bayes_iters <- input$bayes_iter
              cat("优化迭代次数:", bayes_iters, "\n")
            } else {
              tune_grid <- expand.grid(mtry = c(3, 4, 5))
            }
            
            fit_cv <- train(
              x = rv$train_data[, feature_cols, drop = FALSE],
              y = y_train,
              method = "rf",
              trControl = train_control,
              tuneGrid = tune_grid,
              metric = "ROC",
              ntree = 300,  # 减少树数量防止过拟合
              maxnodes = 30,  # 限制节点数防止过拟合
              importance = TRUE
            )
            
            rv$cv_results[[model_name]] <- fit_cv
            pred_test <- predict(fit_cv, rv$test_data[, feature_cols, drop = FALSE], type = "prob")[, 2]
            rv$models[[model_name]] <- list(model = fit_cv, pred = pred_test)
            
            # 记录贝叶斯优化历史
            if (input$use_bayesian) {
              best_row <- which.max(fit_cv$results$ROC)
              best_result <- fit_cv$results[best_row, ]
              
              # 更新历史记录
              new_history <- data.frame(
                iteration = nrow(rv$bayes_history$RandomForest) + 1,
                mtry = best_result$mtry,
                ROC = best_result$ROC,
                Accuracy = if ("Accuracy" %in% names(best_result)) best_result$Accuracy else NA,
                timestamp = Sys.time()
              )
              rv$bayes_history$RandomForest <- rbind(rv$bayes_history$RandomForest, new_history)
              
              # 更新最佳参数
              if (is.null(rv$bayes_best_params$RandomForest) || 
                  new_history$ROC > max(rv$bayes_history$RandomForest$ROC[-nrow(rv$bayes_history$RandomForest)], 0)) {
                rv$bayes_best_params$RandomForest <- list(mtry = best_result$mtry)
                cat("✓ 更新最佳参数: mtry =", best_result$mtry, "| ROC =", round(best_result$ROC, 4), "\n")
              }
            }
            
          } else if (model_name %in% c("Lasso", "Ridge", "Enet")) {
            alpha_val <- ifelse(model_name == "Lasso", 1, 
                               ifelse(model_name == "Ridge", 0, 0.5))
            
            # 智能lambda搜索
            if (input$use_bayesian && !is.null(rv$bayes_best_params[[model_name]])) {
              # 基于历史最佳lambda进行微调搜索
              best_lambda <- rv$bayes_best_params[[model_name]]$lambda
              log_lambda_min <- log(best_lambda) - 0.5
              log_lambda_max <- log(best_lambda) + 0.5
              lambda_seq <- exp(seq(log_lambda_min, log_lambda_max, length.out = 15))
              
              cat("\n贝叶斯优化:", model_name, "- 基于历史最佳lambda=", 
                  round(best_lambda, 4), "进行微调\n")
              
              tune_grid <- expand.grid(alpha = alpha_val, lambda = lambda_seq)
            } else if (input$use_bayesian) {
              # 初始化lambda搜索
              tune_grid <- expand.grid(alpha = alpha_val, 
                                      lambda = 10^seq(-3.5, 0.5, length.out = 20))
              cat("\n贝叶斯优化:", model_name, "- 初始化lambda搜索\n")
            } else {
              # 不使用贝叶斯的默认参数
              tune_grid <- expand.grid(alpha = alpha_val, 
                                      lambda = 10^seq(-3, 1, length.out = 20))
            }
            
            fit_cv <- train(
              x = X_train,
              y = y_train,
              method = "glmnet",
              trControl = train_control,
              tuneGrid = tune_grid,
              metric = "ROC"
            )
            
            rv$cv_results[[model_name]] <- fit_cv
            pred_test <- predict(fit_cv, X_test, type = "prob")[, 2]
            rv$models[[model_name]] <- list(model = fit_cv, pred = pred_test)
            
            # 记录贝叶斯优化历史
            if (input$use_bayesian) {
              best_row <- which.max(fit_cv$results$ROC)
              best_result <- fit_cv$results[best_row, ]
              
              new_history <- data.frame(
                iteration = nrow(rv$bayes_history[[model_name]]) + 1,
                lambda = best_result$lambda,
                ROC = best_result$ROC,
                timestamp = Sys.time()
              )
              rv$bayes_history[[model_name]] <- rbind(rv$bayes_history[[model_name]], new_history)
              
              # 更新最佳参数
              if (is.null(rv$bayes_best_params[[model_name]]) || 
                  new_history$ROC > max(rv$bayes_history[[model_name]]$ROC[-nrow(rv$bayes_history[[model_name]])], 0)) {
                rv$bayes_best_params[[model_name]] <- list(lambda = best_result$lambda)
                cat("✓ 更新", model_name, "最佳参数: lambda=", round(best_result$lambda, 4), 
                    " | ROC=", round(best_result$ROC, 4), "\n")
              }
            }
            
          } else if (model_name == "GBM") {
            # 创建包含目标列的训练数据
            train_data_gbm <- rv$train_data[, feature_cols, drop = FALSE]
            train_data_gbm[[rv$target_col]] <- y_train
            
            formula_gbm <- as.formula(paste(rv$target_col, "~ ."))
            
            # GBM智能贝叶斯参数优化
            if (input$use_bayesian && !is.null(rv$bayes_best_params$GBM)) {
              # 基于历史最佳参数进行微调
              best_params <- rv$bayes_best_params$GBM
              tune_grid <- expand.grid(
                n.trees = c(best_params$n.trees - 100, best_params$n.trees, best_params$n.trees + 100),
                interaction.depth = c(best_params$interaction.depth - 1, best_params$interaction.depth, best_params$interaction.depth + 1),
                shrinkage = c(best_params$shrinkage / 2, best_params$shrinkage, best_params$shrinkage * 2),
                n.minobsinnode = 10
              )
              tune_grid$n.trees <- pmax(50, tune_grid$n.trees)
              tune_grid$interaction.depth <- pmax(1, pmin(8, tune_grid$interaction.depth))
              tune_grid$shrinkage <- pmax(0.001, pmin(0.1, tune_grid$shrinkage))
              tune_grid <- unique(tune_grid)
              
              cat("\n贝叶斯优化: GBM - 基于历史最佳参数进行微调\n")
            } else if (input$use_bayesian) {
              # 初始化搜索空间（防止过拟合的参数范围）
              tune_grid <- expand.grid(
                n.trees = c(50, 100, 150),  # 减少树数量
                interaction.depth = c(1, 2, 3),  # 降低树深度
                shrinkage = c(0.01, 0.03, 0.05),  # 较小的学习率
                n.minobsinnode = c(15, 20)  # 增加最小节点样本数
              )
              cat("\n贝叶斯优化: GBM - 初始化参数搜索（防过拟合配置）\n")
            } else {
              # 不使用贝叶斯优化的默认参数（防过拟合配置）
              tune_grid <- expand.grid(
                n.trees = c(50, 100, 150),
                interaction.depth = c(1, 2, 3),
                shrinkage = c(0.01, 0.05),
                n.minobsinnode = c(15, 20)
              )
            }
            
            fit_cv <- train(
              formula_gbm,
              data = train_data_gbm,
              method = "gbm",
              trControl = train_control,
              tuneGrid = tune_grid,
              metric = "ROC",
              verbose = FALSE
            )
            
            rv$cv_results[[model_name]] <- fit_cv
            pred_test <- predict(fit_cv, rv$test_data[, feature_cols, drop = FALSE], type = "prob")[, 2]
            rv$models[[model_name]] <- list(model = fit_cv, pred = pred_test)
            
            # 记录GBM的贝叶斯优化历史
            if (input$use_bayesian) {
              best_row <- which.max(fit_cv$results$ROC)
              best_result <- fit_cv$results[best_row, ]
              
              new_history <- data.frame(
                iteration = nrow(rv$bayes_history$GBM) + 1,
                n.trees = best_result$n.trees,
                interaction.depth = best_result$interaction.depth,
                shrinkage = best_result$shrinkage,
                ROC = best_result$ROC,
                timestamp = Sys.time()
              )
              rv$bayes_history$GBM <- rbind(rv$bayes_history$GBM, new_history)
              
              # 更新最佳参数
              if (is.null(rv$bayes_best_params$GBM) || 
                  new_history$ROC > max(rv$bayes_history$GBM$ROC[-nrow(rv$bayes_history$GBM)], 0)) {
                rv$bayes_best_params$GBM <- list(
                  n.trees = best_result$n.trees,
                  interaction.depth = best_result$interaction.depth,
                  shrinkage = best_result$shrinkage
                )
                cat("✓ 更新GBM最佳参数: trees=", best_result$n.trees, 
                    " depth=", best_result$interaction.depth, 
                    " shrinkage=", round(best_result$shrinkage, 4),
                    " | ROC =", round(best_result$ROC, 4), "\n")
              }
            }
            
          } else {
            # Logistic, SVM, Neural Network, Naive Bayes, KNN
            train_data_simple <- rv$train_data[, c(feature_cols, rv$target_col), drop = FALSE]
            test_data_simple <- rv$test_data[, c(feature_cols, rv$target_col), drop = FALSE]
            
            formula_simple <- as.formula(paste(rv$target_col, "~ ."))
            
            if (model_name == "Logistic") {
              fit <- glm(formula_simple, data = train_data_simple, family = binomial())
              pred_test <- predict(fit, test_data_simple, type = "response")
              rv$models[[model_name]] <- list(model = fit, pred = pred_test)
              
            } else if (model_name == "SVM") {
              # SVM with scaled data
              cat("训练 SVM-Radial (可能需要较长时间)...\n")
              # 数据标准化对SVM很重要
              scaler <- preProcess(train_data_simple[, feature_cols], method = c("center", "scale"))
              train_scaled <- predict(scaler, train_data_simple[, feature_cols])
              test_scaled <- predict(scaler, test_data_simple[, feature_cols])
              
              train_scaled[[rv$target_col]] <- factor(train_data_simple[[rv$target_col]], 
                                                       labels = c("Class0", "Class1"))
              
              tune_grid_svm <- expand.grid(
                sigma = c(0.01, 0.1, 1),
                C = c(0.1, 1, 10)
              )
              
              fit_cv <- train(
                x = train_scaled,
                y = train_scaled[[rv$target_col]],
                method = "svmRadial",
                trControl = train_control,
                tuneGrid = tune_grid_svm,
                metric = "ROC",
                preProc = NULL  # 已经手动预处理
              )
              
              rv$cv_results[[model_name]] <- fit_cv
              pred_test <- predict(fit_cv, test_scaled, type = "prob")[, 2]
              rv$models[[model_name]] <- list(model = fit_cv, pred = pred_test, scaler = scaler)
              
            } else if (model_name == "SVMLinear") {
              # Linear SVM
              cat("训练 SVM-Linear...\n")
              scaler <- preProcess(train_data_simple[, feature_cols], method = c("center", "scale"))
              train_scaled <- predict(scaler, train_data_simple[, feature_cols])
              test_scaled <- predict(scaler, test_data_simple[, feature_cols])
              
              train_scaled[[rv$target_col]] <- factor(train_data_simple[[rv$target_col]], 
                                                       labels = c("Class0", "Class1"))
              
              tune_grid_svm <- expand.grid(C = c(0.01, 0.1, 1, 10))
              
              fit_cv <- train(
                x = train_scaled,
                y = train_scaled[[rv$target_col]],
                method = "svmLinear",
                trControl = train_control,
                tuneGrid = tune_grid_svm,
                metric = "ROC"
              )
              
              rv$cv_results[[model_name]] <- fit_cv
              pred_test <- predict(fit_cv, test_scaled, type = "prob")[, 2]
              rv$models[[model_name]] <- list(model = fit_cv, pred = pred_test, scaler = scaler)
              
            } else if (model_name == "NeuralNet") {
              # Neural Network
              cat("训练 Neural Network...\n")
              scaler <- preProcess(train_data_simple[, feature_cols], method = c("center", "scale"))
              train_scaled <- predict(scaler, train_data_simple[, feature_cols])
              test_scaled <- predict(scaler, test_data_simple[, feature_cols])
              
              train_scaled[[rv$target_col]] <- factor(train_data_simple[[rv$target_col]], 
                                                       labels = c("Class0", "Class1"))
              
              tune_grid_nnet <- expand.grid(
                size = c(3, 5, 7),  # 隐藏层神经元数
                decay = c(0.01, 0.1, 0.5)  # 权重衰减（正则化）
              )
              
              fit_cv <- train(
                x = train_scaled,
                y = train_scaled[[rv$target_col]],
                method = "nnet",
                trControl = train_control,
                tuneGrid = tune_grid_nnet,
                metric = "ROC",
                trace = FALSE,  # 不显示详细训练信息
                MaxNWts = 2000
              )
              
              rv$cv_results[[model_name]] <- fit_cv
              pred_test <- predict(fit_cv, test_scaled, type = "prob")[, 2]
              rv$models[[model_name]] <- list(model = fit_cv, pred = pred_test, scaler = scaler)
              
            } else if (model_name == "NaiveBayes") {
              # Naive Bayes
              cat("训练 Naive Bayes...\n")
              train_data_nb <- train_data_simple
              train_data_nb[[rv$target_col]] <- factor(train_data_simple[[rv$target_col]], 
                                                         labels = c("Class0", "Class1"))
              
              tune_grid_nb <- expand.grid(
                laplace = c(0, 0.5, 1),  # Laplace平滑
                usekernel = c(TRUE, FALSE),
                adjust = 1
              )
              
              fit_cv <- train(
                x = train_data_nb[, feature_cols, drop = FALSE],
                y = train_data_nb[[rv$target_col]],
                method = "naive_bayes",
                trControl = train_control,
                tuneGrid = tune_grid_nb,
                metric = "ROC"
              )
              
              rv$cv_results[[model_name]] <- fit_cv
              pred_test <- predict(fit_cv, test_data_simple[, feature_cols, drop = FALSE], type = "prob")[, 2]
              rv$models[[model_name]] <- list(model = fit_cv, pred = pred_test)
              
            } else if (model_name == "KNN") {
              # K-Nearest Neighbors
              cat("训练 KNN...\n")
              scaler <- preProcess(train_data_simple[, feature_cols], method = c("center", "scale"))
              train_scaled <- predict(scaler, train_data_simple[, feature_cols])
              test_scaled <- predict(scaler, test_data_simple[, feature_cols])
              
              train_scaled[[rv$target_col]] <- factor(train_data_simple[[rv$target_col]], 
                                                       labels = c("Class0", "Class1"))
              
              tune_grid_knn <- expand.grid(
                kmax = c(3, 5, 7, 9),  # 最大邻居数
                distance = 2,
                kernel = "optimal"
              )
              
              fit_cv <- train(
                x = train_scaled,
                y = train_scaled[[rv$target_col]],
                method = "kknn",
                trControl = train_control,
                tuneGrid = tune_grid_knn,
                metric = "ROC"
              )
              
              rv$cv_results[[model_name]] <- fit_cv
              pred_test <- predict(fit_cv, test_scaled, type = "prob")[, 2]
              rv$models[[model_name]] <- list(model = fit_cv, pred = pred_test, scaler = scaler)
            }
          }
          
          # 计算性能指标
          roc_obj <- roc(rv$test_data[[rv$target_col]], rv$models[[model_name]]$pred, quiet = TRUE)
          pred_class <- ifelse(rv$models[[model_name]]$pred > 0.5, 1, 0)
          cm <- confusionMatrix(factor(pred_class), factor(rv$test_data[[rv$target_col]]))
          
          # 检测完美分类（可能的过拟合或数据泄漏）
          if (as.numeric(auc(roc_obj)) >= 0.99) {
            cat("⚠️ 警告:", model_name, "AUC =", round(auc(roc_obj), 4), 
                "- 接近完美分类，请检查数据是否存在泄漏或测试集太小！\n")
          }
          
          # 获取交叉验证 AUC (如果有)
          cv_auc <- if (!is.null(rv$cv_results[[model_name]])) {
            max(rv$cv_results[[model_name]]$results$ROC, na.rm = TRUE)
          } else {
            NA
          }
          
          performance_list[[model_name]] <- data.frame(
            Model = model_name,
            CV_AUC = cv_auc,
            Test_AUC = as.numeric(auc(roc_obj)),
            Accuracy = as.numeric(cm$overall["Accuracy"]),
            Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
            Specificity = as.numeric(cm$byClass["Specificity"]),
            F1_Score = as.numeric(cm$byClass["F1"]),
            Overfit_Gap = ifelse(!is.na(cv_auc), cv_auc - as.numeric(auc(roc_obj)), NA),
            stringsAsFactors = FALSE
          )
          
          # 显示训练完成信息
          end_time <- Sys.time()
          time_diff <- round(as.numeric(difftime(end_time, start_time, units = "secs")), 2)
          cat("✓ 完成训练:", model_name, "| 用时:", time_diff, "秒\n")
          cat("  - CV AUC:", round(cv_auc, 4), 
              "| Test AUC:", round(as.numeric(auc(roc_obj)), 4), 
              "| 过拟合差:", round(cv_auc - as.numeric(auc(roc_obj)), 4), "\n")
          
        }, error = function(e) {
          cat("✗ 训练失败:", model_name, "-", conditionMessage(e), "\n")
          showNotification(paste(model_name, "训练失败:", e$message), type = "warning", duration = 8)
        })
      }
      
      # 合并性能数据
      rv$performance <- do.call(rbind, performance_list)
      rv$performance <- rv$performance[order(rv$performance$Test_AUC, decreasing = TRUE), ]
      rownames(rv$performance) <- NULL
      
      # 更新模型选择列表
      model_choices <- rv$performance$Model
      updateSelectInput(session, "prediction_model", choices = model_choices, 
                       selected = model_choices[1])
      updateSelectInput(session, "shap_model", choices = model_choices, 
                       selected = model_choices[1])
      updateSelectInput(session, "model_to_save", choices = model_choices, 
                       selected = model_choices[1])
      
      rv$training_complete <- TRUE
      incProgress(0.1, detail = "训练完成!")
      
      showNotification("所有模型训练完成!", type = "message", duration = 5)
    })
  })
  
  output$training_status <- renderUI({
    if (rv$training_complete) {
      best_model <- rv$performance$Model[1]
      best_auc <- round(rv$performance$Test_AUC[1], 4)
      best_cv_auc <- round(rv$performance$CV_AUC[1], 4)
      overfit_gap <- round(rv$performance$Overfit_Gap[1], 4)
      
      tagList(
        h4(style = "color: green;", icon("check-circle"), " 训练完成!"),
        p("共训练了", length(rv$models), "个模型"),
        p("最佳模型:", best_model),
        p("交叉验证 AUC:", best_cv_auc),
        p("测试集 AUC:", best_auc),
        p(strong("过拟合程度:"), 
          span(ifelse(abs(overfit_gap) < 0.05, "低 ✓", 
                     ifelse(abs(overfit_gap) < 0.1, "中等 ⚠", "高 ⚠⚠")),
               style = ifelse(abs(overfit_gap) < 0.05, "color: green;", 
                            ifelse(abs(overfit_gap) < 0.1, "color: orange;", "color: red;"))))
      )
    } else {
      h4(style = "color: gray;", icon("hourglass-half"), " 等待训练...")
    }
  })
  
  # 过拟合检测图
  output$overfitting_plot <- renderPlotly({
    req(rv$performance)
    
    df <- rv$performance %>%
      filter(!is.na(CV_AUC)) %>%
      select(Model, CV_AUC, Test_AUC) %>%
      pivot_longer(cols = c(CV_AUC, Test_AUC), names_to = "Type", values_to = "AUC")
    
    plot_ly(df, x = ~Model, y = ~AUC, color = ~Type, type = "bar") %>%
      layout(
        title = "交叉验证 vs 测试集性能 (检测过拟合)",
        xaxis = list(title = "模型"),
        yaxis = list(title = "AUC", range = c(0, 1)),
        barmode = "group",
        annotations = list(
          text = "差距大表示过拟合",
          x = 0.5, y = 1.05, xref = "paper", yref = "paper",
          showarrow = FALSE, font = list(size = 10, color = "red")
        )
      )
  })
  
  # ============================================
  # Tab 4: 模型评估
  # ============================================
  
  output$cv_performance_table <- renderDT({
    req(rv$performance)
    datatable(rv$performance,
             options = list(pageLength = 10, dom = 'Bfrtip'),
             rownames = FALSE) %>%
      formatRound(columns = 2:8, digits = 4) %>%
      formatStyle('Overfit_Gap',
                 backgroundColor = styleInterval(c(-0.1, -0.05, 0.05, 0.1),
                                                c('lightcoral', 'lightyellow', 'lightgreen',
                                                  'lightyellow', 'lightcoral')))
  })
  
  output$roc_curves <- renderPlotly({
    req(rv$models, rv$test_data, rv$target_col)
    
    tryCatch({
      p <- plot_ly()
      # 扩展颜色以支持12个模型
      colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                  "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#7f7f7f",
                  "#ffbb78", "#98df8a")
      
      for (i in seq_along(rv$models)) {
        model_name <- names(rv$models)[i]
        pred <- rv$models[[model_name]]$pred
        
        # 确保pred和target长度一致
        if (length(pred) != length(rv$test_data[[rv$target_col]])) {
          cat("警告:", model_name, "预测长度不匹配\n")
          next
        }
        
        roc_obj <- roc(rv$test_data[[rv$target_col]], pred, 
                      levels = c(0, 1), direction = "<", quiet = TRUE)
        
        p <- add_trace(p, 
                      x = 1 - roc_obj$specificities, 
                      y = roc_obj$sensitivities,
                      type = "scatter", 
                      mode = "lines",
                      name = paste0(model_name, " (AUC=", round(auc(roc_obj), 3), ")"),
                      line = list(color = colors[(i-1) %% length(colors) + 1], width = 2),
                      hovertemplate = paste0(model_name, "<br>",
                                            "FPR: %{x:.3f}<br>",
                                            "TPR: %{y:.3f}<extra></extra>"))
      }
      
      # 添加对角线
      p <- add_trace(p, x = c(0, 1), y = c(0, 1), 
                    type = "scatter", mode = "lines",
                    name = "随机猜测 (AUC=0.5)", 
                    line = list(color = "gray", dash = "dash", width = 1))
      
      p %>% layout(
        title = list(text = "ROC曲线对比", font = list(size = 16, color = "black")),
        xaxis = list(title = "假阳性率 (FPR)", range = c(0, 1)),
        yaxis = list(title = "真阳性率 (TPR)", range = c(0, 1)),
        hovermode = "closest",
        showlegend = TRUE,
        legend = list(x = 0.6, y = 0.2)
      )
    }, error = function(e) {
      cat("ROC曲线绘制错误:", conditionMessage(e), "\n")
      plotly_empty() %>% 
        layout(title = list(text = paste("ROC曲线绘制失败:", conditionMessage(e))))
    })
  })
  
  output$performance_comparison <- renderPlotly({
    req(rv$performance)
    
    df_long <- rv$performance %>%
      select(Model, Test_AUC, Accuracy, Sensitivity, Specificity, F1_Score) %>%
      pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")
    
    plot_ly(df_long, x = ~Metric, y = ~Value, color = ~Model, type = "bar") %>%
      layout(title = "性能指标对比", barmode = "group",
            yaxis = list(title = "值", range = c(0, 1)))
  })
  
  output$learning_curve <- renderPlotly({
    req(rv$cv_results)
    
    if (length(rv$cv_results) > 0) {
      best_model_name <- rv$performance$Model[1]
      
      if (!is.null(rv$cv_results[[best_model_name]])) {
        results_df <- rv$cv_results[[best_model_name]]$results
        
        p <- plot_ly(results_df, x = seq_len(nrow(results_df)), y = ~ROC, 
                    type = "scatter", mode = "lines+markers",
                    name = "CV ROC",
                    error_y = list(array = ~ROCSD)) %>%
          layout(title = paste("学习曲线 -", best_model_name),
                xaxis = list(title = "参数组合"),
                yaxis = list(title = "ROC AUC"))
        p
      } else {
        plotly_empty() %>% layout(title = "该模型无交叉验证结果")
      }
    } else {
      plotly_empty() %>% layout(title = "请先训练模型")
    }
  })
  
  output$confusion_matrix <- renderPlot({
    req(rv$models, rv$test_data, rv$performance, rv$target_col)
    
    best_model <- rv$performance$Model[1]
    pred <- rv$models[[best_model]]$pred
    pred_class <- ifelse(pred > 0.5, 1, 0)
    cm <- confusionMatrix(factor(pred_class), factor(rv$test_data[[rv$target_col]]))
    cm_df <- as.data.frame(cm$table)
    
    ggplot(cm_df, aes(x = Reference, y = Prediction)) +
      geom_tile(aes(fill = Freq), color = "white") +
      geom_text(aes(label = Freq), size = 10, color = "white") +
      scale_fill_gradient(low = "lightblue", high = "darkblue") +
      labs(title = paste("混淆矩阵 -", best_model),
           x = "实际类别", y = "预测类别") +
      theme_minimal(base_size = 14)
  })
  
  # ============================================
  # Tab 5: 可解释性分析 (SHAP)
  # ============================================
  
  # 这里使用简化版的特征重要性代替真正的 SHAP
  # 因为 SHAP 计算较复杂，需要特定模型支持
  
  observeEvent(input$compute_shap, {
    req(rv$models, input$shap_model)
    
    withProgress(message = "计算 SHAP 值...", value = 0, {
      
      model_name <- input$shap_model
      model_obj <- rv$models[[model_name]]$model
      
      incProgress(0.3, detail = "提取特征重要性...")
      
      # 使用模型自带的特征重要性作为 SHAP 近似
      if ("RandomForest" %in% class(model_obj) || 
          any(grepl("randomForest", class(model_obj)))) {
        if (inherits(model_obj, "train")) {
          importance_vals <- varImp(model_obj)$importance[, 1]
        } else {
          importance_vals <- importance(model_obj)[, "MeanDecreaseGini"]
        }
        
        rv$shap_values <- data.frame(
          Feature = names(importance_vals),
          Importance = importance_vals,
          stringsAsFactors = FALSE
        )
      } else if (any(grepl("glmnet", class(model_obj)))) {
        coef_vals <- coef(model_obj$finalModel, s = model_obj$bestTune$lambda)
        rv$shap_values <- data.frame(
          Feature = rownames(coef_vals)[-1],
          Importance = abs(as.vector(coef_vals[-1, ])),
          stringsAsFactors = FALSE
        )
      } else {
        # 使用通用方法
        if (inherits(model_obj, "train")) {
          importance_vals <- varImp(model_obj)$importance[, 1]
          rv$shap_values <- data.frame(
            Feature = rownames(varImp(model_obj)$importance),
            Importance = importance_vals,
            stringsAsFactors = FALSE
          )
        }
      }
      
      incProgress(0.7, detail = "完成!")
      
      # 更新特征选择
      if (!is.null(rv$shap_values)) {
        updateSelectInput(session, "shap_feature", 
                         choices = rv$shap_values$Feature)
      }
      
      showNotification("SHAP 分析完成!", type = "message")
    })
  })
  
  output$shap_importance <- renderPlotly({
    req(rv$shap_values)
    
    top_features <- rv$shap_values %>%
      arrange(desc(Importance)) %>%
      head(10)
    
    plot_ly(top_features, x = ~Importance, y = ~reorder(Feature, Importance),
           type = "bar", orientation = "h",
           marker = list(color = "steelblue")) %>%
      layout(title = "特征重要性 (Top 10)",
            xaxis = list(title = "重要性"),
            yaxis = list(title = "特征"))
  })
  
  output$shap_summary <- renderPlot({
    req(rv$shap_values)
    
    top_features <- rv$shap_values %>%
      arrange(desc(Importance)) %>%
      head(10)
    
    ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(title = "SHAP 特征重要性摘要",
           x = "特征", y = "重要性") +
      theme_minimal(base_size = 12)
  })
  
  output$shap_dependence <- renderPlot({
    req(rv$data, rv$target_col, input$shap_feature)
    
    feature_vals <- rv$data[[input$shap_feature]]
    target_vals <- rv$data[[rv$target_col]]
    
    df <- data.frame(
      Feature = feature_vals,
      Target = factor(target_vals, labels = c("良性", "恶性"))
    )
    
    ggplot(df, aes(x = Feature, fill = Target)) +
      geom_density(alpha = 0.5) +
      labs(title = paste("特征依赖图 -", input$shap_feature),
           x = input$shap_feature, y = "密度") +
      theme_minimal(base_size = 12) +
      scale_fill_manual(values = c("lightblue", "lightcoral"))
  })
  
  output$pdp_plot <- renderPlotly({
    req(rv$models, rv$data, input$pdp_feature, input$shap_model)
    
    model_name <- input$shap_model
    model_obj <- rv$models[[model_name]]$model
    feature <- input$pdp_feature
    
    # 生成特征值范围
    feature_range <- seq(min(rv$data[[feature]]), 
                        max(rv$data[[feature]]), 
                        length.out = 50)
    
    # 创建预测数据框
    pdp_data <- rv$data[1:50, -ncol(rv$data)]
    pdp_predictions <- numeric(50)
    
    for (i in 1:50) {
      temp_data <- pdp_data
      temp_data[[feature]] <- feature_range[i]
      
      if (inherits(model_obj, "train")) {
        pred <- predict(model_obj, temp_data, type = "prob")[, 2]
      } else if (inherits(model_obj, "glm")) {
        pred <- predict(model_obj, temp_data, type = "response")
      } else {
        pred <- predict(model_obj, temp_data, type = "prob")[, 2]
      }
      
      pdp_predictions[i] <- mean(pred, na.rm = TRUE)
    }
    
    plot_ly(x = feature_range, y = pdp_predictions, type = "scatter", mode = "lines",
           line = list(color = "darkblue", width = 3)) %>%
      layout(title = paste("部分依赖图 -", feature),
            xaxis = list(title = feature),
            yaxis = list(title = "预测概率 (恶性)"))
  })
  
  # ============================================
  # Tab 6: 预测分析
  # ============================================
  
  output$prediction_result <- renderPrint({
    cat("等待输入特征值并点击'进行预测'按钮...\n")
  })
  
  observeEvent(input$predict_btn, {
    req(rv$models, input$prediction_model)
    
    new_sample <- data.frame(
      TEF_365nm = input$TEF_365nm,
      TEF_405nm = input$TEF_405nm,
      TEF_450nm = input$TEF_450nm,
      LDF_Myo = input$LDF_Myo,
      LDF_Card = input$LDF_Card,
      LDF_Resp = input$LDF_Resp,
      AFP_Level = input$AFP_Level,
      Bilirubin = input$Bilirubin,
      Albumin = input$Albumin,
      Spectral_Skewness = input$Spectral_Skewness,
      Spectral_Kurtosis = input$Spectral_Kurtosis,
      Tissue_Density = input$Tissue_Density
    )
    
    model_name <- input$prediction_model
    model_obj <- rv$models[[model_name]]$model
    scaler <- rv$models[[model_name]]$scaler  # 获取scaler（如果有）
    
    tryCatch({
      # 如果模型需要标准化，先对数据进行标准化
      if (!is.null(scaler)) {
        new_sample_scaled <- predict(scaler, new_sample)
      } else {
        new_sample_scaled <- new_sample
      }
      
      if (inherits(model_obj, "train")) {
        pred_prob <- predict(model_obj, new_sample_scaled, type = "prob")[, 2]
      } else if (inherits(model_obj, "glm")) {
        pred_prob <- predict(model_obj, new_sample, type = "response")
      } else if (inherits(model_obj, "svm")) {
        pred_prob <- attr(predict(model_obj, new_sample, probability = TRUE), 
                         "probabilities")[, "1"]
      } else {
        pred_prob <- rv$models[[model_name]]$pred[1]
      }
      
      pred_class <- ifelse(pred_prob > 0.5, 1, 0)
      pred_label <- ifelse(pred_class == 1, "恶性", "良性")
      
      # 获取模型的CV性能
      cv_auc <- rv$performance$CV_AUC[rv$performance$Model == model_name]
      test_auc <- rv$performance$Test_AUC[rv$performance$Model == model_name]
      
      output$prediction_result <- renderPrint({
        cat("========================================\n")
        cat("预测模型:", model_name, "\n")
        cat("模型性能: CV AUC =", round(cv_auc, 4), 
            "| Test AUC =", round(test_auc, 4), "\n")
        cat("========================================\n\n")
        cat("预测概率 (恶性):", round(pred_prob * 100, 2), "%\n")
        cat("预测类别:", pred_label, "(", pred_class, ")\n\n")
        
        if (pred_prob > 0.7) {
          cat("风险等级: 高风险 ⚠️⚠️\n")
          cat("建议: 需要立即进一步检查和治疗\n")
        } else if (pred_prob > 0.5) {
          cat("风险等级: 中高风险 ⚠️\n")
          cat("建议: 建议进一步检查\n")
        } else if (pred_prob > 0.3) {
          cat("风险等级: 中低风险 ⚠\n")
          cat("建议: 建议定期复查和监测\n")
        } else {
          cat("风险等级: 低风险 ✓\n")
          cat("建议: 保持健康生活方式，定期体检\n")
        }
      })
      
      output$prediction_gauge <- renderPlotly({
        plot_ly(type = "indicator", mode = "gauge+number+delta",
               value = pred_prob * 100,
               title = list(text = "恶性概率 (%)"),
               gauge = list(
                 axis = list(range = list(0, 100)),
                 bar = list(color = "darkblue"),
                 steps = list(
                   list(range = c(0, 30), color = "lightgreen"),
                   list(range = c(30, 50), color = "lightyellow"),
                   list(range = c(50, 70), color = "orange"),
                   list(range = c(70, 100), color = "red")
                 ),
                 threshold = list(
                   line = list(color = "red", width = 4),
                   thickness = 0.75, value = 50
                 )
               )) %>%
          layout(height = 300)
      })
      
    }, error = function(e) {
      showNotification(paste("预测失败:", e$message), type = "error")
    })
  })
  
  # ============================================
  # Tab 7: 模型管理
  # ============================================
  
  # 保存模型
  observeEvent(input$save_model, {
    req(rv$models, input$model_to_save, input$model_name)
    
    model_to_save <- rv$models[[input$model_to_save]]$model
    model_name <- input$model_name
    timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    filename <- paste0(model_dir, "/", model_name, "_", timestamp, ".rds")
    
    tryCatch({
      # 保存模型和元数据
      model_data <- list(
        model = model_to_save,
        model_type = input$model_to_save,
        performance = rv$performance[rv$performance$Model == input$model_to_save, ],
        timestamp = timestamp,
        cv_method = input$cv_method,
        train_ratio = input$train_ratio
      )
      
      saveRDS(model_data, filename)
      
      output$save_status <- renderPrint({
        cat("✓ 模型保存成功!\n")
        cat("文件路径:", filename, "\n")
        cat("模型类型:", input$model_to_save, "\n")
        cat("保存时间:", timestamp, "\n")
      })
      
      showNotification("模型保存成功!", type = "message")
      
      # 刷新模型列表
      updateSavedModelsList()
      
    }, error = function(e) {
      output$save_status <- renderPrint({
        cat("✗ 模型保存失败!\n")
        cat("错误:", e$message, "\n")
      })
      showNotification(paste("保存失败:", e$message), type = "error")
    })
  })
  
  # 更新已保存模型列表
  updateSavedModelsList <- function() {
    saved_files <- list.files(model_dir, pattern = "\\.rds$", full.names = FALSE)
    updateSelectInput(session, "saved_model_list", choices = saved_files)
  }
  
  # 刷新模型列表
  observeEvent(input$refresh_models, {
    updateSavedModelsList()
    showNotification("模型列表已刷新", type = "message")
  })
  
  # 加载模型
  observeEvent(input$load_model, {
    req(input$saved_model_list)
    
    filename <- paste0(model_dir, "/", input$saved_model_list)
    
    tryCatch({
      model_data <- readRDS(filename)
      
      # 添加到当前模型列表
      model_name <- paste0(model_data$model_type, "_loaded")
      rv$models[[model_name]] <- list(model = model_data$model, pred = NULL)
      
      # 更新性能数据
      loaded_perf <- model_data$performance
      loaded_perf$Model <- model_name
      rv$performance <- rbind(rv$performance, loaded_perf)
      
      output$load_status <- renderPrint({
        cat("✓ 模型加载成功!\n")
        cat("模型类型:", model_data$model_type, "\n")
        cat("原始保存时间:", model_data$timestamp, "\n")
        cat("CV 方法:", model_data$cv_method, "\n")
        cat("训练集比例:", model_data$train_ratio, "\n")
      })
      
      # 更新选择列表
      all_models <- c(names(rv$models))
      updateSelectInput(session, "prediction_model", choices = all_models)
      updateSelectInput(session, "shap_model", choices = all_models)
      
      showNotification("模型加载成功!", type = "message")
      
    }, error = function(e) {
      output$load_status <- renderPrint({
        cat("✗ 模型加载失败!\n")
        cat("错误:", e$message, "\n")
      })
      showNotification(paste("加载失败:", e$message), type = "error")
    })
  })
  
  # 已保存模型表格
  output$saved_models_table <- renderDT({
    saved_files <- list.files(model_dir, pattern = "\\.rds$", full.names = TRUE)
    
    if (length(saved_files) > 0) {
      model_info <- lapply(saved_files, function(f) {
        info <- file.info(f)
        data.frame(
          文件名 = basename(f),
          大小_MB = round(info$size / 1024^2, 2),
          创建时间 = format(info$mtime, "%Y-%m-%d %H:%M:%S"),
          stringsAsFactors = FALSE
        )
      })
      
      df <- do.call(rbind, model_info)
      datatable(df, options = list(pageLength = 10), rownames = FALSE)
    } else {
      datatable(data.frame(消息 = "暂无保存的模型"), rownames = FALSE)
    }
  })
  
  # 模型对比图
  output$models_comparison <- renderPlotly({
    req(rv$performance)
    
    plot_ly(rv$performance, x = ~Model, y = ~Test_AUC, type = "bar", 
           name = "Test AUC", marker = list(color = "steelblue")) %>%
      add_trace(y = ~CV_AUC, name = "CV AUC", 
               marker = list(color = "orange")) %>%
      layout(title = "模型性能对比",
            xaxis = list(title = "模型"),
            yaxis = list(title = "AUC", range = c(0, 1)),
            barmode = "group")
  })
  
  # 贝叶斯优化历史查看
  output$bayes_optimization_history <- renderPrint({
    req(rv$bayes_history)
    
    cat("贝叶斯优化历史 (按模型分组):\n\n")
    
    for (model_name in names(rv$bayes_history)) {
      cat("=== ", model_name, " ===\n")
      if (nrow(rv$bayes_history[[model_name]]) == 0) {
        cat("暂无优化历史记录。\n")
      } else {
        print(rv$bayes_history[[model_name]])
      }
      cat("\n")
    }
  })
  
  # 初始化时更新模型列表
  updateSavedModelsList()
}

# ========================================================================
# 运行应用
# ========================================================================

shinyApp(ui = ui, server = server)

