# config.py

# 数据集路径
DATASET_PATH = "D:\pycharm\机器学习应用开发\pythonProject2\product_sales_dataset_final.csv"
# 模型与图表保存目录
MODEL_DIR = "models"
PLOT_DIR = "plots"

# 时间序列分割比例
TRAIN_SIZE_RATIO = 0.8

# 随机种子（保证结果可复现）
RANDOM_SEED = 42

# 特征选择配置（RFE递归特征消除）
FEATURE_SELECTION = {
    "method": "rfe",
    "n_features_to_select": 15  # 选择Top15重要特征
}

# Stacking模型参数网格（网格搜索用）
STACKING_PARAM_GRID = {
    "final_estimator__alpha": [0.1, 1.0, 10.0],  # 元模型Ridge正则化系数
    "rf__model__n_estimators": [50, 100],         # 随机森林树数量
    "gbt__model__n_estimators": [50, 100]        # 梯度提升树数量
}

# 数据集列名映射（根据你的实际列名修改）
# 注意：这里的键（左边）必须和CSV文件中的列名完全一致，包括两边的空格！
COLUMN_MAPPING = {
    "Order_Date": "date",
    " Revenue ": "sales_volume",
    "Category": "product_category",
    "Sub_Category": "product_subcategory",
    " Unit_Price ": "Unit_Price",  # <--- 新增这一行
    " Profit ": "Profit"           # <--- 建议也添加这一行，以防后续用到
}

# 需要保留的特征列（根据你的数据集调整）
# 这里使用CSV文件中的原始列名（带空格）
FEATURE_COLUMNS = [
    "Order_ID", "Order_Date", "Customer_Name", "City", "State", "Region", "Country",
    "Category", "Sub_Category", "Product_Name", "Quantity", " Unit_Price ",
    " Revenue ", " Profit "
]