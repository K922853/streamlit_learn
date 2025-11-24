#"D:\pycharm\机器学习应用开发\pythonProject2\product_sales_dataset_final.csv"
import pandas as pd

# 1. 读取 CSV 文件
df = pd.read_csv("D:\pycharm\机器学习应用开发\pythonProject2\product_sales_dataset_final.csv")

# 2. 打印数据集的基本信息
print("数据集形状:", df.shape)
print("\n前5行数据:")
print(df.head())

# 3. 检查是否存在 '无' 值
print("\n检查各列是否包含 '无':")
for col in df.columns:
    if df[col].dtype == "object":  # 只检查字符串类型的列
        has_none = df[col].str.contains("无", na=False).any()
        if has_none:
            count = df[col].str.contains("无", na=False).sum()
            print(f"列 '{col}' 中包含 {count} 个 '无' 值")
        else:
            print(f"列 '{col}' 中不包含 '无' 值")