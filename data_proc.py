import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import config


class DataProcessor:
    def __init__(self, file_path=config.DATASET_PATH):
        self.file_path = file_path
        self.df = None
        self.le_cat = LabelEncoder()
        self.le_subcat = LabelEncoder()
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.feature_cols = None
        self.selected_feature_names = None

    def load_and_clean_data(self):
        """加载并清洗数据集"""
        try:
            # 加载数据集
            self.df = pd.read_csv(self.file_path, usecols=config.FEATURE_COLUMNS)
            print(f"成功加载数据集：{len(self.df)}条记录，{len(self.df.columns)}个字段")

            # 重命名列以匹配模型预期
            self.df.rename(columns=config.COLUMN_MAPPING, inplace=True)
            print("列名映射完成：", config.COLUMN_MAPPING)

            # 处理日期列
            if "date" in self.df.columns:
                self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
                initial_count = len(self.df)
                self.df.dropna(subset=["date"], inplace=True)
                print(f"删除日期为空的记录：{initial_count - len(self.df)}条")
            else:
                raise ValueError("数据集缺少日期列，请检查配置文件中的COLUMN_MAPPING")

            # 处理目标变量（收入）
            if "sales_volume" not in self.df.columns:
                raise ValueError("目标变量列'sales_volume'未找到，请检查COLUMN_MAPPING")

            # 修正负收入
            if (self.df["sales_volume"] < 0).any():
                negative_count = (self.df["sales_volume"] < 0).sum()
                self.df.loc[self.df["sales_volume"] < 0, "sales_volume"] = 0
                print(f"修正负收入为0：{negative_count}条")

            # 处理关键特征缺失值
            required_features = ["product_category", "product_subcategory", "Unit_Price", "Quantity"]
            for col in required_features:
                if col not in self.df.columns:
                    raise ValueError(f"数据集缺少关键特征列：{col}")

            # 数值型特征用中位数填充
            num_imputer = SimpleImputer(strategy="median")
            self.df[["Unit_Price", "Quantity", "sales_volume", "Profit"]] = num_imputer.fit_transform(
                self.df[["Unit_Price", "Quantity", "sales_volume", "Profit"]]
            )

            # 类别型特征用众数填充
            cat_imputer = SimpleImputer(strategy="most_frequent")
            self.df[["product_category", "product_subcategory", "City", "State", "Region",
                     "Country"]] = cat_imputer.fit_transform(
                self.df[["product_category", "product_subcategory", "City", "State", "Region", "Country"]]
            )

            print("数据清洗完成")
            return self

        except FileNotFoundError:
            raise FileNotFoundError(f"未找到文件：{self.file_path}，请确认路径正确")

    def process_time_features(self):
        """提取时间特征（年/月/日/星期/周末/季节）"""
        print("处理时间特征...")
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day
        self.df["weekday"] = self.df["date"].dt.weekday  # 0=周一，6=周日
        self.df["is_weekend"] = np.where(self.df["weekday"].isin([5, 6]), 1, 0)
        self.df["season"] = pd.cut(
            self.df["month"], bins=[0, 3, 6, 9, 12], labels=[1, 2, 3, 4], ordered=False
        ).astype(int)
        print("时间特征处理完成")
        return self

    def process_category_features(self):
        """编码商品类别和其他类别特征"""
        print("处理类别特征...")
        # 编码商品大类和子类
        self.df["cat_encoded"] = self.le_cat.fit_transform(self.df["product_category"])
        self.df["subcat_encoded"] = self.le_subcat.fit_transform(self.df["product_subcategory"])

        # 编码其他类别特征（如果需要）
        for col in ["City", "State", "Region", "Country"]:
            le = LabelEncoder()
            self.df[f"{col}_encoded"] = le.fit_transform(self.df[col])

        print("类别特征处理完成")
        return self

    def process_promotion_features(self):
        """处理促销特征（根据你的数据集调整）"""
        print("处理促销特征...")
        # 假设Unit_Price < 某个阈值为促销商品（可根据实际业务逻辑调整）
        self.df["discount_rate"] = np.where(self.df["Unit_Price"] < 50, 0.2, 0.0)  # 示例：单价<50为8折
        self.df["is_promotion"] = np.where(self.df["discount_rate"] > 0, 1, 0)
        self.df["promotion_type"] = np.where(self.df["is_promotion"] == 1, "折扣", "无")
        self.df["is_big_promo"] = np.where(self.df["discount_rate"] >= 0.3, 1, 0)  # 折扣>=30%为大促

        # 交互特征
        self.df["price_discount"] = self.df["Unit_Price"] * self.df["discount_rate"]
        self.df["price_promo"] = self.df["Unit_Price"] * self.df["is_promotion"]
        self.df["quantity_promo"] = self.df["Quantity"] * self.df["is_promotion"]
        self.df["big_promo_type"] = self.df["is_big_promo"] * self.df["is_promotion"]

        print("促销特征处理完成")
        return self

    def split_data(self):
        """时间序列分割训练集和测试集"""
        print("分割训练集和测试集...")
        # 定义需要排除的列
        exclude_cols = [
            "Order_ID", "Customer_Name", "Product_Name", "date",
            "product_category", "product_subcategory", "City", "State", "Region", "Country",
            "sales_volume"  # 目标变量
        ]

        # 筛选特征列
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        X = self.df[self.feature_cols]
        y = self.df["sales_volume"]

        # 按时间排序后分割
        self.df.sort_values("date", inplace=True)
        train_size = int(config.TRAIN_SIZE_RATIO * len(self.df))
        self.X_train, self.X_test = X.iloc[:train_size], X.iloc[train_size:]
        self.y_train, self.y_test = y.iloc[:train_size], y.iloc[train_size:]

        print(f"训练集：{self.X_train.shape}，测试集：{self.X_test.shape}")
        print(f"特征列：{self.feature_cols}")
        return self

    def run_full_pipeline(self):
        """执行完整的数据预处理流程"""
        self.load_and_clean_data()
        self.process_time_features()
        self.process_category_features()
        self.process_promotion_features()
        self.split_data()
        return self