# model_train.py

import numpy as np
import time
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import config


class ModelTrainer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.X_train = data_processor.X_train
        self.X_test = data_processor.X_test
        self.y_train = data_processor.y_train
        self.y_test = data_processor.y_test
        self.feature_cols = data_processor.feature_cols

        self.model = None
        self.best_model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.train_metrics = None
        self.test_metrics = None
        self.selected_feature_names = None

    def _get_preprocessor(self, feature_cols):
        """创建预处理管道（私有方法）"""
        # 筛选数值型和类别型特征
        numeric_cols = [col for col in feature_cols if any(kw in col.lower() for kw in
                                                           ["price", "quantity", "year", "month", "day", "weekday",
                                                            "discount", "profit", "encoded"])]
        categorical_cols = [col for col in feature_cols if col.lower() == "promotion_type"]

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
            ],
            remainder="drop"
        )

    def perform_feature_selection(self):
        """使用Pipeline和RFE进行特征选择"""
        print("开始特征选择...")

        # 1. 创建预处理管道
        preprocessor = self._get_preprocessor(self.feature_cols)

        # 2. 创建一个基础模型（例如，随机森林）
        base_model = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_SEED)

        # 3. 将预处理和模型组合成一个Pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", base_model)
        ])

        # 4. 使用Pipeline作为RFE的estimator
        n_features = min(config.FEATURE_SELECTION["n_features_to_select"], len(self.feature_cols))
        rfe = RFE(estimator=pipeline, n_features_to_select=n_features, step=1)
        rfe.fit(self.X_train, self.y_train)

        # 5. 获取被选中的特征名称
        self.selected_feature_names = [f for i, f in enumerate(self.feature_cols) if rfe.support_[i]]

        print(f"特征选择完成：从{len(self.feature_cols)}个特征中选出{len(self.selected_feature_names)}个")
        print(f"选中特征：{self.selected_feature_names}")
        return self

    def build_and_tune_model(self):
        """构建并调优Stacking模型"""
        print("构建并调优Stacking模型...")
        if self.selected_feature_names is None:
            self.perform_feature_selection()

        # 只使用被选中的特征
        X_train_selected = self.X_train[self.selected_feature_names]

        # 创建预处理管道
        preprocessor = self._get_preprocessor(self.selected_feature_names)

        # 定义基础模型
        base_models = [
            ("ridge", Pipeline([("preprocessor", preprocessor), ("model", Ridge(random_state=config.RANDOM_SEED))])),
            ("rf", Pipeline(
                [("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=config.RANDOM_SEED))])),
            ("gbt", Pipeline([("preprocessor", preprocessor),
                              ("model", GradientBoostingRegressor(random_state=config.RANDOM_SEED))]))
        ]

        # 定义元模型
        meta_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=config.RANDOM_SEED))])

        # 定义Stacking模型
        from sklearn.ensemble import StackingRegressor
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=KFold(n_splits=3, shuffle=True, random_state=config.RANDOM_SEED),
            n_jobs=-1
        )

        # 网格搜索调优
        grid_search = GridSearchCV(
            estimator=stacking_model,
            param_grid=config.STACKING_PARAM_GRID,
            cv=KFold(n_splits=2, shuffle=True, random_state=config.RANDOM_SEED),
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )

        start_time = time.time()
        grid_search.fit(X_train_selected, self.y_train)
        end_time = time.time()

        self.best_model = grid_search.best_estimator_

        print(f"模型调优完成，耗时：{end_time - start_time:.2f}秒")
        print(f"最佳参数：{grid_search.best_params_}")
        print(f"最佳交叉验证分数（负MSE）：{grid_search.best_score_:.4f}")
        return self

    def evaluate(self):
        """评估模型性能"""
        print("\n开始预测与评估...")
        if self.best_model is None:
            self.build_and_tune_model()

        X_test_selected = self.X_test[self.selected_feature_names]

        self.y_train_pred = self.best_model.predict(self.X_train[self.selected_feature_names])
        self.y_test_pred = self.best_model.predict(X_test_selected)

        def _calculate_metrics(y_true, y_pred, set_name):
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            adj_r2 = 1 - ((1 - r2) * (len(y_true) - 1)) / (len(y_true) - len(self.selected_feature_names) - 1)

            y_true_non_zero = y_true.replace(0, np.nan)
            y_pred_non_zero = y_pred[y_true != 0]
            y_true_non_zero = y_true_non_zero.dropna()
            mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100 if len(
                y_true_non_zero) > 0 else np.nan

            print(f"\n{set_name}集性能：")
            print(f"MSE：{mse:.4f} | RMSE：{rmse:.4f} | MAE：{mae:.4f}")
            print(f"R²：{r2:.4f} | 调整后R²：{adj_r2:.4f} | MAPE：{mape:.2f}%")
            return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "adj_r2": adj_r2, "mape": mape}

        self.train_metrics = _calculate_metrics(self.y_train, self.y_train_pred, "训练")
        self.test_metrics = _calculate_metrics(self.y_test, self.y_test_pred, "测试")

        return self