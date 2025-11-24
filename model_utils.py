import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import config

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False


class ModelUtils:
    @staticmethod
    def save_model(trainer, data_processor, save_dir=config.MODEL_DIR):
        """保存模型及相关组件"""
        os.makedirs(save_dir, exist_ok=True)

        artifacts = {
            "model": trainer.best_model,
            "feature_cols": trainer.selected_feature_names,
            "le_cat": data_processor.le_cat,
            "le_subcat": data_processor.le_subcat,
            "train_metrics": trainer.train_metrics,
            "test_metrics": trainer.test_metrics,
            "column_mapping": config.COLUMN_MAPPING
        }

        joblib.dump(artifacts, os.path.join(save_dir, "model_artifacts.pkl"))

        print(f"\n模型 artifacts 已保存至：{os.path.abspath(save_dir)}/model_artifacts.pkl")

    @staticmethod
    def load_model_artifacts(load_dir=config.MODEL_DIR):
        """加载模型及相关组件"""
        artifact_path = os.path.join(load_dir, "model_artifacts.pkl")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"未找到模型文件：{artifact_path}")

        return joblib.load(artifact_path)

    @staticmethod
    def plot_residuals(y_true, y_pred, set_name, save_dir=config.PLOT_DIR):
        """绘制残差图"""
        os.makedirs(save_dir, exist_ok=True)
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5, color="blue")
        plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
        plt.xlabel("预测收入")
        plt.ylabel("残差（实际-预测）")
        plt.title(f"{set_name}集残差图")
        plt.grid(alpha=0.3)
        save_path = os.path.join(save_dir, f"{set_name.lower()}_residuals.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"残差图已保存至：{save_path}")

    @staticmethod
    def plot_pred_vs_actual(y_true, y_pred, set_name, save_dir=config.PLOT_DIR):
        """绘制预测值vs实际值图"""
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, color="green")
        min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="理想线（y=x）")
        plt.xlabel("实际收入")
        plt.ylabel("预测收入")
        plt.title(f"{set_name}集：预测收入vs实际收入")
        plt.legend()
        plt.grid(alpha=0.3)
        save_path = os.path.join(save_dir, f"{set_name.lower()}_pred_actual.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"预测vs实际图已保存至：{save_path}")

    @staticmethod
    def plot_learning_curve(model, X_train, y_train, X_test, y_test, save_dir=config.PLOT_DIR):
        """绘制学习曲线"""
        os.makedirs(save_dir, exist_ok=True)
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_rmse, test_rmse = [], []

        for size in train_sizes:
            subset_size = int(size * len(X_train))
            X_sub, y_sub = X_train.iloc[:subset_size], y_train.iloc[:subset_size]
            model.fit(X_sub, y_sub)
            train_rmse.append(np.sqrt(mean_squared_error(y_sub, model.predict(X_sub))))
            test_rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes * 100, train_rmse, "o-", color="blue", label="训练集RMSE")
        plt.plot(train_sizes * 100, test_rmse, "o-", color="red", label="测试集RMSE")
        plt.xlabel("训练数据量占比（%）")
        plt.ylabel("RMSE")
        plt.title("模型学习曲线")
        plt.legend()
        plt.grid(alpha=0.3)
        save_path = os.path.join(save_dir, "learning_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"学习曲线已保存至：{save_path}")

    @staticmethod
    def run_full_visualization(trainer, plot_learning_curve=True):
        """执行完整的可视化流程"""
        ModelUtils.plot_residuals(trainer.y_train, trainer.y_train_pred, "训练")
        ModelUtils.plot_residuals(trainer.y_test, trainer.y_test_pred, "测试")
        ModelUtils.plot_pred_vs_actual(trainer.y_train, trainer.y_train_pred, "训练")
        ModelUtils.plot_pred_vs_actual(trainer.y_test, trainer.y_test_pred, "测试")

        if plot_learning_curve and trainer.best_model is not None:
            X_train_selected = trainer.X_train[trainer.selected_feature_names]
            X_test_selected = trainer.X_test[trainer.selected_feature_names]
            ModelUtils.plot_learning_curve(trainer.best_model, X_train_selected, trainer.y_train, X_test_selected,
                                           trainer.y_test)
        print("可视化完成")