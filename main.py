import os
import sys
from data_proc import DataProcessor
from model_train import ModelTrainer
from model_utils import ModelUtils
import config


def main():
    try:
        # 创建必要目录
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.PLOT_DIR, exist_ok=True)

        print("=== 电商平台商品销量预测系统 (OOP版) ===")

        # 1. 数据预处理
        print("\n=== 步骤1：数据预处理 ===")
        data_processor = DataProcessor()
        data_processor.run_full_pipeline()

        # 2. 模型训练与评估
        print("\n=== 步骤2：模型训练与评估 ===")
        model_trainer = ModelTrainer(data_processor)
        model_trainer.build_and_tune_model().evaluate()

        # 3. 模型保存
        print("\n=== 步骤3：模型保存 ===")
        ModelUtils.save_model(model_trainer, data_processor)

        # 4. 性能可视化
        print("\n=== 步骤4：性能可视化 ===")
        if input("\n是否绘制学习曲线？(y/n, 默认n): ").lower() == 'y':
            ModelUtils.run_full_visualization(model_trainer, plot_learning_curve=True)
        else:
            ModelUtils.run_full_visualization(model_trainer, plot_learning_curve=False)

        print("\n=== 全流程执行完成！===")
        print(f"模型保存在: {os.path.abspath(config.MODEL_DIR)}")
        print(f"图表保存在: {os.path.abspath(config.PLOT_DIR)}")

    except Exception as e:
        print(f"\n[错误] 执行失败：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
















