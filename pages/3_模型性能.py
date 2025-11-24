# pages/3_模型性能.py

import streamlit as st
import matplotlib.pyplot as plt
import os
import config
from model_utils import ModelUtils

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

st.title("📊 模型性能")
st.markdown("---")

# 检查模型指标是否存在
try:
    artifacts = ModelUtils.load_model_artifacts(config.MODEL_DIR)
    test_metrics = artifacts["test_metrics"]
except FileNotFoundError:
    st.error("❌ 模型文件未找到！请先运行项目根目录下的 `main.py` 脚本进行训练。")
    st.stop()

# 显示性能指标
st.subheader("模型评估指标 (测试集)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("RMSE (均方根误差)", f"{test_metrics['rmse']:.2f}")
with col2:
    st.metric("MAE (平均绝对误差)", f"{test_metrics['mae']:.2f}")
with col3:
    st.metric("R² (决定系数)", f"{test_metrics['r2']:.4f}")
with col4:
    st.metric("MAPE (平均绝对百分比误差)", f"{test_metrics['mape']:.2f}%")

st.markdown("---")

# 显示性能图表
st.subheader("模型性能可视化")

# 检查图表文件是否存在
plot_files = {
    "训练集预测 vs 实际": "train_pred_actual.png",
    "测试集预测 vs 实际": "test_pred_actual.png",
    "训练集残差图": "train_residuals.png",
    "测试集残差图": "test_residuals.png",
    "学习曲线": "learning_curve.png"
}

for title, filename in plot_files.items():
    filepath = os.path.join(config.PLOT_DIR, filename)
    if os.path.exists(filepath):
        st.subheader(title)
        st.image(filepath, use_column_width=True)
    else:
        st.warning(f"未找到图表: {title} ({filename})")