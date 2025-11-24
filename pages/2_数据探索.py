# pages/2_数据探索.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import config

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

st.title("📈 数据探索")
st.markdown("---")

try:
    # 加载原始数据
    df = pd.read_csv(config.DATASET_PATH)

    # 显示数据集基本信息
    st.subheader("数据集概览")
    st.write(f"数据集包含 **{df.shape[0]}** 条记录和 **{df.shape[1]}** 个字段。")

    # 显示前几行数据
    st.subheader("数据预览")
    st.dataframe(df.head())

    # 显示数据类型和缺失值
    st.subheader("数据类型与缺失值")
    st.dataframe(df.info())

    # 显示基本统计信息
    st.subheader("数值型特征统计描述")
    st.dataframe(df.describe())

    # 简单的可视化
    st.subheader("收入分布")
    fig, ax = plt.subplots()
    ax.hist(df[config.COLUMN_MAPPING['Revenue']], bins=50, alpha=0.7, color='skyblue')
    ax.set_xlabel('收入 (元)')
    ax.set_ylabel('频次')
    ax.set_title('收入分布直方图')
    st.pyplot(fig)

except FileNotFoundError:
    st.error(f"❌ 未找到数据集文件: {config.DATASET_PATH}")
except Exception as e:
    st.error(f"加载数据时出错: {e}")