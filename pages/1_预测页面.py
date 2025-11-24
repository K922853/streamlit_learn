# pages/1_预测页面.py

import streamlit as st
import pandas as pd
import numpy as np
from model_utils import ModelUtils
import config

# 页面标题
st.title("🚀 商品收入预测")
st.markdown("---")

# 检查模型是否存在
try:
    artifacts = ModelUtils.load_model_artifacts(config.MODEL_DIR)
    model = artifacts["model"]
    feature_cols = artifacts["feature_cols"]
    le_cat = artifacts["le_cat"]
    le_subcat = artifacts["le_subcat"]
except FileNotFoundError:
    st.error("❌ 模型文件未找到！请先运行项目根目录下的 `main.py` 脚本进行训练。")
    st.stop()

# 侧边栏用户输入
st.sidebar.header("⚙️ 输入参数")

# 1. 商品分类信息
st.sidebar.subheader("1. 商品分类")
product_cat = st.sidebar.selectbox("商品大类", options=le_cat.classes_, index=0)
product_subcat = st.sidebar.selectbox("商品子类", options=le_subcat.classes_, index=0)

# 2. 商品基本属性
st.sidebar.subheader("2. 商品属性")
quantity = st.sidebar.slider("销售数量", min_value=1, max_value=100, value=10, step=1)
unit_price = st.sidebar.slider("商品单价 (元)", min_value=1.0, max_value=1000.0, value=99.0, step=1.0)

# 3. 促销信息
st.sidebar.subheader("3. 促销策略")
discount_rate = st.sidebar.slider("折扣率", min_value=0.0, max_value=0.9, value=0.1, step=0.05)
is_promotion = 1 if discount_rate > 0 else 0
promotion_type = "折扣" if is_promotion else "无"
is_big_promo = 1 if discount_rate >= 0.3 else 0

# 4. 时间特征
st.sidebar.subheader("4. 时间特征")
month = st.sidebar.slider("月份", min_value=1, max_value=12, value=6, step=1)
weekday = st.sidebar.slider("星期几", min_value=0, max_value=6, value=2, step=1,
                            format_func=lambda x: ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][x])

st.sidebar.markdown("---")


# 主界面：预测逻辑和结果展示
def prepare_input_data():
    is_weekend = 1 if weekday in [5, 6] else 0
    if month in [12, 1, 2]:
        season = 1
    elif month in [3, 4, 5]:
        season = 2
    elif month in [6, 7, 8]:
        season = 3
    else:
        season = 4

    input_dict = {
        "cat_encoded": [le_cat.transform([product_cat])[0]],
        "subcat_encoded": [le_subcat.transform([product_subcat])[0]],
        "Quantity": [quantity],
        "Unit_Price": [unit_price],
        "discount_rate": [discount_rate],
        "is_promotion": [is_promotion],
        "month": [month],
        "weekday": [weekday],
        "is_weekend": [is_weekend],
        "season": [season],
        "is_big_promo": [is_big_promo],
        "price_discount": [unit_price * discount_rate],
        "price_promo": [unit_price * is_promotion],
        "quantity_promo": [quantity * is_promotion],
        "big_promo_type": [is_big_promo * is_promotion]
    }

    # 处理其他可能的编码特征
    for col in feature_cols:
        if col not in input_dict and any(kw in col.lower() for kw in ["city", "state", "region", "country"]):
            input_dict[col] = [0]

    input_df = pd.DataFrame(input_dict)
    return input_df[feature_cols]


# 预测按钮
if st.sidebar.button("开始预测"):
    with st.spinner("正在计算预测结果..."):
        input_data = prepare_input_data()
        prediction = model.predict(input_data)[0]

        st.success("✅ 预测完成！")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 预测结果")
            st.metric(label="预测收入 (元)", value=f"{np.round(prediction, 2):.2f}")
        with col2:
            st.subheader("🔍 输入参数摘要")
            st.write(f"**商品大类:** {product_cat}")
            st.write(f"**商品子类:** {product_subcat}")
            st.write(f"**销售数量:** {quantity}")
            st.write(f"**商品单价:** ¥{unit_price}")
            st.write(f"**折扣率:** {discount_rate:.0%}")