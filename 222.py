import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
from io import BytesIO

# 设置页面标题和布局
st.set_page_config(
    page_title="血液指标预测模型",
    page_icon="🩺",
    layout="wide"
)

# 标题和说明
st.title("🩺 血液指标疾病预测模型")
st.markdown("""
基于XGBoost机器学习模型的血液指标预测系统。请输入以下血液检测指标，系统将给出预测结果。
""")

# 创建侧边栏用于输入特征
st.sidebar.header("📊 输入血液检测指标")

# 定义特征名称（与训练时相同的顺序）
features = [
    'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT',
    'LYMPH%', 'MONO%', 'NEUT%', 'EO%', 'BASO%',
    'LYMPH#', 'MONO#', 'NEUT#', 'EO#', 'BASO#',
    'RDW-CV', 'PDW', 'MPV', 'PCT', 'P-LCR', 'CRP'
]

# 为每个特征创建输入框
feature_values = {}

# 将特征分组显示，提高用户体验
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    st.subheader("基础血细胞计数")
    feature_values['WBC'] = st.number_input('WBC (10^9/L)', min_value=0.0, max_value=100.0, value=6.5, step=0.1)
    feature_values['RBC'] = st.number_input('RBC (10^12/L)', min_value=0.0, max_value=10.0, value=4.5, step=0.1)
    feature_values['HGB'] = st.number_input('HGB (g/L)', min_value=0.0, max_value=200.0, value=130.0, step=1.0)
    feature_values['HCT'] = st.number_input('HCT (%)', min_value=0.0, max_value=100.0, value=40.0, step=0.1)
    feature_values['PLT'] = st.number_input('PLT (10^9/L)', min_value=0.0, max_value=1000.0, value=250.0, step=1.0)

with col2:
    st.subheader("红细胞指标")
    feature_values['MCV'] = st.number_input('MCV (fL)', min_value=0.0, max_value=150.0, value=90.0, step=0.1)
    feature_values['MCH'] = st.number_input('MCH (pg)', min_value=0.0, max_value=50.0, value=30.0, step=0.1)
    feature_values['MCHC'] = st.number_input('MCHC (g/L)', min_value=0.0, max_value=400.0, value=330.0, step=1.0)
    feature_values['RDW-CV'] = st.number_input('RDW-CV (%)', min_value=0.0, max_value=30.0, value=13.0, step=0.1)

    st.subheader("血小板指标")
    feature_values['PDW'] = st.number_input('PDW (fL)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    feature_values['MPV'] = st.number_input('MPV (fL)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    feature_values['PCT'] = st.number_input('PCT (%)', min_value=0.0, max_value=10.0, value=0.2, step=0.01)
    feature_values['P-LCR'] = st.number_input('P-LCR (%)', min_value=0.0, max_value=100.0, value=30.0, step=0.1)

with col3:
    st.subheader("白细胞分类计数(%)")
    feature_values['LYMPH%'] = st.number_input('LYMPH% (%)', min_value=0.0, max_value=100.0, value=35.0, step=0.1)
    feature_values['MONO%'] = st.number_input('MONO% (%)', min_value=0.0, max_value=100.0, value=8.0, step=0.1)
    feature_values['NEUT%'] = st.number_input('NEUT% (%)', min_value=0.0, max_value=100.0, value=55.0, step=0.1)
    feature_values['EO%'] = st.number_input('EO% (%)', min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    feature_values['BASO%'] = st.number_input('BASO% (%)', min_value=0.0, max_value=100.0, value=0.5, step=0.1)

    st.subheader("白细胞分类计数(#)")
    feature_values['LYMPH#'] = st.number_input('LYMPH# (10^9/L)', min_value=0.0, max_value=20.0, value=2.0, step=0.1)
    feature_values['MONO#'] = st.number_input('MONO# (10^9/L)', min_value=0.0, max_value=10.0, value=0.5, step=0.1)
    feature_values['NEUT#'] = st.number_input('NEUT# (10^9/L)', min_value=0.0, max_value=20.0, value=3.5, step=0.1)
    feature_values['EO#'] = st.number_input('EO# (10^9/L)', min_value=0.0, max_value=5.0, value=0.1, step=0.01)
    feature_values['BASO#'] = st.number_input('BASO# (10^9/L)', min_value=0.0, max_value=5.0, value=0.03, step=0.01)

    feature_values['CRP'] = st.number_input('CRP (mg/L)', min_value=0.0, max_value=200.0, value=5.0, step=0.1)


# 加载模型函数
@st.cache_resource
def load_model():
    """
    加载训练好的XGBoost模型
    注意：你需要先用joblib或pickle保存你的模型
    """
    try:
        # 方法1：如果模型保存为joblib文件
        model = joblib.load('XGB.pkl')
    except:
        try:
            # 方法2：如果模型保存为pkl文件
            model = pickle.load(open('XGB.pkl', 'rb'))
        except:
            # 方法3：重新创建模型（临时方案，实际使用时请替换为你的模型文件）
            st.warning("⚠️ 未找到模型文件，使用默认参数创建新模型")
            model = xgb.XGBClassifier(
                max_depth=9, learning_rate=0.1, eta=0.98, gamma=0.4,
                subsample=0.9, colsample_bytree=0.7
            )
    return model


# 加载模型
model = load_model()


# 预测函数
def predict_disease(input_data):
    """
    使用模型进行预测
    """
    try:
        # 转换为DataFrame，确保特征顺序正确
        input_df = pd.DataFrame([input_data], columns=features)

        # 预测概率
        prediction_proba = model.predict_proba(input_df)[0]

        # 预测类别
        prediction = model.predict(input_df)[0]

        return prediction, prediction_proba
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")
        return None, None


# 主内容区域
st.header("🔍 预测结果")

# 预测按钮
if st.sidebar.button("开始预测", type="primary"):
    # 显示加载指示器
    with st.spinner('正在进行分析预测...'):
        # 准备输入数据
        input_data = [feature_values[feature] for feature in features]

        # 进行预测
        prediction, prediction_proba = predict_disease(input_data)

        if prediction is not None:
            # 显示预测结果
            st.success("预测完成！")

            # 创建结果展示列
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("预测结果")
                # 根据你的模型定义，这里假设1表示阳性，0表示阴性
                if prediction == 1:
                    st.error(f"🔴 **高风险** (阳性)")
                else:
                    st.success(f"🟢 **低风险** (阴性)")

                # 显示概率
                st.subheader("预测概率")
                st.metric(
                    label="阳性概率",
                    value=f"{prediction_proba[1]:.3f}",
                    delta=f"{(prediction_proba[1] - 0.5) * 100:+.1f}%" if prediction_proba[1] > 0.5 else ""
                )

                # 概率进度条
                st.progress(float(prediction_proba[1]))
                st.caption(f"阴性概率: {prediction_proba[0]:.3f} | 阳性概率: {prediction_proba[1]:.3f}")

            with col2:
                st.subheader("概率分布")
                # 简单的概率条形图
                prob_df = pd.DataFrame({
                    '类别': ['阴性', '阳性'],
                    '概率': [prediction_proba[0], prediction_proba[1]]
                })
                st.bar_chart(prob_df.set_index('类别'))

            # 显示输入数据摘要
            st.subheader("📋 输入数据摘要")
            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.write("**基础指标**")
                st.write(f"WBC: {feature_values['WBC']}")
                st.write(f"RBC: {feature_values['RBC']}")
                st.write(f"HGB: {feature_values['HGB']}")
                st.write(f"HCT: {feature_values['HCT']}")
                st.write(f"PLT: {feature_values['PLT']}")

            with summary_col2:
                st.write("**红细胞指标**")
                st.write(f"MCV: {feature_values['MCV']}")
                st.write(f"MCH: {feature_values['MCH']}")
                st.write(f"MCHC: {feature_values['MCHC']}")
                st.write(f"RDW-CV: {feature_values['RDW-CV']}")

            with summary_col3:
                st.write("**炎症指标**")
                st.write(f"CRP: {feature_values['CRP']}")
                st.write(f"NEUT%: {feature_values['NEUT%']}")
                st.write(f"LYMPH%: {feature_values['LYMPH%']}")

# 添加说明和信息部分
st.sidebar.markdown("---")
st.sidebar.info("""
**使用说明：**
1. 在左侧输入所有血液检测指标
2. 点击"开始预测"按钮
3. 查看右侧的预测结果和概率

**注意：** 预测结果仅供参考，不能替代专业医疗诊断。
""")

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "基于XGBoost的血液指标预测模型 • 仅供研究使用"
    "</div>",
    unsafe_allow_html=True
)

# 如果模型文件不存在，显示警告和保存模型的代码
try:
    joblib.load('xgb_model.joblib')
except:
    st.warning("""
    ⚠️ **模型文件未找到**

    要使用此应用，你需要先保存训练好的XGBoost模型。在你的训练代码中添加：

    ```python
    import joblib
    # 在训练完成后保存模型
    joblib.dump(model, 'xgb_model.joblib')
    ```

    或者使用pickle：
    ```python
    import pickle
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    ```

    """)
