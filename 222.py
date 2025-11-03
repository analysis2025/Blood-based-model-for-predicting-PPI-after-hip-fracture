import streamlit as st
import numpy as np
import joblib

# 加载模型
model = load('XGB.pkl')  # 请确保路径正确，且模型文件名为your_model.joblib

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

# 定义特征名称（与训练时相同的顺序）
features = [
    'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT',
    'LYMPH%', 'MONO%', 'NEUT%', 'EO%', 'BASO%',
    'LYMPH#', 'MONO#', 'NEUT#', 'EO#', 'BASO#',
    'RDW-CV', 'PDW', 'MPV', 'PCT', 'P-LCR', 'CRP'
]

# 定义预测结果的标签
result_labels = {0: 'normal', 1: 'RB'}


def predict_and_show_results(input_data):
    # 使用模型进行预测
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    # 提取预测结果的类别标签
    predicted_class = result_labels[predictions[0]]

    # 提取每个类别的概率
    class_probabilities = {result_labels[label]: prob for label, prob in enumerate(probabilities[0])}

    return predicted_class, class_probabilities


def main():
    st.title('RF model screening for retinoblastoma (RB)')
    st.write('Please enter the following indicators to predict:')

    # 创建侧边栏用于输入特征
    st.sidebar.header("📊 输入血液检测指标")

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
        feature_values['LYMPH#'] = st.number_input('LYMPH# (10^9/L)', min_value=0.0, max_value=20.0, value=2.0,
                                                   step=0.1)
        feature_values['MONO#'] = st.number_input('MONO# (10^9/L)', min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        feature_values['NEUT#'] = st.number_input('NEUT# (10^9/L)', min_value=0.0, max_value=20.0, value=3.5, step=0.1)
        feature_values['EO#'] = st.number_input('EO# (10^9/L)', min_value=0.0, max_value=5.0, value=0.1, step=0.01)
        feature_values['BASO#'] = st.number_input('BASO# (10^9/L)', min_value=0.0, max_value=5.0, value=0.03, step=0.01)

        feature_values['CRP'] = st.number_input('CRP (mg/L)', min_value=0.0, max_value=200.0, value=5.0, step=0.1)

    # 创建一个按钮
    if st.button('Screening'):
        # 检查是否所有输入都已填写
        if all(value is not None for value in feature_values.values()):
            # 按照features列表的顺序准备输入数据
            input_array = np.array([[feature_values[feature] for feature in features]]).astype(float)

            # 进行预测
            predicted_class, class_probabilities = predict_and_show_results(input_array)

            # 显示预测结果
            st.success(f'Classification results of screening: {predicted_class}')

            # 显示每个类别的概率
            st.write('Probability of screening:')
            for class_name, prob in class_probabilities.items():
                st.write(f'{class_name}: {prob:.4f} ({prob * 100:.2f}%)')
        else:
            st.warning('Please fill in all feature values!')


if __name__ == '__main__':
    main()


