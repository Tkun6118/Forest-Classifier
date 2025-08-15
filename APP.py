# Author：Tkun
# createTime：2025/8/15
# FileName：APP
# Description：simple introduction of the code
# app.py
# 导入必要的库
#pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
import shap
from io import BytesIO

# --- 页面基础设置 ---
st.set_page_config(
    page_title="多模型智能林型分类器",
    page_icon="🌳",
    layout="wide"
)

# --- 环境与绘图设置 ---
# Streamlit 会处理 matplotlib 的后端，通常不需要手动设置 'Agg'
# matplotlib.use('Agg')
# --- 环境与绘图设置 ---
try:
    # 步骤1：尝试执行这里的代码
    # 我们乐观地认为系统中有 'Microsoft YaHei' 字体
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    print("字体成功设置为 'Microsoft YaHei'") # 这行是可选的，用于调试
except Exception as e:
    # 步骤2：如果 'try' 块中的代码执行失败（抛出异常），则立即跳转到这里执行
    # 这种情况通常发生在 Linux 服务器等没有预装 Windows 字体的地方
    print(f"警告：未找到 'Microsoft YaHei' 字体，错误信息: {e}。正在回退到备用字体...")
    # 我们使用一个在 Linux 系统上更常见的备用中文字体'SimHei'（黑体）
    # 'sans-serif'是最后的备选项，它会使用系统默认的无衬线字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'sans-serif']

# 这一行代码无论字体设置成功与否都应该执行，所以放在 try-except 块的外面
# 它的作用是确保图表中的负号'-'能正常显示
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 核心功能函数 ---

# 使用 Streamlit 缓存来加载和预处理数据，避免重复读取
@st.cache_data
def load_and_process_data(uploaded_file):
    """加载数据、划分数据集并进行标准化"""
    df = pd.read_excel(uploaded_file)

    y = df.iloc[:, 1]
    x_df = df.iloc[:, 2:]
    feature_names = x_df.columns.tolist()
    class_names = [
        '落叶阔叶林', '常绿阔叶林', '针阔混交林', '常绿针叶林',
        '落叶针叶林', '竹林', '灌木林', '经济林'
    ]
    num_classes = len(np.unique(y))

    if len(class_names) != num_classes:
        st.error(f"指定的类别名称数量 ({len(class_names)}) 与数据中的类别数量 ({num_classes}) 不匹配。")
        return None

    x_train_df, x_test_df, y_train, y_test = train_test_split(x_df, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled_df = pd.DataFrame(scaler.fit_transform(x_train_df), columns=feature_names)
    X_test_scaled_df = pd.DataFrame(scaler.transform(x_test_df), columns=feature_names)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, feature_names, class_names, num_classes


# 使用缓存来训练模型，只有在数据变化时才重新训练，极大提升体验
@st.cache_data
def train_and_evaluate_models(_X_train, _y_train, _X_test, _y_test, class_names, num_classes):
    """训练所有模型并评估性能"""
    models = {
        'XGBoost': xgb.XGBClassifier(random_state=42, objective='multi:softprob', num_class=num_classes,
                                     eval_metric='mlogloss', use_label_encoder=False),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, multi_class='ovr', max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42, probability=True)
    }

    results = {}

    progress_bar = st.progress(0)
    total_models = len(models)

    for i, (model_name, model) in enumerate(models.items()):
        model.fit(_X_train, _y_train)
        y_pred_proba = model.predict_proba(_X_test)
        y_pred_class = model.predict(_X_test)

        accuracy = classification_report(_y_test, y_pred_class, target_names=class_names, output_dict=True)['accuracy']
        weighted_roc_auc = roc_auc_score(_y_test, y_pred_proba, multi_class='ovr', average='weighted')

        results[model_name] = {
            'model': model,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'weighted_roc_auc': weighted_roc_auc
        }
        progress_bar.progress((i + 1) / total_models, text=f"正在训练模型: {model_name}...")

    progress_bar.empty()  # 完成后隐藏进度条
    return results


def plot_multimodel_roc_comparison(results, y_true, class_names):
    """绘制多模型ROC性能比较图"""
    fig, ax = plt.subplots(figsize=(12, 9))
    mean_fpr = np.linspace(0, 1, 100)

    for model_name, result in results.items():
        y_proba = result['y_pred_proba']
        all_tpr = []
        # 注意 y_true 是从 1 开始的，需要转换为 0-indexed
        y_true_0_indexed = y_true - 1
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve((y_true_0_indexed == i).astype(int), y_proba[:, i])
            tpr = np.interp(mean_fpr, fpr, tpr)
            tpr[0] = 0.0
            all_tpr.append(tpr)
        mean_tpr = np.mean(all_tpr, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        ax.plot(mean_fpr, mean_tpr, label=f'{model_name} (宏平均AUC = {mean_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率', fontsize=14)
    ax.set_ylabel('真正率', fontsize=14)
    ax.set_title('多模型ROC性能比较 (宏平均)', fontsize=18)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True)
    return fig


def perform_shap_analysis(model, model_name, X_train, X_test, feature_names, class_names):
    """对最佳模型进行SHAP分析并返回图像"""
    st.write(f"正在为最佳模型 [{model_name}] 计算 SHAP 值，请稍候...")

    N_SHAP_SAMPLES = 50
    if len(X_test) > N_SHAP_SAMPLES:
        X_test_shap = X_test.sample(N_SHAP_SAMPLES, random_state=42)
    else:
        X_test_shap = X_test

    tree_based_models = ['XGBoost', 'LightGBM', 'RandomForest', 'DecisionTree']

    if model_name in tree_based_models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_shap)
    else:
        # KernelExplainer很慢，给用户一个提示
        with st.spinner(f'正在使用 KernelExplainer 计算 {model_name} 的 SHAP 值，这可能需要几分钟...'):
            background_data = shap.kmeans(X_train, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            shap_values = explainer.shap_values(X_test_shap)

    # 绘制 SHAP summary plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test_shap, feature_names=feature_names, class_names=class_names, show=False,
                      plot_type='bar')
    plt.title(f'{model_name} 综合特征重要性排序', fontsize=16)
    # 手动调整布局以防止标签被截断
    plt.tight_layout()
    return fig


# --- Streamlit UI 界面 ---
st.title("🌳 多模型智能林型分类与可解释性分析平台")
st.markdown("""
欢迎使用本平台！请上传您的林型数据（Excel格式），平台将自动完成以下任务：
1.  **数据处理**：划分训练集和测试集，并进行标准化。
2.  **多模型训练**：使用七种主流分类模型进行训练。
3.  **性能评估**：比较各模型的准确率和AUC，并可视化ROC曲线。
4.  **模型优选**：自动选出表现最佳的模型。
5.  **可解释性分析**：利用SHAP对最佳模型进行特征重要性分析。
""")

# --- 侧边栏：用户输入 ---
with st.sidebar:
    st.header("⚙️ 操作面板")
    uploaded_file = st.file_uploader("上传您的Excel数据文件", type=["xlsx"])

    # 使用默认数据作为示例
    use_default_data = st.checkbox("使用内置示例数据", value=True)

    analyze_button = st.button("🚀 开始分析")

# --- 主页面：结果展示 ---
if analyze_button:
    if uploaded_file is not None or use_default_data:
        if use_default_data:
            # 如果用户选择使用默认数据，我们从本地加载
            try:
                default_file_path = '林型分类数据.xlsx'
                uploaded_file = open(default_file_path, 'rb')
                st.info("正在使用内置示例数据...")
            except FileNotFoundError:
                st.error("错误：未找到默认数据文件 '林型分类数据.xlsx'。请取消勾选或将文件放置在应用根目录。")
                st.stop()

        # 1. 数据准备
        st.header("1. 数据准备与处理")
        with st.spinner('正在加载和处理数据...'):
            data_load_state = st.text('加载数据中...')
            processed_data = load_and_process_data(uploaded_file)
            data_load_state.text('数据加载和处理完成！')

        if processed_data:
            X_train, X_test, y_train, y_test, feature_names, class_names, num_classes = processed_data
            st.success("数据已成功划分为训练集和测试集，并完成标准化。")
            st.write(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

        # 2. 模型训练与评估
        st.header("2. 模型训练与比较")
        with st.spinner('正在训练多个模型，这可能需要一些时间...'):
            results = train_and_evaluate_models(X_train, y_train, X_test, y_test, class_names, num_classes)

        st.success("所有模型训练完成！")

        # 将结果整理成DataFrame展示
        results_df = pd.DataFrame({
            '模型': list(results.keys()),
            '准确率 (Accuracy)': [res['accuracy'] for res in results.values()],
            '加权ROC AUC': [res['weighted_roc_auc'] for res in results.values()]
        }).sort_values(by='加权ROC AUC', ascending=False).reset_index(drop=True)

        st.dataframe(results_df)

        # 3. 性能可视化
        st.header("3. 模型性能可视化")
        with st.spinner('正在生成ROC对比图...'):
            roc_fig = plot_multimodel_roc_comparison(results, y_test, class_names)
            st.pyplot(roc_fig)

        # 4. 最佳模型选择与下载
        st.header("4. 最佳模型选择与可解释性分析")
        best_model_name = max(results, key=lambda name: results[name]['weighted_roc_auc'])
        best_model = results[best_model_name]['model']

        st.info(f"**最佳模型是: {best_model_name}** (加权ROC AUC: {results[best_model_name]['weighted_roc_auc']:.4f})")

        # 将模型保存到内存中，以便用户下载
        model_buffer = BytesIO()
        joblib.dump(best_model, model_buffer)
        model_buffer.seek(0)

        st.download_button(
            label=f"📥 下载最佳模型 ({best_model_name}.joblib)",
            data=model_buffer,
            file_name=f'Best_Model_{best_model_name}.joblib',
            mime='application/octet-stream'
        )

        # 5. SHAP分析
        with st.spinner(f'正在为最佳模型 [{best_model_name}] 进行SHAP分析...'):
            shap_fig = perform_shap_analysis(best_model, best_model_name, X_train, X_test, feature_names, class_names)
            st.pyplot(shap_fig)

        st.balloons()
        st.success("所有分析完成！")

    else:
        st.warning("请上传一个文件或选择使用示例数据，然后点击'开始分析'按钮。")