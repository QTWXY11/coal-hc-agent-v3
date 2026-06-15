import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import requests

# ---------- 页面配置 ----------
st.set_page_config(page_title="煤基硬碳AI工艺智能体", layout="wide")
st.title("🏭 煤基硬碳合成工艺AI智能体")
st.markdown("**左侧输入参数 → 右侧自动检索相似案例 + 容量预测 + 大模型生成智能工艺方案**")

# ---------- 加载知识库 ----------
@st.cache_resource
def load_data():
    df = pd.read_csv("data.csv")
    feature_cols = ['ash', 'volatile', 'carbon_temp', 'hold_time', 'heating_rate']
    X = df[feature_cols].fillna(df[feature_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_scaled, df['capacity'])
    nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    nn.fit(X_scaled)
    return df, feature_cols, scaler, model, nn

df, feature_cols, scaler, model, nn = load_data()

# ---------- 调用大模型 ----------
def call_llm(prompt):
    """使用 secrets.toml 中配置的 API 调用大模型"""
    api_key = st.secrets.get("API_KEY")
    base_url = st.secrets.get("BASE_URL", "https://api.siliconflow.cn/v1")
    model_name = st.secrets.get("MODEL", "Qwen/Qwen2-7B-Instruct")

    if not api_key:
        return "⚠️ 未配置 API Key，请确认 `.streamlit/secrets.toml` 文件存在且包含 `API_KEY`。"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API 调用失败：{resp.text}"
    except Exception as e:
        return f"❌ 请求异常：{str(e)}"

# ---------- 左侧输入 ----------
st.sidebar.header("⚙️ 输入原料特性与目标")
coal_type = st.sidebar.selectbox("煤种", df['coal_type'].unique())
ash = st.sidebar.number_input("灰分 (%)", 0.0, 20.0, 8.0, step=0.1)
volatile = st.sidebar.number_input("挥发分 (%)", 0.0, 50.0, 35.0, step=0.1)
temp = st.sidebar.slider("碳化温度 (℃)", 800, 2000, 1300, step=10)
hold = st.sidebar.slider("保温时间 (h)", 0.5, 5.0, 2.0, step=0.5)
rate = st.sidebar.slider("升温速率 (℃/min)", 1, 20, 5, step=1)
target_cap = st.sidebar.number_input("目标容量 (mAh/g) 可选", 200, 500, 300, step=10)

# ---------- 主区域 ----------
if st.sidebar.button("🚀 生成工艺方案", use_container_width=True):
    with st.spinner("正在检索知识库、预测容量，并召唤大模型思考..."):
        # --- 1. 相似案例检索 ---
        input_vec = np.array([[ash, volatile, temp, hold, rate]])
        input_scaled = scaler.transform(input_vec)
        distances, indices = nn.kneighbors(input_scaled)
        similar_df = df.iloc[indices[0]].copy()
        
        # --- 2. 机器学习预测容量 ---
        pred_capacity = model.predict(input_scaled)[0]
        
        # --- 3. 原有规则工艺描述（保留，作为对比）---
        best_case = similar_df.iloc[0]
        rule_process = f"""
        根据最相似案例（{best_case['coal_type']}）调整：
        - 预处理：{best_case['process'].split('→')[0] if '→' in best_case['process'] else '酸洗（灰分>5%）或碱洗（挥发分>35%）'}
        - 碳化：{temp}℃ 氩气气氛，保温{hold}h，升温速率{rate}℃/min
        - 后处理：自然冷却（若追求更高倍率可考虑急冷）
        """
        
        # --- 4. 构造大模型提示词 ---
        similar_cases_text = ""
        for _, row in similar_df.iterrows():
            similar_cases_text += f"- 煤种{row['coal_type']}，灰分{row['ash']}%，挥发分{row['volatile']}%，工艺：{row['process']}，容量{row['capacity']}mAh/g\n"
        
        prompt = f"""你是一名煤基硬碳负极材料合成专家。用户需要制备煤基硬碳，原料参数和工艺要求如下：
- 煤种：{coal_type}
- 灰分：{ash}%
- 挥发分：{volatile}%
- 用户指定的碳化工艺：温度{temp}℃，保温{hold}h，升温速率{rate}℃/min
- 目标容量：{target_cap} mAh/g

知识库中与用户原料最相似的3条实验记录为：
{similar_cases_text}

请结合这些知识和你的专业经验，为用户提供一套**完整、具体、可执行**的煤基硬碳合成工艺方案。包括：
1. 原料预处理方法（酸洗、碱洗、氧化等，说明浓度、温度、时间）
2. 碳化工艺参数（是否微调用户给定的参数？是否需要两段碳化？）
3. 后处理（冷却方式、是否球磨等）
4. 预期电化学性能（可逆容量、首次库伦效率、倍率性能等）
5. 可能的优化建议和注意事项

请用清晰的条目输出，语言专业且简洁。
"""
        llm_response = call_llm(prompt)
        
        # --- 5. 显示结果 ---
        st.subheader("📋 大模型生成的智能工艺方案")
        st.markdown(llm_response)
        
        st.subheader("📊 容量预测与对比")
        col1, col2 = st.columns(2)
        col1.metric("机器学习预测可逆容量", f"{pred_capacity:.1f} mAh/g")
        col2.metric("最相似案例实际容量", f"{best_case['capacity']} mAh/g")
        
        if target_cap:
            st.metric("与目标容量差距", f"{pred_capacity - target_cap:.1f} mAh/g")
        
        with st.expander("🔍 规则备选工艺（仅供参考）"):
            st.markdown(rule_process)
        
        st.subheader("🔎 知识库中最相似的3个案例")
        st.dataframe(similar_df[['coal_type', 'ash', 'volatile', 'carbon_temp', 'hold_time', 'heating_rate', 'capacity', 'process']])
        
        st.info("💡 处理逻辑：先用数值特征检索相似案例 → 随机森林预测容量 → 大模型（基于用户参数+相似案例）生成完整方案。")
else:
    st.info("👈 请在左侧输入参数，然后点击「生成工艺方案」。右侧将展示大模型生成的专业方案、容量预测及参考案例。")

st.markdown("---")
st.caption("本项目基于 Streamlit 部署 | 知识库 + 机器学习 + 大模型（SiliconFlow API）")
