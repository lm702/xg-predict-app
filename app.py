import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    parse_uploaded_excel,
    build_all_team_matches,
    get_team_features_before,
    predict_base_lambdas,
    apply_player_factors,
    generate_bivariate_poisson,
    get_win_draw_probs,
    get_ou_probs,
    LEAGUE_AVG_XG,
    STYLE_MATRIX
)

# ---------- 页面设置 ----------
st.set_page_config(page_title="⚽ xG 智能预测系统", layout="wide")
st.title("⚽ xG 智能博彩预测系统")
st.markdown("#### 贝叶斯动态博弈 · 玩家灵魂注入")

# ---------- 会话状态初始化 ----------
if 'all_matches' not in st.session_state:
    st.session_state.all_matches = []
if 'teams' not in st.session_state:
    st.session_state.teams = set()
if 'current_features' not in st.session_state:
    st.session_state.current_features = {}
if 'base_lambda' not in st.session_state:
    st.session_state.base_lambda = {'h': 1.5, 'a': 0.9}
if 'adjusted_lambda' not in st.session_state:
    st.session_state.adjusted_lambda = {'h': 1.5, 'a': 0.9}
if 'prob_matrix' not in st.session_state:
    st.session_state.prob_matrix = []
if 'last_home_team' not in st.session_state:
    st.session_state.last_home_team = ''
if 'last_away_team' not in st.session_state:
    st.session_state.last_away_team = ''

# ---------- 侧边栏：数据上传 ----------
st.sidebar.header("📁 数据加载")
home_file = st.sidebar.file_uploader("主队历史数据 (Excel)", type=["xlsx", "xls"], key="home")
away_file = st.sidebar.file_uploader("客队历史数据 (Excel)", type=["xlsx", "xls"], key="away")

if home_file and away_file:
    if st.sidebar.button("加载数据并初始化"):
        try:
            # 解析两个文件
            home_df = parse_uploaded_excel(home_file)
            away_df = parse_uploaded_excel(away_file)
            st.session_state.all_matches = build_all_team_matches(home_df, away_df)
            # 提取球队名称
            teams = set()
            for m in st.session_state.all_matches:
                teams.add(m['home_team'])
                teams.add(m['away_team'])
            st.session_state.teams = sorted(teams)
            st.sidebar.success(f"✅ 成功加载 {len(st.session_state.all_matches)} 场比赛，{len(teams)} 支球队。")
        except Exception as e:
            st.sidebar.error(f"❌ 解析错误：{e}")
else:
    st.sidebar.info("👆 请上传主队和客队的 Excel 数据（文件结构与对话中相同）。")

# ---------- 主界面：预测操作 ----------
if len(st.session_state.teams) > 1:
    st.header("⚔️ 比赛设定")
    col1, col2, col3 = st.columns(3)
    with col1:
        home_team = st.selectbox("主队", st.session_state.teams, key="home_team")
    with col2:
        away_team = st.selectbox("客队", st.session_state.teams, key="away_team")
    with col3:
        match_date = st.date_input("比赛日期 (预测基准)", key="match_date")

    if st.button("🔮 执行预测 (计算特征 & 基础 λ)"):
        if home_team == away_team:
            st.error("主客队不能相同。")
        else:
            # 计算特征
            match_dt = pd.Timestamp(match_date)
            home_f = get_team_features_before(home_team, match_dt, st.session_state.all_matches)
            away_f = get_team_features_before(away_team, match_dt, st.session_state.all_matches)
            if home_f is None or away_f is None:
                st.error("历史比赛不足（至少需要10场），无法计算特征。")
            else:
                st.session_state.current_features = {'home': home_f, 'away': away_f}
                st.session_state.base_lambda = predict_base_lambdas(home_f, away_f)
                st.session_state.adjusted_lambda = dict(st.session_state.base_lambda)
                st.session_state.last_home_team = home_team
                st.session_state.last_away_team = away_team
                # 生成初步概率
                st.session_state.prob_matrix = generate_bivariate_poisson(
                    st.session_state.adjusted_lambda['h'],
                    st.session_state.adjusted_lambda['a'], rho=0.15
                )
                st.success("特征计算完成，基础λ已生成。请查看下方修正面板。")

    # ---------- 特征展示 ----------
    if st.session_state.current_features:
        st.header("📊 核心特征对比")
        col_h, col_a = st.columns(2)
        with col_h:
            f = st.session_state.current_features['home']
            st.subheader(st.session_state.last_home_team)
            st.metric("进攻强度 λ_att", f"{f['lambda_att']:.2f}")
            st.metric("防守强度 λ_def", f"{f['lambda_def']:.2f}")
            st.metric("转化率", f"{f['conv']:.2f}")
            st.metric("对手转化率", f"{f['opp_conv']:.2f}")
            st.metric("定位球攻/防", f"{f['lambda_sp_att']:.2f} / {f['lambda_sp_def']:.2f}")
            st.metric("渗透效率", f"{f['penetration']:.2f}")
            st.metric("控球率", f"{f['poss']:.1f}%")
            st.metric("风格标签", f"{f['label']}")
        with col_a:
            f = st.session_state.current_features['away']
            st.subheader(st.session_state.last_away_team)
            st.metric("进攻强度 λ_att", f"{f['lambda_att']:.2f}")
            st.metric("防守强度 λ_def", f"{f['lambda_def']:.2f}")
            st.metric("转化率", f"{f['conv']:.2f}")
            st.metric("对手转化率", f"{f['opp_conv']:.2f}")
            st.metric("定位球攻/防", f"{f['lambda_sp_att']:.2f} / {f['lambda_sp_def']:.2f}")
            st.metric("渗透效率", f"{f['penetration']:.2f}")
            st.metric("控球率", f"{f['poss']:.1f}%")
            st.metric("风格标签", f"{f['label']}")

        # ---------- 玩家修正面板 ----------
        st.header("🧠 玩家经验修正")
        with st.expander("展开修正选项", expanded=True):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                home_style = st.selectbox("主队战术风格", 
                    ['POS','DIR','TRN','GEG','LOW','POSL','BALA'],
                    index=['POS','DIR','TRN','GEG','LOW','POSL','BALA'].index(
                        st.session_state.current_features['home'].get('label', 'BALA')))
                home_injury = st.checkbox("主队核心伤停 (λ×0.85)")
            with col_s2:
                away_style = st.selectbox("客队战术风格", 
                    ['POS','DIR','TRN','GEG','LOW','POSL','BALA'],
                    index=['POS','DIR','TRN','GEG','LOW','POSL','BALA'].index(
                        st.session_state.current_features['away'].get('label', 'BALA')))
                away_injury = st.checkbox("客队核心伤停 (λ×0.80)")

            mot = st.slider("主队战意 (1=无欲求, 5=生死战)", 1, 5, 3)
            weather = st.selectbox("天气", ["晴", "雨", "大风"])
            ref = st.selectbox("裁判尺度", ["正常", "严哨", "松哨"])

            if st.button("应用修正并重新计算概率"):
                base = st.session_state.base_lambda
                adj = apply_player_factors(
                    base['h'], base['a'],
                    home_style, away_style,
                    home_injury, away_injury,
                    mot, weather, ref
                )
                st.session_state.adjusted_lambda = adj
                st.session_state.prob_matrix = generate_bivariate_poisson(adj['h'], adj['a'], rho=0.15)
                st.success("修正已应用，概率已更新。")

        # ---------- 结果显示 ----------
        if len(st.session_state.prob_matrix) > 0:
            st.header("🎯 预测结果")
            lam_h = st.session_state.adjusted_lambda['h']
            lam_a = st.session_state.adjusted_lambda['a']
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("主队 λ_h", f"{lam_h:.2f}")
            col_m2.metric("客队 λ_a", f"{lam_a:.2f}")
            col_m3.metric("总进球期望", f"{lam_h+lam_a:.2f}")

            probs = get_win_draw_probs(st.session_state.prob_matrix)
            ou = get_ou_probs(st.session_state.prob_matrix)
            col_w, col_o = st.columns(2)
            with col_w:
                st.subheader("胜平负概率")
                st.write(f"主胜: {probs['home']*100:.1f}% (公平赔率 {1/probs['home']:.2f})")
                st.write(f"平局: {probs['draw']*100:.1f}% (公平赔率 {1/probs['draw']:.2f})")
                st.write(f"客胜: {probs['away']*100:.1f}% (公平赔率 {1/probs['away']:.2f})")
            with col_o:
                st.subheader("大小球 (2.5)")
                st.write(f"小于2.5: {ou['under']*100:.1f}% (公平赔率 {1/ou['under']:.2f})")
                st.write(f"大于2.5: {ou['over']*100:.1f}% (公平赔率 {1/ou['over']:.2f})")

            # 热力图
            st.subheader("比分概率热力图")
            max_g = min(6, len(st.session_state.prob_matrix)-1)
            heat_data = []
            for i in range(max_g+1):
                row = []
                for j in range(max_g+1):
                    row.append(round(st.session_state.prob_matrix[i][j]*100, 2))
                heat_data.append(row)
            fig = st.dataframe(heat_data, use_container_width=True)
            st.caption("行：主队进球  列：客队进球  数值为概率(%)")

            # 市场赔率对比
            st.subheader("📈 价值发现 (公平赔率 vs 市场)")
            col_odds = st.columns(3)
            with col_odds[0]:
                mkt_h = st.number_input("主胜赔率", 1.0, 50.0, 2.2, 0.01)
            with col_odds[1]:
                mkt_d = st.number_input("平局赔率", 1.0, 50.0, 3.5, 0.01)
            with col_odds[2]:
                mkt_a = st.number_input("客胜赔率", 1.0, 50.0, 3.4, 0.01)

            fair_h = 1/probs['home']
            fair_d = 1/probs['draw']
            fair_a = 1/probs['away']
            val_h = (mkt_h * probs['home'] - 1)*100
            val_d = (mkt_d * probs['draw'] - 1)*100
            val_a = (mkt_a * probs['away'] - 1)*100

            st.write(f"主胜: 公平 {fair_h:.2f} | 价值 {val_h:.1f}%")
            st.write(f"平局: 公平 {fair_d:.2f} | 价值 {val_d:.1f}%")
            st.write(f"客胜: 公平 {fair_a:.2f} | 价值 {val_a:.1f}%")
            if val_h > 2 or val_d > 2 or val_a > 2:
                st.success("⚠️ 存在正向价值！请谨慎决策。")
else:
    st.info("请上传至少两支球队的数据以开始预测。")