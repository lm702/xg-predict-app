"""
Streamlit Web App: xG 智能博彩预测系统
功能：上传主客队 xG 数据 -> 特征工程 -> 玩家经验修正 -> 泊松概率生成 -> 价值投注分析
"""

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

# ----------------------------------------------------------------------
# 页面设置
# ----------------------------------------------------------------------
st.set_page_config(page_title="⚽ xG 智能预测系统", layout="wide")
st.title("⚽ xG 智能博彩预测系统")
st.markdown("#### 贝叶斯动态博弈 · 玩家灵魂注入")

# ----------------------------------------------------------------------
# 会话状态初始化（用于在多次交互中保持数据）
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 侧边栏：数据上传
# ----------------------------------------------------------------------
st.sidebar.header("📁 数据加载")
home_file = st.sidebar.file_uploader("主队历史数据 (Excel)", type=["xlsx", "xls"], key="home")
away_file = st.sidebar.file_uploader("客队历史数据 (Excel)", type=["xlsx", "xls"], key="away")

if home_file and away_file:
    if st.sidebar.button("加载数据并初始化"):
        try:
            home_df = parse_uploaded_excel(home_file)
            away_df = parse_uploaded_excel(away_file)
            st.session_state.all_matches = build_all_team_matches(home_df, away_df)
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

# ----------------------------------------------------------------------
# 主界面：预测操作
# ----------------------------------------------------------------------
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
            match_dt = pd.Timestamp(match_date)
            home_f = get_team_features_before(home_team, match_dt, st.session_state.all_matches)
            away_f = get_team_features_before(away_team, match_dt, st.session_state.all_matches)
            if home_f is None or away_f is None:
                st.error("历史比赛不足（至少需要5场），无法计算特征。")
            else:
                st.session_state.current_features = {'home': home_f, 'away': away_f}
                st.session_state.base_lambda = predict_base_lambdas(home_f, away_f)
                st.session_state.adjusted_lambda = dict(st.session_state.base_lambda)
                st.session_state.last_home_team = home_team
                st.session_state.last_away_team = away_team
                # 生成初步概率（未修正）
                st.session_state.prob_matrix = generate_bivariate_poisson(
                    st.session_state.adjusted_lambda['h'],
                    st.session_state.adjusted_lambda['a'], rho=0.15
                )
                st.success("特征计算完成，基础 λ 已生成。请查看下方修正面板。")

    # ------------------------------------------------------------------
    # 特征展示
    # ------------------------------------------------------------------
    if st.session_state.current_features:
        st.header("📊 核心特征对比")
        col_h, col_a = st.columns(2)
        home_f = st.session_state.current_features['home']
        away_f = st.session_state.current_features['away']

        with col_h:
            st.subheader(st.session_state.last_home_team)
            st.metric("进攻强度 λ_att", f"{home_f['lambda_att']:.2f}")
            st.metric("防守强度 λ_def", f"{home_f['lambda_def']:.2f}")
            st.metric("转化率", f"{home_f['conv']:.2f}")
            st.metric("对手转化率", f"{home_f['opp_conv']:.2f}")
            st.metric("定位球攻/防", f"{home_f['lambda_sp_att']:.2f} / {home_f['lambda_sp_def']:.2f}")
            st.metric("渗透效率", f"{home_f['penetration']:.2f}")
            st.metric("控球率", f"{home_f['poss']:.1f}%")
            st.metric("风格标签", f"{home_f['label']}")

        with col_a:
            st.subheader(st.session_state.last_away_team)
            st.metric("进攻强度 λ_att", f"{away_f['lambda_att']:.2f}")
            st.metric("防守强度 λ_def", f"{away_f['lambda_def']:.2f}")
            st.metric("转化率", f"{away_f['conv']:.2f}")
            st.metric("对手转化率", f"{away_f['opp_conv']:.2f}")
            st.metric("定位球攻/防", f"{away_f['lambda_sp_att']:.2f} / {away_f['lambda_sp_def']:.2f}")
            st.metric("渗透效率", f"{away_f['penetration']:.2f}")
            st.metric("控球率", f"{away_f['poss']:.1f}%")
            st.metric("风格标签", f"{away_f['label']}")

        # ------------------------------------------------------------------
        # 自动生成比赛解读（位于特征展示和玩家修正之间）
        # ------------------------------------------------------------------
        st.header("🔍 数据洞察 & 玩家解读")
        with st.expander("点击展开深度分析（包含攻防画像、战术洞察与决策参考）", expanded=False):
            # 读取最新概率（可能是修正前的，也可能是修正后的，这里用当前 session_state）
            probs = get_win_draw_probs(st.session_state.prob_matrix)
            ou = get_ou_probs(st.session_state.prob_matrix)
            lam_h = st.session_state.adjusted_lambda['h']
            lam_a = st.session_state.adjusted_lambda['a']

            # 调用洞察生成函数（定义在文件末尾）
            insights = generate_match_insights(
                st.session_state.last_home_team,
                st.session_state.last_away_team,
                home_f, away_f,
                lam_h, lam_a,
                probs, ou
            )
            st.markdown(insights, unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # 玩家经验修正面板
        # ------------------------------------------------------------------
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

            col_mot1, col_mot2 = st.columns(2)
            with col_mot1:
                mot_home = st.slider("主队战意 (1=无欲求, 5=生死战)", 1, 5, 3)
            with col_mot2:
                mot_away = st.slider("客队战意 (1=无欲求, 5=生死战)", 1, 5, 3)

            weather = st.selectbox("天气", ["晴", "雨", "大风"])
            ref = st.selectbox("裁判尺度", ["正常", "严哨", "松哨"])

            if st.button("应用修正并重新计算概率"):
                base = st.session_state.base_lambda
                adj = apply_player_factors(
                    base['h'], base['a'],
                    home_style, away_style,
                    home_injury, away_injury,
                    mot_home, mot_away,      # 传入主客队战意
                    weather, ref
                )
                st.session_state.adjusted_lambda = adj
                st.session_state.prob_matrix = generate_bivariate_poisson(adj['h'], adj['a'], rho=0.15)
                st.success("修正已应用，概率已更新。页面将自动刷新解读内容。")
                st.rerun()  # 强制刷新以让解读区使用最新概率

        # ------------------------------------------------------------------
        # 最终结果显示（概率矩阵 + 价值对比）
        # ------------------------------------------------------------------
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

            # 比分概率热力图（表格形式）
            st.subheader("比分概率热力图")
            max_g = min(6, len(st.session_state.prob_matrix)-1)
            heat_data = []
            for i in range(max_g+1):
                row = []
                for j in range(max_g+1):
                    row.append(round(st.session_state.prob_matrix[i][j]*100, 2))
                heat_data.append(row)
            st.dataframe(heat_data, use_container_width=True)
            st.caption("行：主队进球  列：客队进球  数值为概率(%)")

            # 市场赔率对比与价值发现
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


# ======================================================================
# 辅助函数：生成比赛洞察文本
# ======================================================================
def generate_match_insights(home_team, away_team, home_f, away_f, lam_h, lam_a, probs, ou):
    """
    根据两队特征和预测概率，生成类似专家分析的解读文本。
    返回 Markdown 字符串。
    """
    # 提取特征
    home_att = home_f['lambda_att']
    home_def = home_f['lambda_def']
    away_att = away_f['lambda_att']
    away_def = away_f['lambda_def']
    home_style = home_f['label']
    away_style = away_f['label']
    home_conv = home_f['conv']
    away_conv = away_f['conv']
    home_opp_conv = home_f['opp_conv']
    away_opp_conv = away_f['opp_conv']
    home_cv = home_f['cv_att']
    away_cv = away_f['cv_att']
    home_poss = home_f['poss']
    away_poss = away_f['poss']
    home_pen = home_f['penetration']
    away_pen = away_f['penetration']
    home_sp_att = home_f['lambda_sp_att']
    away_sp_att = away_f['lambda_sp_att']
    home_sp_def = home_f['lambda_sp_def']
    away_sp_def = away_f['lambda_sp_def']

    diff_att = home_att - away_def
    diff_def = home_def - away_att
    total_xg = lam_h + lam_a

    insights = f"""
### 📊 攻防画像对比

**{home_team}** (风格: {home_style})  
进攻强度 λ_att: {home_att:.2f} · 防守强度 λ_def: {home_def:.2f}  
转化率: {home_conv:.2f} · 对手转化率: {home_opp_conv:.2f} · 进攻波动 CV: {home_cv:.2f}  
控球率: {home_poss:.1f}% · 渗透效率: {home_pen:.2f} · 定位球 xG/失: {home_sp_att:.2f} / {home_sp_def:.2f}

**{away_team}** (风格: {away_style})  
进攻强度 λ_att: {away_att:.2f} · 防守强度 λ_def: {away_def:.2f}  
转化率: {away_conv:.2f} · 对手转化率: {away_opp_conv:.2f} · 进攻波动 CV: {away_cv:.2f}  
控球率: {away_poss:.1f}% · 渗透效率: {away_pen:.2f} · 定位球 xG/失: {away_sp_att:.2f} / {away_sp_def:.2f}

---

### 🧠 玩家洞察
"""
    # 1. 攻防差异
    insights += f"- **进攻优势差** (主队攻 - 客队守): {diff_att:+.2f} "
    if diff_att > 0.5:
        insights += "→ 主队进攻明显压制客队防线。\n"
    elif diff_att > 0.2:
        insights += "→ 主队进攻略占上风。\n"
    else:
        insights += "→ 客队防守有能力限制主队火力。\n"

    insights += f"- **防守劣势差** (主队守 - 客队攻): {diff_def:+.2f} "
    if diff_def < -0.3:
        insights += "→ 主队防线面对客队进攻处于下风。\n"
    elif diff_def > 0.2:
        insights += "→ 主队防守相对稳固。\n"
    else:
        insights += "→ 双方攻防基本均衡。\n"

    # 2. 转化率异常警报
    if home_opp_conv > 1.3:
        insights += f"- ⚠️ **{home_team} 防守警报：对手射门转化率 {home_opp_conv:.2f} 极高，说明球队在限制对手射门质量上存在严重漏洞（门将或防线问题）。**\n"
    if away_opp_conv > 1.3:
        insights += f"- ⚠️ **{away_team} 防守警报：对手射门转化率 {away_opp_conv:.2f} 极高。**\n"
    if home_opp_conv < 0.9:
        insights += f"- ✅ {home_team} 防守出色，对手转化率仅 {home_opp_conv:.2f}。\n"
    if away_opp_conv < 0.9:
        insights += f"- ✅ {away_team} 防守出色，对手转化率仅 {away_opp_conv:.2f}。\n"

    # 3. 进球波动性
    if home_cv > 0.6:
        insights += f"- 🎲 {home_team} 进攻极不稳定 (CV={home_cv:.2f})，可能大胜也可能哑火。\n"
    if away_cv > 0.6:
        insights += f"- 🎲 {away_team} 进攻极不稳定 (CV={away_cv:.2f})。\n"

    # 4. 风格克制提示
    style_map = {
        'POS': {'TRN': '传控 vs 防反 → 客队反击威胁大', 'LOW': '传控破大巴 → 主队可能久攻不下', 'GEG': '高位传控 vs 高压迫 → 高强度对攻'},
        'TRN': {'POS': '防反克制传控 → 主队反击利器', 'GEG': '转换 vs 高压 → 高失误率比赛'},
        'LOW': {'POS': '大巴 vs 传控 → 沉闷或偷袭', 'TRN': '低位 vs 转换 → 效率优先'},
        'GEG': {'POS': '压迫 vs 传控 → 抢断后反击', 'TRN': '压迫 vs 防反 → 高强度往返'},
    }
    if home_style in style_map and away_style in style_map[home_style]:
        insights += f"- ⚔️ 战术克制: {style_map[home_style][away_style]}\n"

    # 5. 定位球威胁
    if home_sp_att > 0.5:
        insights += f"- 🎯 {home_team} 定位球攻击力强 (xG {home_sp_att:.2f})，需警惕。\n"
    if away_sp_att > 0.5:
        insights += f"- 🎯 {away_team} 定位球攻击力强 (xG {away_sp_att:.2f})。\n"

    # 6. 修正后的概率
    insights += f"""
---

### 📈 修正后概率与预期

- 预期进球：主 {lam_h:.2f} - 客 {lam_a:.2f} (总 {total_xg:.2f})
- 胜平负：主 {probs['home']*100:.1f}% / 平 {probs['draw']*100:.1f}% / 客 {probs['away']*100:.1f}%
- 大小球 (2.5)：小 {ou['under']*100:.1f}% / 大 {ou['over']*100:.1f}%

---
"""
    # 7. 玩家决策参考
    decision = "\n### 🎲 玩家决策参考\n"
    if probs['home'] > 0.45:
        decision += "- 主胜概率较高，但需结合市场赔率判断价值。\n"
    elif probs['away'] > 0.40:
        decision += "- 客队不败可能性较大，关注下盘机会。\n"
    else:
        decision += "- 平局概率较高，可考虑保守策略。\n"

    if ou['under'] > 0.55:
        decision += "- 倾向小球 (<2.5)，适合关注比分 1-0/0-1/1-1。\n"
    elif ou['over'] > 0.55:
        decision += "- 倾向大球 (>2.5)，可能打出 2-1/2-2 等比分。\n"
    else:
        decision += "- 大小球均势，谨慎参与。\n"

    decision += "- 若市场赔率与公平概率明显偏离（>3%），可考虑小注价值投注。\n"
    decision += "- 注意临场首发和天气变化可能改变模型假设。\n"

    insights += decision
    return insights
