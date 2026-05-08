import pandas as pd
import numpy as np
import math
from io import BytesIO

# ---------- 常量 ----------
ALPHA = 0.94
WINDOW = 10
LEAGUE_AVG_XG = 1.4   # 近似联赛平均每队每场 xG

# ---------- 风格修正矩阵 ----------
STYLE_MATRIX = {
    'POS': {'TRN': (0.85, 1.10), 'LOW': (0.75, 1.05), 'GEG': (0.90, 1.0), 'DIR': (1.15, 0.95), 'BALA': (1.0, 1.0), 'POSL': (1.05, 0.95)},
    'TRN': {'POS': (1.20, 0.90), 'GEG': (1.25, 0.85), 'LOW': (0.90, 1.0), 'DIR': (1.10, 0.95), 'BALA': (1.05, 0.98)},
    'LOW': {'POS': (0.95, 1.0), 'TRN': (0.85, 1.10), 'GEG': (0.80, 1.05), 'BALA': (0.95, 1.0)},
    'GEG': {'POS': (1.05, 0.95), 'TRN': (0.90, 1.15), 'LOW': (0.85, 1.05), 'POSL': (1.10, 0.85)},
    'DIR':  {'POS': (0.90, 1.05), 'LOW': (1.05, 0.95), 'BALA': (1.0, 1.0)},
    'POSL': {'POS': (0.95, 1.0), 'TRN': (0.80, 1.20), 'GEG': (0.85, 1.10)}
}

# ---------- 解析上传的 Excel ----------
def parse_uploaded_excel(uploaded_file):
    """读取上传的Excel，返回标准化DataFrame"""
    df = pd.read_excel(uploaded_file, sheet_name=0)
    # 显示原始列名，方便调试（在Streamlit界面中可以看到）
    print("检测到的列名：", list(df.columns))
    
    # 更精确的列名映射字典
    column_mapping = {
        '日期': '日期',
        '赛事': '赛事',
        '主队': '主队',
        '主队比分': '主队比分',
        '客队比分': '客队比分',
        '客队': '客队',
        '主队xG': '主队xG',
        '客队xG': '客队xG',
        '主队xG Open Play': '主队xG Open Play',
        '客队xG Open Play': '客队xG Open Play',
        '主队xG Set Play': '主队xG Set Play',
        '客队xG Set Play': '客队xG Set Play',
        '主队Non-Pen xG': '主队Non-Pen xG',
        '客队Non-Pen xG': '客队Non-Pen xG',
        '主队xGOT': '主队xGOT',
        '客队xGOT': '客队xGOT',
        '主队控球率': '主队控球率',
        '客队控球率': '客队控球率',
        '主队对方禁区触球': '主队对方禁区触球',
        '客队对方禁区触球': '客队对方禁区触球',
    }
    # 重命名现有列（只重命名那些在映射字典中的列）
    df.rename(columns=column_mapping, inplace=True)
    return df

def build_all_team_matches(home_df, away_df):
    """将两个DataFrame转换为统一的比赛字典列表"""
    required = ['日期','主队','客队','主队Non-Pen xG','客队Non-Pen xG',
                '主队xG Set Play','客队xG Set Play','主队xGOT','客队xGOT',
                '主队控球率','客队控球率','主队对方禁区触球','客队对方禁区触球']
    matches = []
    for df in [home_df, away_df]:
        # 检查是否包含所有必要列
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"缺失必要列：{missing}。当前列：{list(df.columns)}")
        for _, row in df.iterrows():
            try:
                date = pd.Timestamp(row['日期'])
                m = {
                    'date': date,
                    'home_team': str(row['主队']).strip(),
                    'away_team': str(row['客队']).strip(),
                    'home_nonpen_xg': float(row['主队Non-Pen xG'] or 0),
                    'away_nonpen_xg': float(row['客队Non-Pen xG'] or 0),
                    'home_xg_set': float(row['主队xG Set Play'] or 0),
                    'away_xg_set': float(row['客队xG Set Play'] or 0),
                    'home_xgot': float(row['主队xGOT'] or 0),
                    'away_xgot': float(row['客队xGOT'] or 0),
                    'home_poss': float(row['主队控球率'] or 50),
                    'away_poss': float(row['客队控球率'] or 50),
                    'home_touches_box': float(row['主队对方禁区触球'] or 0),
                    'away_touches_box': float(row['客队对方禁区触球'] or 0)
                }
                matches.append(m)
            except Exception as e:
                continue  # 跳过解析失败的行
    return sorted(matches, key=lambda x: x['date'])

# ---------- 特征工程 (滚动窗口) ----------
def get_team_features_before(team, as_of_date, all_matches):
    """返回球队在 as_of_date 之前的特征字典"""
    team_matches = []
    for m in all_matches:
        if (m['home_team'] == team or m['away_team'] == team) and m['date'] < as_of_date:
            team_matches.append(m)
    team_matches.sort(key=lambda x: x['date'], reverse=True)
    recent = team_matches[:WINDOW]
    if len(recent) < 5:
        return None

    now = as_of_date
    weights = []
    for m in recent:
        diff_days = (now - m['date']).days
        w = ALPHA ** (diff_days / 7)
        weights.append(w)
    total_w = sum(weights)
    w = [x / total_w for x in weights]

    def extract(getter):
        return [getter(m) for m in recent]

    my_nonpen = extract(lambda m: m['home_nonpen_xg'] if m['home_team'] == team else m['away_nonpen_xg'])
    opp_nonpen = extract(lambda m: m['away_nonpen_xg'] if m['home_team'] == team else m['home_nonpen_xg'])
    my_xg_set = extract(lambda m: m['home_xg_set'] if m['home_team'] == team else m['away_xg_set'])
    opp_xg_set = extract(lambda m: m['away_xg_set'] if m['home_team'] == team else m['home_xg_set'])
    my_xgot = extract(lambda m: m['home_xgot'] if m['home_team'] == team else m['away_xgot'])
    opp_xgot = extract(lambda m: m['away_xgot'] if m['home_team'] == team else m['home_xgot'])
    my_poss = extract(lambda m: m['home_poss'] if m['home_team'] == team else m['away_poss'])
    my_touches = extract(lambda m: m['home_touches_box'] if m['home_team'] == team else m['away_touches_box'])
    opp_touches = extract(lambda m: m['away_touches_box'] if m['home_team'] == team else m['home_touches_box'])

    w_sum = lambda arr: sum(arr[i]*w[i] for i in range(len(arr)))
    lambda_att = w_sum(my_nonpen)
    lambda_def = w_sum(opp_nonpen)
    lambda_sp_att = w_sum(my_xg_set)
    lambda_sp_def = w_sum(opp_xg_set)
    conv = w_sum(my_xgot) / lambda_att if lambda_att else 1.0
    opp_conv = w_sum(opp_xgot) / lambda_def if lambda_def else 1.0
    penetration = w_sum(my_touches) / (w_sum(my_poss)/100) if w_sum(my_poss) else 0
    xg_per_touch = w_sum(my_nonpen) / w_sum(my_touches) if w_sum(my_touches) else 0
    poss = w_sum(my_poss)
    box_protection = w_sum(opp_touches)

    variance = sum(w[i] * (my_nonpen[i] - lambda_att)**2 for i in range(len(recent)))
    sigma_att = np.sqrt(variance)
    cv_att = sigma_att / lambda_att if lambda_att else 0

    def determine_label():
        if poss > 55 and penetration < 0.6 and xg_per_touch < 0.08: return 'POSL'
        if poss > 55 and penetration >= 0.6: return 'POS'
        if poss < 45 and penetration > 0.7 and xg_per_touch > 0.09: return 'TRN'
        if poss < 40 and xg_per_touch > 0.1: return 'LOW'
        if 50 <= poss <= 58 and penetration > 0.65: return 'GEG'
        if poss < 45 and xg_per_touch < 0.07: return 'DIR'
        return 'BALA'

    return {
        'lambda_att': lambda_att,
        'lambda_def': lambda_def,
        'lambda_sp_att': lambda_sp_att,
        'lambda_sp_def': lambda_sp_def,
        'conv': conv,
        'opp_conv': opp_conv,
        'penetration': penetration,
        'xg_per_touch': xg_per_touch,
        'poss': poss,
        'box_protection': box_protection,
        'cv_att': cv_att,
        'label': determine_label()
    }

# ---------- 基础 λ 预测 (手工替代 XGBoost) ----------
def predict_base_lambdas(home_f, away_f):
    home_att = home_f['lambda_att']
    away_def = away_f['lambda_def']
    away_att = away_f['lambda_att']
    home_def = home_f['lambda_def']
    home_factor = 1.08
    h = (home_att * home_factor + LEAGUE_AVG_XG - away_def * 0.95) / 2
    a = (away_att + LEAGUE_AVG_XG - home_def * 0.95) / 2
    return {'h': max(0.3, h), 'a': max(0.3, a)}

# ---------- 玩家修正 (支持主客队独立战意) ----------
def apply_player_factors(lamH, lamA, home_style, away_style,
                         home_inj, away_inj,
                         mot_home, mot_away,
                         weather, ref):
    # 战术修正
    if home_style in STYLE_MATRIX and away_style in STYLE_MATRIX[home_style]:
        coeff_h, coeff_a = STYLE_MATRIX[home_style][away_style]
        lamH *= coeff_h
        lamA *= coeff_a
    # 伤病
    if home_inj: lamH *= 0.85
    if away_inj: lamA *= 0.80
    # 战意 (分别影响)
    mot_map = {1: 0.88, 2: 0.94, 3: 1.0, 4: 1.02, 5: 1.04}
    lamH *= mot_map.get(mot_home, 1.0)
    lamA *= mot_map.get(mot_away, 1.0)
    # 天气
    if weather == '雨':
        lamH *= 0.88; lamA *= 0.88
    elif weather == '大风':
        lamH *= 0.92; lamA *= 0.92
    # 裁判 (点球期望微调)
    if ref == '严哨':
        lamH += 0.03; lamA += 0.03
    elif ref == '松哨':
        lamH *= 0.98; lamA *= 0.98
    return {'h': max(0.3, lamH), 'a': max(0.3, lamA)}

# ---------- 双变量泊松 (独立 + 对角线强化) ----------
def factorial(n):
    return math.factorial(n)

def poisson_pmf(k, lam):
    return np.exp(-lam) * (lam**k) / factorial(k)

def generate_bivariate_poisson(lamH, lamA, rho=0.15, max_g=6):
    mat = np.zeros((max_g+1, max_g+1))
    for i in range(max_g+1):
        for j in range(max_g+1):
            p = poisson_pmf(i, lamH) * poisson_pmf(j, lamA)
            if i == j:
                p *= (1 + rho)
            elif abs(i-j) == 1:
                p *= (1 - rho*0.5)
            mat[i, j] = p
    mat /= mat.sum()
    return mat

def get_win_draw_probs(mat):
    home, draw, away = 0.0, 0.0, 0.0
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if i > j: home += mat[i,j]
            elif i == j: draw += mat[i,j]
            else: away += mat[i,j]
    return {'home': home, 'draw': draw, 'away': away}

def get_ou_probs(mat, line=2.5):
    under, over = 0.0, 0.0
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if i + j < line: under += mat[i,j]
            else: over += mat[i,j]
    return {'under': under, 'over': over}
