"""
DS Mini Project - DAY2 1단계: 셀 단위 초기 사이클 피처
=====================================================
notebooks/01_EDA.ipynb (Q3, Q4, Q5)의 피처 계산 코드를 그대로 옮긴 모듈
1) parse_policy()         : 충전 정책 문자열 -> C1, switch_SOC, C2
2) deltaq_features()      : ΔQ(V) = Q_100(V) - Q_10(V) 통계량
3) early_cycle_features() : 사이클 2~100 summary 피처 (용량, 충전시간, 온도, 내부저항)
4) build_features()       : 셀 1행 피처 테이블

사용법:
    python -m src.features                 # 프로젝트 루트에서 실행 -> processed/features.csv

    from src.preprocess import run_all
    from src.features import build_features, FEATURE_SETS
    feat = build_features(run_all())
"""
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

from src.preprocess import OUT_DIR, run_all


# =========================================================
# 설정
# =========================================================
CYC_A, CYC_B = 10, 100      # ΔQ(V) = Q_CYC_B(V) - Q_CYC_A(V)

DELTAQ_FEATURES = ['dQ_min', 'dQ_mean', 'dQ_var', 'dQ_skew', 'dQ_kurt', 'log_var', 'log_abs_min']
EARLY_FEATURES = ['QD_2', 'QD_max_minus_2', 'QD_100_minus_2', 'fade_slope_2_100', 'fade_intercept_2_100',
                  'chargetime_2_6', 'Tavg_mean', 'Tmax_mean', 'Tmin_mean', 'Tmax_max',
                  'IR_2', 'IR_min', 'IR_100_minus_2']

# 피처 세트 (DAY1 모델 설계 전략)
FEATURE_SETS = {
    'A_variance': ['log_var'],                                   # 베이스라인
    'B_deltaQ': ['log_var', 'log_abs_min', 'dQ_mean'],           # 메인 (배치 일관 피처)
    'C_full': ['log_var', 'log_abs_min', 'dQ_mean', 'dQ_skew', 'dQ_kurt',   # 대조군 (Q5 피처 전체, IR 제외)
               'QD_2', 'QD_max_minus_2', 'QD_100_minus_2', 'fade_slope_2_100', 'fade_intercept_2_100',
               'chargetime_2_6', 'Tavg_mean', 'Tmax_mean', 'Tmin_mean', 'Tmax_max'],
}


# =========================================================
# 1. 충전 정책
# =========================================================
def parse_policy(p):
    # '5.4C(40%)-3.6C' -> (5.4, 40, 3.6). 형식이 다르면 NaN ('-newstructure' 접미사는 무시됨)
    m = re.match(r'([\d.]+)C\((\d+)%\)-([\d.]+)C', str(p))
    return pd.Series(dict(zip(['C1', 'switch_SOC', 'C2'], map(float, m.groups())))) if m else \
           pd.Series({'C1': np.nan, 'switch_SOC': np.nan, 'C2': np.nan})


# =========================================================
# 2. ΔQ(V)
# =========================================================
def get_qdlin(c, n):
    q = c['Qdlin']
    if n in q:
        return q[n].astype(float), n
    k = min(q, key=lambda x: abs(x - n))      # 해당 사이클이 없으면 가장 가까운 사이클
    return q[k].astype(float), k


def deltaq_features(cell, cyc_a=CYC_A, cyc_b=CYC_B):
    """cell : run_all()['cells'][cell_id]"""
    if not cell['Qdlin']:   # Qdlin이 없는 셀 (제외 셀 포함 시)
        return dict.fromkeys(DELTAQ_FEATURES, np.nan)
    qa, _ = get_qdlin(cell, cyc_a)
    qb, _ = get_qdlin(cell, cyc_b)
    dq = qb - qa
    return {
        'dQ_min': dq.min(), 'dQ_mean': dq.mean(), 'dQ_var': dq.var(),
        'dQ_skew': stats.skew(dq), 'dQ_kurt': stats.kurtosis(dq),
        'log_var': np.log10(dq.var()),
        'log_abs_min': np.log10(np.abs(dq.min())),
    }


# =========================================================
# 3. 초기 사이클 summary 피처
# =========================================================
def early_cycle_features(g):
    """g : 셀 하나의 summary (run_all()['cells'][cell_id]['summary']). 이상치 행은 여기서 제외"""
    if 'is_outlier' in g.columns:
        g = g[~g['is_outlier']]
    g = g.sort_values('cycle')
    e = g[(g['cycle'] >= 2) & (g['cycle'] <= 100)]
    if len(e) < 2:          # 초기 사이클 데이터가 없는 셀 (제외 셀 포함 시)
        return dict.fromkeys(EARLY_FEATURES, np.nan)
    slope, intercept = np.polyfit(e['cycle'], e['QD'], 1)
    ir = e.loc[e['IR'] > 0, 'IR']
    return {
        'QD_2': e['QD'].iloc[0],
        'QD_max_minus_2': e['QD'].max() - e['QD'].iloc[0],
        'QD_100_minus_2': e['QD'].iloc[-1] - e['QD'].iloc[0],
        'fade_slope_2_100': slope,
        'fade_intercept_2_100': intercept,
        'chargetime_2_6': g.loc[(g['cycle'] >= 2) & (g['cycle'] <= 6), 'chargetime'].mean(),
        'Tavg_mean': e['Tavg'].mean(), 'Tmax_mean': e['Tmax'].mean(), 'Tmin_mean': e['Tmin'].mean(),
        'Tmax_max': e['Tmax'].max(),
        'IR_2': ir.iloc[0] if len(ir) else np.nan,
        'IR_min': ir.min() if len(ir) else np.nan,
        'IR_100_minus_2': (ir.iloc[-1] - ir.iloc[0]) if len(ir) else np.nan,
    }


# =========================================================
# 4. 셀 단위 피처 테이블
# =========================================================
def build_features(data, include_excluded=False):
    """data : run_all() 결과. 반환 : 셀 1행 피처 테이블"""
    ci = data['cell_info']
    if not include_excluded:
        ci = ci[~ci['excluded']]

    rows = []
    for r in ci.itertuples(index=False):
        cell = data['cells'][r.cell_id]
        rows.append({
            'cell_id': r.cell_id, 'batch': r.batch,
            'structure': 'new' if 'newstructure' in r.policy else 'old',
            'policy': r.policy, 'cycle_life': r.cycle_life,
            'log_life': np.log10(r.cycle_life),
            **deltaq_features(cell),
            **early_cycle_features(cell['summary']),
        })
    feat = pd.DataFrame(rows)
    return pd.concat([feat, feat['policy'].apply(parse_policy)], axis=1)


if __name__ == '__main__':
    feat = build_features(run_all())
    out = os.path.join(OUT_DIR, 'features.csv')
    feat.to_csv(out, index=False)
    print(f'\n피처 테이블 저장 : {out}  shape = {feat.shape}')
    print(feat.groupby('batch').size().rename('n_cells').to_string())
