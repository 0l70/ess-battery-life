"""
DS Mini Project - DAY2 Step 6: 오류 분석 (H5)
=============================================
Step 3 결과(results/model_comparison.csv, predictions.csv)만 읽어서 표·그래프 생성 (재학습 없음)
- 표 : 모델 비교, Batch 2 구조별 MAPE, Batch 3 노이즈 셀 포함/제외 MAPE, 오차 상위 10셀
- 계수표 (선형 모델만) : 계수, 배치별 log 수명 상관, Step 4 반복 분할 재학습 시 선택 횟수 -> results/<model>_coef.csv
- 그래프 (results/figures/D2_*.png) : 예측 vs 실제, log 잔차 vs log_var, 모델별 MAPE 막대

사용법:
    python -m src.evaluate                         # 최종 모델 = src.train.FINAL_MODEL
    python -m src.evaluate --model A_linear        # 최종 모델 직접 지정
"""
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.split import RESULTS_DIR, load_features
from src.train import BASELINE_MODEL, FINAL_MODEL, MODEL_DIR, TRAIN_BATCH


# =========================================================
# 설정
# =========================================================
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
NOISY_B3 = ['b3c2', 'b3c37', 'b3c42', 'b3c43']     # 원논문 노이즈 채널 (01_EDA Q3과 동일)
TEST_SPLITS = ['b2', 'b3']
BATCH_COLOR = {'b1': '#1f77b4', 'b2': '#ff7f0e', 'b3': '#2ca02c'}
BATCH_LABEL = {'b1': 'Batch 1 (CV / Hold-out)', 'b2': 'Batch 2 (test)', 'b3': 'Batch 3 (extra)'}


def load_results():
    """반환 : (모델 비교표, 예측 + log_var/policy). 예측의 split = train_cv / holdout / b2 / b3"""
    comp = pd.read_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'))
    pred = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions.csv'))
    feat = load_features()[['cell_id', 'policy', 'log_var']]
    pred = pred.merge(feat, on='cell_id', how='left')
    assert pred['log_var'].notna().all()
    pred['log_resid'] = np.log10(pred['y_pred']) - np.log10(pred['y_true'])   # (+) = 과대예측
    return comp, pred


# =========================================================
# 1. 표
# =========================================================
def model_table(comp):
    return comp[['model', 'feature_set', 'train_mape', 'valid_mape', 'b2_mape', 'b3_mape', 'candidate']]


def structure_table(pred):
    """Batch 2 구조별 MAPE (모델 × old/new), 과대예측 셀 수 포함"""
    d = pred[pred['split'] == 'b2']
    g = d.groupby(['model', 'structure'], sort=False)
    return pd.DataFrame({'n_cells': g.size(), 'mape': g['ape'].mean(),
                         'median_log_resid': g['log_resid'].median(),
                         'n_over': g['log_resid'].apply(lambda s: int((s > 0).sum()))}).reset_index()


def noise_table(pred):
    """Batch 3 노이즈 셀 포함 / 제외 MAPE (모델별)"""
    d = pred[pred['split'] == 'b3']
    rows = []
    for m, g in d.groupby('model', sort=False):
        noisy = g['cell_id'].isin(NOISY_B3)
        rows.append({'model': m, 'mape_all': g['ape'].mean(), 'n_all': len(g),
                     'mape_excl_noisy': g.loc[~noisy, 'ape'].mean(), 'n_excl_noisy': int((~noisy).sum()),
                     'mape_noisy_only': g.loc[noisy, 'ape'].mean(), 'n_noisy': int(noisy.sum())})
    return pd.DataFrame(rows)


def noisy_cells(pred, model):
    d = pred[(pred['model'] == model) & pred['cell_id'].isin(NOISY_B3)]
    return d[['cell_id', 'y_true', 'y_pred', 'ape', 'log_var']].sort_values('cell_id').reset_index(drop=True)


def top_errors(pred, model, n=10, splits=TEST_SPLITS):
    """테스트 배치(기본 Batch 2·3)에서 APE 상위 n셀"""
    d = pred[(pred['model'] == model) & pred['split'].isin(splits)]
    cols = ['cell_id', 'batch', 'structure', 'policy', 'y_true', 'y_pred', 'ape', 'log_var']
    out = d.sort_values('ape', ascending=False).head(n)[cols].reset_index(drop=True)
    out['over_under'] = np.where(out['y_pred'] > out['y_true'], 'over', 'under')
    return out


def coef_table(model=FINAL_MODEL, feat=None):
    """선형 모델 계수 (Batch 1 전체 fit, models/*.joblib) + 피처별 배치별 log 수명 상관
    + Step 4 반복 분할(seed별 best params)로 재학습했을 때 계수 선택 횟수·부호"""
    from scipy import stats
    from src.features import FEATURE_SETS
    from src.split import iter_repeated_splits
    from src.train import MODELS, TARGET, build_pipeline

    feat = load_features() if feat is None else feat
    fs = FEATURE_SETS[MODELS[model][0]]
    pipe = joblib.load(os.path.join(MODEL_DIR, f'{model}.joblib')).regressor_
    coef, sc = pipe.named_steps['model'].coef_, pipe.named_steps['scaler']

    r = pd.DataFrame({b: [stats.pearsonr(g[c], np.log10(g[TARGET]))[0] for c in fs]
                      for b, g in feat.groupby('batch')}, index=fs)
    tab = pd.DataFrame({'coef(표준화)': coef, '원단위 계수': coef / sc.scale_}, index=fs).join(r.add_prefix('r_'))
    tab['부호 일관(r)'] = np.where((np.sign(r).nunique(axis=1) == 1) & (r.abs().min(axis=1) >= 0.3), '일관', '')
    tab['|coef| 순위'] = tab['coef(표준화)'].abs().rank(ascending=False).astype(int)

    raw = pd.read_csv(os.path.join(RESULTS_DIR, 'holdout_repeats_raw.csv'))
    raw = raw[raw['model'] == model].set_index('seed')
    sel = []
    for seed, tr, _ in iter_repeated_splits(feat):
        bp = {f'regressor__model__{k}': v for k, v in json.loads(raw.loc[seed, 'best_params']).items()}
        est = build_pipeline(MODELS[model][1]()).set_params(**bp).fit(tr[fs], tr[TARGET])
        sel.append(est.regressor_.named_steps['model'].coef_)
    sel = pd.DataFrame(sel, columns=fs)
    n = len(sel)
    stab = pd.DataFrame({f'|coef|>1e-3 횟수({n})': (sel.abs() > 1e-3).sum(), '+부호': (sel > 1e-3).sum(),
                         '-부호': (sel < -1e-3).sum(), 'coef 중앙값': sel.median()})
    return tab.join(stab).sort_values('|coef| 순위')


def train_logvar_range(feat=None):
    """최종 모델 학습 데이터(Batch 1 전체 36셀)의 log_var 범위"""
    feat = load_features() if feat is None else feat
    lv = feat.loc[feat['batch'] == TRAIN_BATCH, 'log_var']
    return lv.min(), lv.max()


# =========================================================
# 2. 그래프 (기본 matplotlib)
# =========================================================
def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=150, bbox_inches='tight')


def plot_pred_vs_true(pred, model):
    """예측 vs 실제 (log 축, y=x). Batch 1은 out-of-fold(train_cv) + Hold-out 예측"""
    d = pred[pred['model'] == model]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for b, g in d.groupby('batch'):
        ax.scatter(g['y_true'], g['y_pred'], s=25, alpha=0.75, color=BATCH_COLOR[b],
                   label=f'{BATCH_LABEL[b]} (n={len(g)})')
    lim = [min(d['y_true'].min(), d['y_pred'].min()) * 0.9, max(d['y_true'].max(), d['y_pred'].max()) * 1.1]
    ax.plot(lim, lim, 'k--', lw=1, label='y = x')
    ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim, xlabel='Actual cycle life', ylabel='Predicted cycle life',
           title=f'Predicted vs Actual ({model})')
    ax.legend(fontsize=9)
    _save(fig, 'D2_pred_vs_actual')
    return fig


def plot_resid_vs_logvar(pred, model, train_range):
    """log 잔차(log10 예측 − log10 실제) vs log_var, 학습 log_var 범위 음영"""
    d = pred[pred['model'] == model]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(*train_range, color='gray', alpha=0.15, label='Batch 1 log_var range (train)')
    for b, g in d.groupby('batch'):
        ax.scatter(g['log_var'], g['log_resid'], s=25, alpha=0.75, color=BATCH_COLOR[b], label=BATCH_LABEL[b])
    ax.axhline(0, color='k', lw=1)
    ax.set(xlabel='log10(var ΔQ)', ylabel='log10(pred) − log10(actual)  (+ = over-prediction)',
           title=f'Log residual vs log_var ({model})')
    ax.legend(fontsize=9)
    _save(fig, 'D2_resid_vs_logvar')
    return fig


def plot_mape_bars(comp):
    """모델별 Valid / Batch 2 / Batch 3 MAPE"""
    cols = [('valid_mape', 'Valid (Hold-out)'), ('b2_mape', 'Batch 2'), ('b3_mape', 'Batch 3')]
    x = np.arange(len(comp))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, (c, lab) in enumerate(cols):
        bars = ax.bar(x + (k - 1) * w, comp[c], w, label=lab)
        ax.bar_label(bars, fmt='%.1f', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(comp['model'])
    ax.set(ylabel='MAPE (%)', title='MAPE by model', ylim=(0, comp[[c for c, _ in cols]].max().max() * 1.2))
    ax.legend(ncol=3, loc='upper left')
    _save(fig, 'D2_mape_by_model')
    return fig


# =========================================================
# 3. 실행
# =========================================================
def run_all_tables(model=None):
    comp, pred = load_results()
    model = model or FINAL_MODEL
    tables = {
        'model': model_table(comp),
        'structure': structure_table(pred),
        'noise': noise_table(pred),
        'noisy_cells': noisy_cells(pred, model),
        'top10': top_errors(pred, model),
    }
    return model, comp, pred, tables


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help=f'최종 모델 (기본 : {FINAL_MODEL})')
    args = parser.parse_args()

    model, comp, pred, t = run_all_tables(args.model)
    assert model in set(comp['model']), f'알 수 없는 모델 : {model}'
    rng = train_logvar_range()

    t['structure'].to_csv(os.path.join(RESULTS_DIR, 'error_b2_structure.csv'), index=False)
    t['noise'].to_csv(os.path.join(RESULTS_DIR, 'error_b3_noise.csv'), index=False)
    t['top10'].assign(model=model).to_csv(os.path.join(RESULTS_DIR, 'error_top10.csv'), index=False)
    for f in [plot_pred_vs_true(pred, model), plot_resid_vs_logvar(pred, model, rng), plot_mape_bars(comp)]:
        plt.close(f)
    coef = None
    if hasattr(joblib.load(os.path.join(MODEL_DIR, f'{model}.joblib')).regressor_.named_steps['model'], 'coef_'):
        coef = coef_table(model)
        coef.to_csv(os.path.join(RESULTS_DIR, f'{model.lower()}_coef.csv'))

    pd.set_option('display.width', 200)
    print(f'최종 모델 : {model} / 기준 모델 : {BASELINE_MODEL}')
    print('\n[모델 비교 (MAPE %)]')
    print(t['model'].round(2).to_string(index=False))
    print('\n[Batch 2 구조별 MAPE (%)]  log_resid = log10(pred) − log10(actual), n_over = 과대예측 셀 수')
    print(t['structure'].round(3).to_string(index=False))
    print(f'\n[Batch 3 노이즈 셀 {NOISY_B3} 포함 / 제외 MAPE (%)]')
    print(t['noise'].round(2).to_string(index=False))
    print(f'\n[노이즈 셀 예측 ({model})]')
    print(t['noisy_cells'].round(3).to_string(index=False))
    print(f'\n[오차 상위 10셀 ({model}, Batch 2·3)]  Batch 1 학습 log_var 범위 : {rng[0]:.3f} ~ {rng[1]:.3f}')
    print(t['top10'].round(3).to_string(index=False))
    if coef is not None:
        print(f'\n[{model} 계수 (Batch 1 전체 fit) + 배치별 log 수명 상관 + 반복 분할 재학습 시 선택 횟수]')
        print(coef.round(4).to_string())
    print('\n저장 완료 : results/error_b2_structure.csv, error_b3_noise.csv, error_top10.csv, '
          + (f'{model.lower()}_coef.csv, ' if coef is not None else '')
          + 'figures/D2_pred_vs_actual.png, D2_resid_vs_logvar.png, D2_mape_by_model.png')
