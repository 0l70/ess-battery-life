"""
DS Mini Project - DAY2 Step 3·4: 파이프라인과 모델 비교, Hold-out 안정성
=====================================================================
1) 학습 셀(Batch 1 중 29셀)에서 GridSearchCV(GroupKFold(5), groups=policy) -> Train MAPE = best CV 평균
2) Hold-out(7셀) 예측 -> Valid MAPE
3) 같은 best 파라미터로 Batch 1 전체(36셀) 재학습 -> Batch 2·3 평가
4) (Step 4) seed 0~19 반복 분할마다 1)~2)를 반복 -> 모델별 Valid MAPE 평균 ± 표준편차 (모델 선택에는 쓰지 않음)
Batch 2·3은 학습, 하이퍼파라미터 탐색, 모델 선택에 사용하지 않음

파이프라인 : TransformedTargetRegressor(log10) [ SimpleImputer(median) -> StandardScaler -> 모델 ]
MAPE는 사이클 단위 (TransformedTargetRegressor가 예측을 10**y로 복원)

사용법:
    python -m src.train                    # 프로젝트 루트에서 실행 (Step 3 + Step 4)
    python -m src.train --step 3           # 모델 비교만
    python -m src.train --step 4           # Hold-out 반복 안정성만
"""
import os
import json
import argparse
import warnings
from functools import partial
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocess import BASE_DIR
from src.features import FEATURE_SETS
from src.split import (GROUP_COL, N_REPEATS, RANDOM_STATE, RESULTS_DIR, TRAIN_BATCH,
                       iter_repeated_splits, load_features, make_holdout_split)


# =========================================================
# 설정
# =========================================================
TARGET = 'cycle_life'
N_FOLDS = 5
SCORING = 'neg_mean_absolute_percentage_error'
PAPER_TARGET_MAPE = 9.1          # 노션 Target (Severson 2019)
SELECT_TOL = 1.0                 # Valid MAPE 최저 대비 1%p 이내면 더 단순한 모델
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FINAL_MODEL = 'C_elasticnet'     # 최종 모델 (사용자 확정 : 선택 규칙상 후보와 동일)
BASELINE_MODEL = 'A_linear'      # 기준 모델 (함께 보고)

EN_GRID = {'alpha': np.logspace(-4, 0, 9), 'l1_ratio': [0.1, 0.5, 0.9]}

# 이름: (피처 세트, 모델, 탐색 범위)  -- 순서 = 단순한 모델 순 (후보 선택 규칙의 동률 처리에 사용)
MODELS = {
    'A_linear': ('A_variance', lambda: LinearRegression(), {}),
    'B_ridge': ('B_deltaQ', lambda: Ridge(), {'alpha': np.logspace(-3, 2, 11)}),
    'B_elasticnet': ('B_deltaQ', lambda: ElasticNet(max_iter=50000), EN_GRID),
    'C_elasticnet': ('C_full', lambda: ElasticNet(max_iter=50000), EN_GRID),
    'B_rf': ('B_deltaQ', lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
             {'max_depth': [2, 3, None], 'min_samples_leaf': [2, 4]}),
    'B_gbr': ('B_deltaQ', lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
              {'n_estimators': [100, 300], 'max_depth': [1, 2], 'learning_rate': [0.05, 0.1]}),
}


# log10 타깃 역변환 = 10 ** y
# (lambda나 이 파일의 함수는 python -m 실행 시 __main__에 묶여 joblib 로드가 깨져서 numpy 함수로 정의)
pow10 = partial(np.power, 10.0)


def build_pipeline(model):
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler()),
                     ('model', model)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log10, inverse_func=pow10)


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def _assert_b1_only(df):
    assert (df['batch'] == TRAIN_BATCH).all(), 'Batch 2·3 셀이 fit 데이터에 포함됨'


# =========================================================
# 1. 학습 / 평가
# =========================================================
def fit_search(name, train):
    """학습 셀에서 정책 단위 GroupKFold 그리드 탐색 (refit=True -> best_estimator_는 학습 셀 전체로 fit)"""
    _assert_b1_only(train)
    fset, make_model, grid = MODELS[name]
    X, y, groups = train[FEATURE_SETS[fset]], train[TARGET], train[GROUP_COL]
    cv = GroupKFold(n_splits=N_FOLDS)
    for tr_idx, va_idx in cv.split(X, y, groups):
        assert set(groups.iloc[tr_idx]).isdisjoint(groups.iloc[va_idx]), f'{name}: fold 정책 중복'
    param_grid = {f'regressor__model__{k}': list(v) for k, v in grid.items()}
    search = GridSearchCV(build_pipeline(make_model()), param_grid, cv=cv, scoring=SCORING)
    search.fit(X, y, groups=groups)
    return search


def run_model(name, train, holdout, b1_all, tests):
    """반환 : (비교표 1행, 예측 DataFrame, Batch 1 전체로 재학습한 모델)"""
    fset = MODELS[name][0]
    feats = FEATURE_SETS[fset]
    search = fit_search(name, train)
    best = {k.replace('regressor__model__', ''): v for k, v in search.best_params_.items()}

    # Train : best CV 평균 MAPE + 셀별 out-of-fold 예측 (predictions.csv용)
    train_mape = -search.best_score_ * 100
    oof = cross_val_predict(build_pipeline(MODELS[name][1]()).set_params(**search.best_params_),
                            train[feats], train[TARGET], groups=train[GROUP_COL], cv=GroupKFold(n_splits=N_FOLDS))
    # Valid : 학습 셀 29개로 fit한 best_estimator_로 Hold-out 예측
    valid_pred = search.best_estimator_.predict(holdout[feats])

    # Test : 같은 best 파라미터로 Batch 1 전체(36셀) 재학습
    _assert_b1_only(b1_all)
    final = build_pipeline(MODELS[name][1]()).set_params(**search.best_params_)
    final.fit(b1_all[feats], b1_all[TARGET])

    preds = [(train, 'train_cv', oof), (holdout, 'holdout', valid_pred)]
    row = {'model': name, 'feature_set': fset, 'n_features': len(feats),
           'train_mape': train_mape, 'valid_mape': mape(holdout[TARGET], valid_pred)}
    for b, d in tests.items():
        p = final.predict(d[feats])
        preds.append((d, b, p))
        row[f'{b}_mape'] = mape(d[TARGET], p)
    row['best_params'] = json.dumps({k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                     for k, v in best.items()})

    pred_df = pd.concat([d[['cell_id', 'batch', 'structure']].assign(
        model=name, split=s, y_true=d[TARGET].values, y_pred=p,
        ape=np.abs(p - d[TARGET].values) / d[TARGET].values * 100) for d, s, p in preds], ignore_index=True)
    return row, pred_df, final


# =========================================================
# 2. 후보 선택 (규칙 표시만, 최종 선택은 사용자)
# =========================================================
def mark_candidate(comp):
    """Valid MAPE 최저 모델 대비 SELECT_TOL(%p) 이내 모델 중 MODELS 순서상 가장 단순한 모델"""
    best_valid = comp['valid_mape'].min()
    comp['within_1pp'] = comp['valid_mape'] <= best_valid + SELECT_TOL
    comp['candidate'] = False
    comp.loc[comp.index[comp['within_1pp']][0], 'candidate'] = True
    return comp


def notion_table(row):
    """노션 리포팅 포맷. Gap은 (+)가 뒤 항목에서 성능이 나빠짐 (MAPE 증가)"""
    tr, va, b2, b3 = row['train_mape'], row['valid_mape'], row['b2_mape'], row['b3_mape']
    return pd.DataFrame([
        ('Train (Batch 1 CV)', tr, ''),
        ('Valid (Batch 1 Hold-out)', va, ''),
        ('Test (Batch 2)', b2, ''),
        ('Gap (Train − Valid)', va - tr, 'Valid MAPE − Train MAPE'),
        ('Gap (Valid − Test)', b2 - va, 'Batch 2 MAPE − Valid MAPE'),
        ('Gap (Target − Test)', b2 - PAPER_TARGET_MAPE, f'Batch 2 MAPE − Target {PAPER_TARGET_MAPE}'),
        ('Test (Batch 3)', b3, ''),
        ('Gap (Batch 2 − Batch 3)', b3 - b2, 'Batch 3 MAPE − Batch 2 MAPE'),
    ], columns=['구분', 'MAPE (%)', '계산식'])


def compare_models(feat):
    """Step 3 : 모델 6종 비교 -> model_comparison / model_performance / predictions.csv, models/*.joblib"""
    train, holdout = make_holdout_split(feat)
    b1_all = feat[feat['batch'] == TRAIN_BATCH]
    tests = {b: feat[feat['batch'] == b] for b in ['b2', 'b3']}
    print(f'학습 {len(train)}셀 / Hold-out {len(holdout)}셀 / Batch 1 전체 {len(b1_all)}셀 / '
          f'Batch 2 {len(tests["b2"])}셀 / Batch 3 {len(tests["b3"])}셀')

    os.makedirs(MODEL_DIR, exist_ok=True)
    rows, preds = [], []
    for name in MODELS:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            row, pred_df, final = run_model(name, train, holdout, b1_all, tests)
        row['n_warnings'] = len(w)
        if w:
            print(f'  [{name}] 경고 {len(w)}건 : {sorted({type(x.message).__name__ for x in w})}')
        rows.append(row)
        preds.append(pred_df)
        joblib.dump(final, os.path.join(MODEL_DIR, f'{name}.joblib'))
        print(f'  {name} 완료')

    comp = mark_candidate(pd.DataFrame(rows))
    pred_all = pd.concat(preds, ignore_index=True)

    # 검증 : 예측값이 사이클 단위인지
    print('\n[예측값 범위 (사이클)]')
    print(pred_all.groupby(['model', 'split'])['y_pred'].agg(['min', 'max']).unstack('split').round(0).to_string())
    assert pred_all['y_pred'].between(100, 10000).all(), '예측값이 사이클 단위 범위를 벗어남'

    cols = ['model', 'feature_set', 'n_features', 'train_mape', 'valid_mape', 'b2_mape', 'b3_mape',
            'within_1pp', 'candidate', 'best_params', 'n_warnings']
    comp = comp[cols]
    cand = comp.loc[comp['candidate']].iloc[0]
    if cand['model'] != FINAL_MODEL:
        print(f'[주의] 규칙상 후보({cand["model"]})와 FINAL_MODEL({FINAL_MODEL})이 다름')
    # 노션 포맷 : 최종 모델 + 기준 모델을 열로 나란히
    by_name = comp.set_index('model')
    tables = [notion_table(by_name.loc[m]) for m in (FINAL_MODEL, BASELINE_MODEL)]
    perf = tables[0][['구분']].copy()
    perf[f'{FINAL_MODEL} (최종)'] = tables[0]['MAPE (%)']
    perf[f'{BASELINE_MODEL} (기준)'] = tables[1]['MAPE (%)']
    perf['계산식'] = tables[0]['계산식']

    comp.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
    perf.to_csv(os.path.join(RESULTS_DIR, 'model_performance.csv'), index=False)
    pred_all.to_csv(os.path.join(RESULTS_DIR, 'predictions.csv'), index=False)

    pd.set_option('display.width', 200)
    print('\n[모델 비교 (MAPE %)]')
    print(comp.drop(columns='best_params').round(2).to_string(index=False))
    print('\n[best params]')
    print(comp[['model', 'best_params']].to_string(index=False))
    print(f'\n[규칙상 후보 : {cand["model"]}] (Valid MAPE 최저 대비 {SELECT_TOL}%p 이내 중 가장 단순한 모델)')
    print(f'[성능표] 최종 {FINAL_MODEL} / 기준 {BASELINE_MODEL}')
    print(perf.round(2).to_string(index=False))
    print(f'\n저장 완료 : results/model_comparison.csv, model_performance.csv, predictions.csv, models/*.joblib')
    return comp


# =========================================================
# 3. Hold-out 안정성 (Step 4) -- 모델 선택에는 쓰지 않음
# =========================================================
def holdout_repeats(feat, n_repeats=N_REPEATS):
    """seed 0 ~ n_repeats-1 반복 분할마다 Step 3과 같은 절차(학습 셀 GridSearchCV -> Hold-out 예측)로 Valid MAPE 계산
    반환 : (seed별 결과, 모델별 요약)"""
    rows = []
    for seed, train, holdout in iter_repeated_splits(feat, n_repeats=n_repeats):
        _assert_b1_only(train)
        _assert_b1_only(holdout)
        for name in MODELS:
            feats = FEATURE_SETS[MODELS[name][0]]
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                search = fit_search(name, train)
            best = {k.replace('regressor__model__', ''): v for k, v in search.best_params_.items()}
            rows.append({'seed': seed, 'model': name, 'n_train': len(train), 'n_holdout': len(holdout),
                         'train_mape': -search.best_score_ * 100,
                         'valid_mape': mape(holdout[TARGET], search.best_estimator_.predict(holdout[feats])),
                         'best_params': json.dumps({k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                    for k, v in best.items()}),
                         'n_warnings': len(w)})
        print(f'  seed {seed} 완료 (Hold-out {len(holdout)}셀)')
    raw = pd.DataFrame(rows)

    g = raw.groupby('model', sort=False)['valid_mape']
    summ = pd.DataFrame({'valid_mape_mean': g.mean(), 'valid_mape_std': g.std(),
                         'valid_mape_min': g.min(), 'valid_mape_median': g.median(), 'valid_mape_max': g.max(),
                         'train_mape_mean': raw.groupby('model', sort=False)['train_mape'].mean(),
                         'n_repeats': g.size()})
    # seed별 Valid MAPE 순위 (1 = 최저)
    rank = raw.pivot(index='seed', columns='model', values='valid_mape').rank(axis=1)
    summ['rank_mean'] = rank.mean()
    summ['n_rank1'] = (rank == 1).sum()
    return raw, summ.reset_index()


def run_holdout_repeats(feat):
    raw, summ = holdout_repeats(feat)
    raw.to_csv(os.path.join(RESULTS_DIR, 'holdout_repeats_raw.csv'), index=False)
    summ.to_csv(os.path.join(RESULTS_DIR, 'holdout_repeats.csv'), index=False)
    print(f'\n[Hold-out 반복 분할 {N_REPEATS}회 : Valid MAPE (%)] std = 표본 표준편차 (ddof=1)')
    print(summ.round(2).to_string(index=False))
    print(f'경고 합계 : {raw["n_warnings"].sum()}건')
    print('\n저장 완료 : results/holdout_repeats.csv, holdout_repeats_raw.csv')
    return raw, summ


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['3', '4', 'all'], default='all',
                        help='3 = 모델 비교, 4 = Hold-out 반복 안정성, all = 둘 다 (기본)')
    args = parser.parse_args()
    feat = load_features()
    if args.step in ('3', 'all'):
        compare_models(feat)
    if args.step in ('4', 'all'):
        run_holdout_repeats(feat)
