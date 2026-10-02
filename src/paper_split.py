"""
DS Mini Project - DAY2 Step 5: 논문 조건 근사 재현 (H6)
======================================================
원논문(Severson 2019)처럼 Batch 1+2를 섞어서 학습했을 때의 MAPE를 우리 조건(Batch 1만 학습)과 나란히 비교
- 분할 : Batch 1+2를 cell_id 순(배치 -> 셀 번호 숫자 순)으로 정렬하고 교대로 train / test 배정
         (정렬 후 0, 2, 4, ... 번째 = train / 1, 3, 5, ... 번째 = test), Batch 3 = secondary test
- 모델 : Step 3 최종 모델 + A_linear, Step 3과 같은 탐색 방식 (train 셀에서 GroupKFold(5) GridSearchCV, groups=policy)
- 완전 재현이 아닌 근사 : 원논문은 b1c0~c4를 Batch 2 데이터로 이어붙인 셀을 포함했고 셀 인덱스도 다름
- 이 결과는 메인 성능표(model_comparison / model_performance)와 섞지 않음

사용법:
    python -m src.paper_split                          # 최종 모델 = src.train.FINAL_MODEL
    python -m src.paper_split --model B_elasticnet     # 최종 모델 직접 지정
"""
import os
import json
import argparse
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.features import FEATURE_SETS
from src.split import GROUP_COL, RESULTS_DIR, load_features
from src.train import (BASELINE_MODEL, FINAL_MODEL, MODEL_DIR, MODELS, N_FOLDS, SCORING, TARGET, TRAIN_BATCH,
                       build_pipeline, mape)


# =========================================================
# 설정
# =========================================================
PAPER_BATCHES = ['b1', 'b2']     # 섞어서 train / test로 나누는 배치
SECONDARY_BATCH = 'b3'
APPROX_NOTE = ('근사 재현: 원논문은 b1c0~c4를 Batch 2로 이어붙인 셀을 포함했고 셀 인덱스가 달라 '
               '분할이 원논문과 동일하지 않음')


def paper_split(feat_df):
    """Batch 1+2를 cell_id(배치, 셀 번호) 순으로 정렬 후 교대 배정. 반환 : (train, test, secondary)"""
    d = feat_df[feat_df['batch'].isin(PAPER_BATCHES)].copy()
    d['_cell_no'] = d['cell_id'].str.extract(r'c(\d+)$', expand=False).astype(int)
    d = d.sort_values(['batch', '_cell_no']).drop(columns='_cell_no').reset_index(drop=True)
    train, test = d.iloc[0::2], d.iloc[1::2]
    secondary = feat_df[feat_df['batch'] == SECONDARY_BATCH]
    assert set(train['cell_id']).isdisjoint(test['cell_id'])
    assert len(train) + len(test) == len(d)
    return train, test, secondary


def fit_search_paper(name, train):
    """Step 3 fit_search와 같은 탐색 (Batch 2 셀 포함을 허용하는 것만 다름)"""
    fset, make_model, grid = MODELS[name]
    X, y, groups = train[FEATURE_SETS[fset]], train[TARGET], train[GROUP_COL]
    cv = GroupKFold(n_splits=N_FOLDS)
    for tr_idx, va_idx in cv.split(X, y, groups):
        assert set(groups.iloc[tr_idx]).isdisjoint(groups.iloc[va_idx]), f'{name}: fold 정책 중복'
    param_grid = {f'regressor__model__{k}': list(v) for k, v in grid.items()}
    search = GridSearchCV(build_pipeline(make_model()), param_grid, cv=cv, scoring=SCORING)
    search.fit(X, y, groups=groups)
    return search


def run(feat, models):
    train, test, secondary = paper_split(feat)
    ours = pd.read_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv')).set_index('model')

    rows, preds = [], []
    for name in models:
        feats = FEATURE_SETS[MODELS[name][0]]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            search = fit_search_paper(name, train)
        est = search.best_estimator_     # refit=True : paper train 셀 전체로 fit
        best = {k.replace('regressor__model__', ''): v for k, v in search.best_params_.items()}

        p_tr, p_te, p_se = est.predict(train[feats]), est.predict(test[feats]), est.predict(secondary[feats])
        te_b = test['batch'].values
        rows.append({
            'condition': 'paper_style (Batch 1+2 교대 분할)', 'model': name,
            'n_train': len(train), 'n_test': len(test), 'n_secondary': len(secondary),
            'train_cv_mape': -search.best_score_ * 100,
            'train_fit_mape': mape(train[TARGET], p_tr),
            'test_mape': mape(test[TARGET], p_te),
            'test_mape_b1_cells': mape(test[TARGET][te_b == 'b1'], p_te[te_b == 'b1']),
            'test_mape_b2_cells': mape(test[TARGET][te_b == 'b2'], p_te[te_b == 'b2']),
            'secondary_test_mape': mape(secondary[TARGET], p_se),
            'best_params': json.dumps({k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                      for k, v in best.items()}),
            'n_warnings': len(w), 'note': APPROX_NOTE,
        })
        # 우리 조건 (Step 3 결과를 그대로 옮김) : train = Batch 1 학습 셀 CV, test = Batch 2, secondary = Batch 3
        o = ours.loc[name]
        b1_all = feat[feat['batch'] == TRAIN_BATCH]
        ours_fit = joblib.load(os.path.join(MODEL_DIR, f'{name}.joblib')).predict(b1_all[feats])
        rows.append({
            'condition': 'ours (Batch 1만 학습)', 'model': name,
            'n_train': 36, 'n_test': 39, 'n_secondary': 44,
            'train_cv_mape': o['train_mape'], 'train_fit_mape': mape(b1_all[TARGET], ours_fit),
            'test_mape': o['b2_mape'], 'test_mape_b1_cells': np.nan, 'test_mape_b2_cells': o['b2_mape'],
            'secondary_test_mape': o['b3_mape'], 'best_params': o['best_params'],
            'n_warnings': o['n_warnings'], 'note': 'Step 3 model_comparison.csv (train_cv = 학습 29셀 CV, train_fit = models/*.joblib의 Batch 1 36셀 in-sample)',
        })
        for d, s, p in [(train, 'paper_train', p_tr), (test, 'paper_test', p_te), (secondary, 'secondary', p_se)]:
            preds.append(d[['cell_id', 'batch', 'structure', 'policy']].assign(
                model=name, split=s, y_true=d[TARGET].values, y_pred=p,
                ape=np.abs(p - d[TARGET].values) / d[TARGET].values * 100))
    return train, test, secondary, pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, choices=list(MODELS),
                        help=f'최종 모델 (기본 : {FINAL_MODEL})')
    args = parser.parse_args()
    final_model = args.model or FINAL_MODEL
    models = list(dict.fromkeys([final_model, BASELINE_MODEL]))

    feat = load_features()
    train, test, secondary, comp, pred = run(feat, models)

    print(f'[논문식 분할] {APPROX_NOTE}')
    print(f'  최종 모델 : {final_model} / 기준 모델 : {BASELINE_MODEL}')
    for name, d in [('train', train), ('test', test)]:
        print(f'  {name:5s} : {len(d)}셀 {d["batch"].value_counts().sort_index().to_dict()}, '
              f'구조 {d["structure"].value_counts().sort_index().to_dict()}, '
              f'수명 {d[TARGET].min():.0f}~{d[TARGET].max():.0f} (median {d[TARGET].median():.0f})')
    print(f'  secondary (Batch 3) : {len(secondary)}셀')
    shared = test[GROUP_COL].isin(set(train[GROUP_COL]))
    print(f'  test 셀 중 같은 정책이 train에 있는 셀 : {shared.sum()} / {len(test)}')

    print('\n[예측값 범위 (사이클)]')
    print(pred.groupby(['model', 'split'])['y_pred'].agg(['min', 'max']).round(0).to_string())
    assert pred['y_pred'].between(100, 10000).all(), '예측값이 사이클 단위 범위를 벗어남'

    comp.to_csv(os.path.join(RESULTS_DIR, 'paper_split_comparison.csv'), index=False)
    pred.to_csv(os.path.join(RESULTS_DIR, 'paper_split_predictions.csv'), index=False)

    pd.set_option('display.width', 220)
    print('\n[우리 조건 vs 논문식 조건 (MAPE %)]')
    print(comp.drop(columns=['note', 'best_params']).round(2).to_string(index=False))
    print('\n[best params]')
    print(comp[['condition', 'model', 'best_params']].to_string(index=False))
    print('\n저장 완료 : results/paper_split_comparison.csv, paper_split_predictions.csv')
