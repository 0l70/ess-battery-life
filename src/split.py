"""
DS Mini Project - DAY2 Step 2: 데이터 분할
==========================================
Batch 1을 충전 정책 단위로 학습 / Hold-out 분리 (같은 정책이 양쪽에 들어가지 않음)
Batch 2·3은 분할에 사용하지 않음 (테스트 전용)
1) make_holdout_split()   : GroupShuffleSplit 1회 (random_state=42)
2) iter_repeated_splits() : seed 0~19 반복 분할 (Step 4 Hold-out 안정성용)

사용법:
    python -m src.split                    # 프로젝트 루트에서 실행 -> results/split.json

    from src.split import load_features, make_holdout_split
    feat = load_features()
    train, holdout = make_holdout_split(feat)
"""
import os
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.preprocess import BASE_DIR, OUT_DIR, run_all
from src.features import build_features


# =========================================================
# 설정
# =========================================================
TRAIN_BATCH = 'b1'
GROUP_COL = 'policy'
TEST_SIZE = 0.2           # GroupShuffleSplit 기준 = 정책(그룹) 수의 비율
RANDOM_STATE = 42
N_REPEATS = 20
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def load_features():
    """processed/features.csv (python -m src.features 결과). 없으면 새로 계산"""
    path = os.path.join(OUT_DIR, 'features.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return build_features(run_all())


# =========================================================
# 1. 분할
# =========================================================
def make_holdout_split(feat_df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Batch 1을 정책 단위로 학습 / Hold-out 분리. 반환 : (train_df, holdout_df)"""
    b1 = feat_df[feat_df['batch'] == TRAIN_BATCH].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, ho_idx = next(gss.split(b1, groups=b1[GROUP_COL]))
    train, holdout = b1.iloc[tr_idx], b1.iloc[ho_idx]
    check_split(b1, train, holdout)
    return train, holdout


def iter_repeated_splits(feat_df, n_repeats=N_REPEATS, test_size=TEST_SIZE):
    """seed 0 ~ n_repeats-1 반복 분할. (seed, train_df, holdout_df)를 순서대로 반환"""
    for seed in range(n_repeats):
        train, holdout = make_holdout_split(feat_df, test_size=test_size, random_state=seed)
        yield seed, train, holdout


def check_split(b1, train, holdout):
    """정책 비중복, 셀 중복 없음, 셀 수 합 = Batch 1 전체"""
    assert set(train[GROUP_COL]).isdisjoint(holdout[GROUP_COL]), '정책이 학습과 Hold-out에 중복됨'
    assert set(train['cell_id']).isdisjoint(holdout['cell_id']), '셀이 학습과 Hold-out에 중복됨'
    assert len(train) + len(holdout) == len(b1), f'셀 수 합 {len(train) + len(holdout)} != {len(b1)}'
    assert (train['batch'] == TRAIN_BATCH).all() and (holdout['batch'] == TRAIN_BATCH).all()


# =========================================================
# 2. 요약
# =========================================================
def split_summary(train, holdout):
    rows = {}
    for name, d in [('train', train), ('holdout', holdout)]:
        life = d['cycle_life']
        rows[name] = {'n_cells': len(d), 'n_policies': d[GROUP_COL].nunique(),
                      'life_min': life.min(), 'life_median': life.median(), 'life_max': life.max()}
    return pd.DataFrame(rows).T.astype({'n_cells': int, 'n_policies': int})


if __name__ == '__main__':
    feat = load_features()
    b1 = feat[feat['batch'] == TRAIN_BATCH]
    train, holdout = make_holdout_split(feat)
    assert len(train) + len(holdout) == 36, f'Batch 1 셀 수 합 {len(train) + len(holdout)} != 36'

    summ = split_summary(train, holdout)
    print(f'[Batch 1 분할] 정책 단위 GroupShuffleSplit (test_size={TEST_SIZE}, random_state={RANDOM_STATE})')
    print(f'  Batch 1 : 셀 {len(b1)}개, 정책 {b1[GROUP_COL].nunique()}개')
    print(summ.to_string())
    print('\n[Hold-out 셀]')
    print(holdout[['cell_id', 'policy', 'cycle_life']].sort_values('cell_id').to_string(index=False))

    repeats = []
    for seed, tr, ho in iter_repeated_splits(feat):
        repeats.append({'seed': seed, 'n_train': len(tr), 'n_holdout': len(ho),
                        'holdout_cells': sorted(ho['cell_id']),
                        'holdout_policies': sorted(ho[GROUP_COL].unique())})
    rep = pd.DataFrame(repeats)
    print(f'\n[반복 분할 seed 0~{N_REPEATS - 1}] Hold-out 셀 수 : '
          f'min {rep.n_holdout.min()} / median {rep.n_holdout.median():.0f} / max {rep.n_holdout.max()}, '
          f'서로 다른 Hold-out 조합 {rep.holdout_cells.map(tuple).nunique()}개')

    out = {
        'method': 'GroupShuffleSplit on Batch 1, groups = policy',
        'train_batch': TRAIN_BATCH, 'group_col': GROUP_COL,
        'test_size': TEST_SIZE, 'random_state': RANDOM_STATE,
        'train_cells': sorted(train['cell_id']),
        'holdout_cells': sorted(holdout['cell_id']),
        'train_policies': sorted(train[GROUP_COL].unique()),
        'holdout_policies': sorted(holdout[GROUP_COL].unique()),
        'summary': summ.to_dict(orient='index'),
        'repeats': repeats,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'split.json')
    with open(path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f'\n저장 완료 : {path}')
