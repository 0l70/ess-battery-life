"""
DS Mini Project - DAY1 0단계: 데이터 준비
=========================================
1) 3개 배치 .mat 로드 -> 필요한 정보만 추출 -> 배치별 캐시 저장 (.mat는 한 번만 연다)
2) Batch1 <-> Batch2 이어지는 셀 검증 후 병합
3) 이상치 플래그 (첫 사이클 0값, QD/chargetime/IR/온도 스파이크)
4) EOL(0.88Ah) 도달 여부로 cycle_life 재검증
5) 제외 셀 표시 (삭제하지 않고 플래그만) -> 최종 pickle 저장

사용법:
    python -m src.preprocess               # 프로젝트 루트에서 실행

    from src.preprocess import run_all     # 노트북 (sys.path에 프로젝트 루트 추가)
    data = run_all()                       # 최초 1회: .mat 로드 (수 분 소요)
    data = run_all()                       # 이후: 캐시에서 바로 로드
    cell_info, summary = data['cell_info'], data['summary']
"""
import os
import gc
import time
import logging
import pickle
import numpy as np
import pandas as pd

try:
    import mat73
except ImportError:
    mat73 = None
import scipy.io as sio


# =========================================================
# 설정
# =========================================================
# 프로젝트 구조:  DS Mini Project/
#                   ├── data/          (.mat 4개)
#                   ├── processed/     (자동 생성: 캐시 & 결과)
#                   └── src/preprocess.py
# 이 파일 위치(src/)의 상위 폴더 = 프로젝트 루트 기준이라 노트북을 어디서 실행해도 경로가 안 깨짐
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'processed')

BATCH_FILES = {
    'b1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'b2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'b3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}

NOMINAL_CAPACITY = 1.1                    # 공칭 용량 (Ah)
EOL_CAPACITY = 0.8 * NOMINAL_CAPACITY     # EOL 기준 0.88 Ah (원논문 정의)
EOL_TOL = 0.005                           # EOL 판정 허용오차 (Ah)
MAX_CYCLE_DETAIL = 100                    # Qdlin/Tdlin을 저장할 최대 사이클
APPLY_PAPER_EXCLUDE = False               # 원논문 제외 목록 적용 여부 (인덱스 검증 전엔 False)

# summary 원본 키 -> 사용할 컬럼명
SUMMARY_COLS = {
    'cycle': 'cycle', 'QDischarge': 'QD', 'QCharge': 'QC', 'IR': 'IR',
    'Tavg': 'Tavg', 'Tmax': 'Tmax', 'Tmin': 'Tmin', 'chargetime': 'chargetime',
}

# [참고용] 원논문 공개 코드의 매핑 (Kaggle 파일은 batch2 셀 수가 달라 인덱스 불일치)
# -> 실제 병합은 find_continuation()이 정책 + 용량 연속성으로 데이터에서 직접 찾음
B1_B2_CONTINUATION = {   # batch1 셀: (이어지는 batch2 셀, 원논문 add_len)
    'b1c0': ('b2c7', 662),
    'b1c1': ('b2c8', 981),
    'b1c2': ('b2c9', 1060),
    'b1c3': ('b2c15', 208),
    'b1c4': ('b2c16', 482),
}

# [확인 필요] 원논문에서 제외한 셀 (기억 기반)
# EOL 미도달 셀은 eol_check()에서 데이터로도 자동 판정됨
EXCLUDE_CELLS = {
    'b1c8': '원논문 제외 (EOL 미도달)',
    'b1c10': '원논문 제외 (EOL 미도달)',
    'b1c12': '원논문 제외 (EOL 미도달)',
    'b1c13': '원논문 제외 (EOL 미도달)',
    'b1c22': '원논문 제외 (EOL 미도달)',
    'b3c2': '원논문 제외 (노이즈 채널)',
    'b3c23': '원논문 제외 (노이즈 채널)',
    'b3c32': '원논문 제외 (노이즈 채널)',
    'b3c37': '원논문 제외 (노이즈 채널)',
    'b3c42': '원논문 제외 (노이즈 채널)',
    'b3c43': '원논문 제외 (노이즈 채널)',
}

# 이상치 판정 기준 (rolling median 대비 편차)
OUTLIER_RULES = {
    'QD': ('abs', 0.05),          # 0.05 Ah 이상 튀면 이상치
    'QC': ('abs', 0.05),
    'chargetime': ('rel', 0.30),  # 30% 이상 튀면 이상치
    'IR': ('rel', 0.30),
    'Tavg': ('abs', 5.0),         # 5도 이상 튀면 이상치
    'Tmax': ('abs', 5.0),
}
ROLL_WINDOW = 11


# =========================================================
# 1. 로드 & 추출
# =========================================================
def load_mat(path):
    """MATLAB v7.3(HDF5)는 mat73, 그 이하는 scipy로 로드
    - barcode/channel_id 같은 MATLAB string 필드는 mat73이 지원하지 않아
      'MATLAB type not supported: string' 로그를 대량으로 찍음 -> 무해하므로 숨김
      (해당 필드는 None으로 들어오고, 분석에 쓰지 않음)
    """
    if mat73 is not None:
        logging.disable(logging.ERROR)
        try:
            return mat73.loadmat(path)
        except Exception as e:
            print(f'  mat73 로드 실패 -> scipy로 재시도: {e}')
        finally:
            logging.disable(logging.NOTSET)
    return sio.loadmat(path, simplify_cells=True)


def _to_records(d):
    """mat73의 dict-of-lists 구조를 list-of-dicts로 변환"""
    if isinstance(d, dict):
        keys = list(d.keys())
        n = len(d[keys[0]])
        return [{k: d[k][i] for k in keys} for i in range(n)]
    return list(d)


def _scalar(x):
    try:
        return float(np.asarray(x, dtype=float).squeeze())
    except (TypeError, ValueError):
        return np.nan


def extract_batch(batch_key, path, max_cycle=MAX_CYCLE_DETAIL):
    """
    배치 하나를 로드해서 셀별 필요한 정보만 추출
    반환: {cell_id: {batch, cell_id, policy, cycle_life, summary(DataFrame),
                     Qdlin{cycle: array}, Tdlin{cycle: array}, Vdlin}}
    """
    print(f'[{batch_key}] 로딩 중... {os.path.basename(path)} (수 분 소요)', flush=True)
    t0 = time.time()
    mat = load_mat(path)
    print(f'[{batch_key}] 로드 완료 ({time.time() - t0:.0f}초), 추출 시작', flush=True)
    raw_cells = _to_records(mat['batch'])
    del mat
    gc.collect()

    cells = {}
    for i, cell in enumerate(raw_cells):
        cid = f'{batch_key}c{i}'
        if i % 10 == 0:
            print(f'  추출 {i}/{len(raw_cells)}', flush=True)

        # summary -> DataFrame
        summ = cell['summary']
        sdf = pd.DataFrame({
            new: np.asarray(summ[old], dtype=float).ravel()
            for old, new in SUMMARY_COLS.items() if old in summ
        })
        if 'cycle' not in sdf.columns:
            sdf.insert(0, 'cycle', np.arange(1, len(sdf) + 1))
        sdf['cycle'] = sdf['cycle'].astype(int)

        # cycles -> 초기 max_cycle 사이클의 Qdlin, Tdlin만 저장
        # (cycles[j]와 summary j번째 행이 같은 사이클이라는 가정 -> 길이 불일치 시 경고)
        cycles = _to_records(cell['cycles'])
        if len(cycles) != len(sdf):
            print(f'  [경고] {cid}: cycles 길이({len(cycles)}) != summary 길이({len(sdf)})')

        qdlin, tdlin = {}, {}
        for j, cyc in enumerate(cycles):
            cyc_num = int(sdf['cycle'].iloc[j]) if j < len(sdf) else j + 1
            if cyc_num > max_cycle:
                break
            if cyc is None or cyc.get('Qdlin') is None:
                continue  # 첫 사이클 등 빈 데이터
            qdlin[cyc_num] = np.asarray(cyc['Qdlin'], dtype=np.float32).ravel()
            if cyc.get('Tdlin') is not None:
                tdlin[cyc_num] = np.asarray(cyc['Tdlin'], dtype=np.float32).ravel()

        policy = cell.get('policy_readable') or cell.get('policy') or 'unknown'
        vdlin = cell.get('Vdlin')

        cells[cid] = {
            'batch': batch_key,
            'cell_id': cid,
            'policy': str(policy),
            'cycle_life': _scalar(cell.get('cycle_life')),
            'summary': sdf,
            'Qdlin': qdlin,
            'Tdlin': tdlin,
            'Vdlin': None if vdlin is None else np.asarray(vdlin, dtype=np.float32).ravel(),
        }

    del raw_cells
    gc.collect()
    print(f'[{batch_key}] 셀 {len(cells)}개 추출 완료')
    return cells


def load_or_extract(batch_key, data_dir=DATA_DIR, out_dir=OUT_DIR):
    """배치별 캐시가 있으면 캐시에서, 없으면 .mat에서 추출 후 캐시 저장"""
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, f'raw_{batch_key}.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    cells = extract_batch(batch_key, os.path.join(data_dir, BATCH_FILES[batch_key]))
    with open(cache, 'wb') as f:
        pickle.dump(cells, f)
    return cells


# =========================================================
# 2. Batch1 <-> Batch2 연속 셀 검증 & 병합
# =========================================================
def _valid(sdf):
    return sdf[sdf['QD'] > 0]


def _norm_policy(p):
    """batch2에는 같은 정책에 '-newstructure' 접미사가 붙어 있어 비교 전에 제거"""
    return str(p).replace('-newstructure', '').strip()


def find_continuation(b1, b2, qd_gap_tol=0.03, cut_margin=0.02):
    """
    데이터 기반으로 batch1 <-> batch2 이어지는 셀 쌍 탐색
    (Kaggle 파일은 batch2 셀 수가 원논문과 달라서 인덱스 하드코딩이 맞지 않음)
    1) batch1에서 마지막 용량이 EOL보다 충분히 높은 셀 = 실험이 중간에 끊긴 셀
    2) batch2에서 정책이 같은 셀 중 시작 용량이 batch1 마지막 용량과 가장 가까운 셀
    3) 용량 차이가 qd_gap_tol 이내면 ok
    """
    rows, used = [], set()
    for c1, cell1 in b1.items():
        s1 = _valid(cell1['summary'])
        tail = s1['QD'].tail(10).median()
        if not tail > EOL_CAPACITY + cut_margin:
            continue  # EOL 근처까지 간 셀 = 끊기지 않음
        cands = []
        for c2, cell2 in b2.items():
            if c2 in used or _norm_policy(cell2['policy']) != _norm_policy(cell1['policy']):
                continue
            head = _valid(cell2['summary'])['QD'].head(10).median()
            cands.append((abs(tail - head), c2, head))
        row = {'b1_cell': c1, 'b1_policy': cell1['policy'],
               'b1_last_cycle': int(s1['cycle'].max()), 'b1_tail_QD': round(tail, 4),
               'n_candidates': len(cands)}
        if cands:
            gap, c2, head = min(cands)
            row.update({'b2_cell': c2, 'b2_policy': b2[c2]['policy'],
                        'b2_head_QD': round(head, 4), 'QD_gap': round(gap, 4),
                        'b2_n_cycles': len(_valid(b2[c2]['summary'])),
                        'ok': bool(gap < qd_gap_tol)})
            if row['ok']:
                used.add(c2)
        else:
            row.update({'b2_cell': None, 'ok': False})
        rows.append(row)
    return pd.DataFrame(rows)


def merge_continuation(b1, b2, match_df):
    """ok인 쌍만 병합. batch2 쪽 셀은 batch2에서 제거 (테스트셋 누수 방지)
    병합 후 cycle_life는 run_all()에서 QD 기반 EOL로 다시 계산"""
    if match_df.empty:
        return b1, b2
    for _, r in match_df[match_df['ok']].iterrows():
        c1, c2 = r['b1_cell'], r['b2_cell']
        s1 = b1[c1]['summary']
        s2 = _valid(b2[c2]['summary']).copy()
        # batch2 사이클 번호를 batch1 마지막 사이클 다음부터 이어붙임
        s2['cycle'] = s2['cycle'] - s2['cycle'].min() + s1['cycle'].max() + 1
        b1[c1]['summary'] = pd.concat([s1, s2], ignore_index=True)
        b1[c1]['cycle_life_raw'] = b1[c1]['cycle_life']
        b1[c1]['merged_from'] = c2
        del b2[c2]
        print(f'  병합: {c1} + {c2}')
    return b1, b2


# =========================================================
# 3. 이상치 플래그
# =========================================================
def flag_outliers(sdf, rules=OUTLIER_RULES, window=ROLL_WINDOW):
    """
    행 단위 이상치 플래그 (삭제하지 않음)
    - flag_zero: QD <= 0 (첫 사이클 등 빈 행)
    - flag_<col>: rolling median 대비 rules 기준 이상 튄 값
    - is_outlier: 하나라도 해당
    """
    sdf = sdf.copy()
    sdf['flag_zero'] = sdf['QD'] <= 0
    valid = ~sdf['flag_zero']
    flags = [sdf['flag_zero']]

    for col, (kind, th) in rules.items():
        if col not in sdf.columns:
            continue
        s = sdf[col].where(valid)
        med = s.rolling(window, center=True, min_periods=3).median()
        dev = (s - med).abs()
        if kind == 'rel':
            dev = dev / med.abs().replace(0, np.nan)
        flag = (dev > th) | (valid & (sdf[col] <= 0) & (col != 'IR'))
        flag = flag.fillna(False) & valid
        sdf[f'flag_{col}'] = flag
        flags.append(flag)

    sdf['is_outlier'] = np.logical_or.reduce(flags)
    return sdf


# =========================================================
# 4. EOL 검증
# =========================================================
def eol_from_qd(sdf, eol=EOL_CAPACITY, tol=None, window=5):
    """정상 행의 QD를 rolling median으로 다듬은 뒤, 처음 EOL(+허용오차) 이하로 내려간 사이클
    (실험이 0.88Ah 도달 직후 종료되면 마지막 값이 0.880x처럼 살짝 위일 수 있어 허용오차를 둠)"""
    tol = EOL_TOL if tol is None else tol
    s = sdf.loc[~sdf['is_outlier'], ['cycle', 'QD']]
    smooth = s['QD'].rolling(window, center=True, min_periods=1).median()
    below = s.loc[smooth <= eol + tol, 'cycle']
    return int(below.iloc[0]) if len(below) else np.nan


# =========================================================
# 5. 전체 파이프라인
# =========================================================
def build_tables(all_cells):
    """셀 정보 테이블 + 전체 summary 테이블 생성"""
    info_rows, summ_list = [], []
    for cid, c in all_cells.items():
        sdf = c['summary']
        v = sdf[~sdf['is_outlier']]
        info_rows.append({
            'batch': c['batch'], 'cell_id': cid, 'policy': c['policy'],
            'cycle_life': c['cycle_life'],
            'cycle_life_from_qd': c['cycle_life_from_qd'],
            'n_cycles': len(v),
            'first_QD': v['QD'].iloc[0] if len(v) else np.nan,
            'last_QD': v['QD'].iloc[-1] if len(v) else np.nan,
            'min_QD': v['QD'].min() if len(v) else np.nan,
            'paper_exclude': EXCLUDE_CELLS.get(cid, ''),
            'n_outlier_rows': int(sdf['is_outlier'].sum() - sdf['flag_zero'].sum()),
            'outlier_ratio': round((sdf['is_outlier'].sum() - sdf['flag_zero'].sum())
                                   / max(len(sdf) - sdf['flag_zero'].sum(), 1) * 100, 2),
            'n_qdlin_cycles': len(c['Qdlin']),
            'merged_from': c.get('merged_from'),
            'excluded': c['excluded'], 'exclude_reason': c['exclude_reason'],
        })
        s = sdf.copy()
        s.insert(0, 'cell_id', cid)
        s.insert(0, 'batch', c['batch'])
        s['policy'] = c['policy']
        s['cycle_life'] = c['cycle_life']
        summ_list.append(s)
    return pd.DataFrame(info_rows), pd.concat(summ_list, ignore_index=True)


def run_all(data_dir=DATA_DIR, out_dir=OUT_DIR, force=False):
    final_path = os.path.join(out_dir, 'battery_data.pkl')
    if os.path.exists(final_path) and not force:
        print(f'캐시 로드: {final_path}')
        with open(final_path, 'rb') as f:
            return pickle.load(f)

    # 1) 배치별 추출 (한 번에 하나씩 로드)
    batches = {k: load_or_extract(k, data_dir, out_dir) for k in BATCH_FILES}
    n_before = {k: len(v) for k, v in batches.items()}

    # 2) batch1 <-> batch2 연속 셀 검증 & 병합
    print('\n[Batch1 <-> Batch2 연속 셀 탐색]')
    verify_df = find_continuation(batches['b1'], batches['b2'])
    print(verify_df.to_string(index=False) if len(verify_df) else '  끊긴 batch1 셀 없음')
    batches['b1'], batches['b2'] = merge_continuation(batches['b1'], batches['b2'], verify_df)

    # 3) 이상치 플래그 & 4) EOL 검증 & 5) 제외 표시
    all_cells = {}
    for k, cells in batches.items():
        for cid, c in cells.items():
            c['summary'] = flag_outliers(c['summary'])
            c['cycle_life_from_qd'] = eol_from_qd(c['summary'])
            if c.get('merged_from'):   # 병합 셀은 이어붙인 데이터로 수명 재계산
                c['cycle_life'] = c['cycle_life_from_qd']
            reasons = []
            pol = c['policy']
            if APPLY_PAPER_EXCLUDE and cid in EXCLUDE_CELLS:
                reasons.append(EXCLUDE_CELLS[cid])
            if 'VarCharge' in pol or 'SLOWCYCLE' in pol:
                reasons.append('다른 실험 프로토콜 (VarCharge/SLOWCYCLE)')
            if np.isnan(c['cycle_life_from_qd']):
                if cid in B1_B2_CONTINUATION and not c.get('merged_from'):
                    reasons.append('실험 중단: 수명 미확인 (원논문은 batch2로 이어붙였으나 Kaggle 파일에 이어지는 데이터 없음)')
                else:
                    reasons.append('EOL(0.88Ah) 미도달')
            if np.isnan(c['cycle_life']):
                reasons.append('cycle_life 없음')
            c['excluded'] = len(reasons) > 0
            c['exclude_reason'] = ' / '.join(reasons)
            all_cells[cid] = c

    cell_info, summary = build_tables(all_cells)

    # 리포트
    print('\n[배치별 셀 수]')
    report = pd.DataFrame({
        '원본': pd.Series(n_before),
        '병합 후': cell_info.groupby('batch').size(),
        '최종(제외 후)': cell_info[~cell_info['excluded']].groupby('batch').size(),
    }).fillna(0).astype(int)
    print(report.to_string())

    print('\n[이상치 행 비율 (첫 사이클 0값 제외)]')
    s_valid = summary[~summary['flag_zero']]
    print(s_valid.groupby('batch')['is_outlier'].mean().mul(100).round(2).astype(str) + ' %')

    print('\n[cycle_life vs QD 기반 EOL 차이 상위 10개]')
    diff = (cell_info['cycle_life'] - cell_info['cycle_life_from_qd']).abs()
    print(cell_info.assign(diff=diff).sort_values('diff', ascending=False)
          [['cell_id', 'cycle_life', 'cycle_life_from_qd', 'diff', 'exclude_reason']]
          .head(10).to_string(index=False))

    print('\n[제외 셀]')
    print(cell_info[cell_info['excluded']][['cell_id', 'policy', 'cycle_life', 'n_cycles',
                                            'last_QD', 'min_QD', 'exclude_reason']]
          .to_string(index=False))

    print('\n[배치별 QD 분포 진단: 마지막 용량 / 최소 용량]')
    print(cell_info.groupby('batch')[['first_QD', 'last_QD', 'min_QD']].describe()
          .round(4).T.to_string())

    data = {
        'cells': all_cells,          # 셀별 상세 (summary, Qdlin, Tdlin, Vdlin)
        'cell_info': cell_info,      # 셀 1행 요약
        'summary': summary,          # 전체 사이클 summary (+ 이상치 플래그)
        'verify_continuation': verify_df,
        'report': report,
        'config': {'EOL_CAPACITY': EOL_CAPACITY, 'MAX_CYCLE_DETAIL': MAX_CYCLE_DETAIL,
                   'OUTLIER_RULES': OUTLIER_RULES, 'ROLL_WINDOW': ROLL_WINDOW},
    }
    with open(final_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'\n저장 완료: {final_path}')
    return data


if __name__ == '__main__':
    run_all()