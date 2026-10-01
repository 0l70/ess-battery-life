# Data
Kaggle: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle

아래 3개 파일을 이 폴더에 넣으세요 (varcharge 파일은 사용하지 않음)
- 2017-05-12_batchdata_updated_struct_errorcorrect.mat  (Batch 1, 학습)
- 2018-02-20_batchdata_updated_struct_errorcorrect.mat  (Batch 2, 테스트)
- 2018-04-12_batchdata_updated_struct_errorcorrect.mat  (Batch 3, 추가 테스트)

이후 `python preprocess.py` 실행 시 processed/ 에 캐시가 생성됩니다.
