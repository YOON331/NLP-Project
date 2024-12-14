import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np

# 일관된 결과를 위한 시드 설정
seed = 42
random.seed(seed)
np.random.seed(seed)

dataset = pd.read_csv('./dataset_v5.csv')
answerdata = pd.read_csv('./answerdata_v5.csv')

# 결측값 제거
dataset.dropna(subset=['text', 'emoji'], inplace=True)
answerdata.dropna(subset=['text', 'emoji'], inplace=True)

# 'topic' 컬럼 삭제 (필요 시)

# 3. 데이터셋 분할
# 3.1 answerdata 분할: 학습 30%, 검증 35%, 테스트 35%
train_answer, temp_answer = train_test_split(
    answerdata,
    test_size=0.7,
    random_state=seed,
)

val_answer, test_answer = train_test_split(
    temp_answer,
    test_size=0.5,
    random_state=seed,
)

# 3.2 dataset 분할: 학습 70%, 검증 15%, 테스트 15%
train_dataset, temp_dataset = train_test_split(
    dataset,
    test_size=0.3,
    random_state=seed,
)

val_dataset, test_dataset = train_test_split(
    temp_dataset,
    test_size=0.5,
    random_state=seed,
)

# 3.3 최종적으로 합치기
final_train_dataset = pd.concat([train_answer, train_dataset], ignore_index=True)
final_val_dataset = pd.concat([val_answer, val_dataset], ignore_index=True)
final_test_dataset = pd.concat([test_answer, test_dataset], ignore_index=True)

# 분할된 데이터셋 크기 출력
print(f"Final Train Set: {len(final_train_dataset)} samples")
print(f"Final Validation Set: {len(final_val_dataset)} samples")
print(f"Final Test Set: {len(final_test_dataset)} samples")

# 4. 데이터셋 저장
final_train_dataset.to_csv('final_train_dataset.csv', index=False, encoding='utf-8-sig')
final_val_dataset.to_csv('final_val_dataset.csv', index=False, encoding='utf-8-sig')
final_test_dataset.to_csv('final_test_dataset.csv', index=False, encoding='utf-8-sig')

print("Datasets have been saved as CSV files.")