import openai
import pandas as pd
import time
import csv
import os 
from dotenv import load_dotenv

# ENV 파일 내용 가져오기
load_dotenv()
API_KEY = os.environ.get("API_KEY")
EMOJI_DATA_PATH = os.environ.get("EMOJI_DATA_PATH")
EMOJI_KOR_DATA_PATH = os.environ.get("EMOJI_KOR_DATA_PATH")

# API 키 설정
openai.api_key = API_KEY
# 단계별 확인용 코드
print("API 키 설정 완료")

# CSV 파일 불러오기
try:
    df = pd.read_csv(EMOJI_DATA_PATH)
    print("CSV 파일 불러오기 성공")
except Exception as e:
    print("CSV 파일 불러오기 실패:", e)

# 데이터 출력
print("데이터 샘플:", df.head())

# 새로운 CSV 파일을 쓰기 모드로 열기
with open(EMOJI_KOR_DATA_PATH, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 헤더 작성
    writer.writerow(['unicode', 'emoji', 'description', 'description_ko'])

    # OpenAI API를 사용해 description 열을 한글로 번역하고 한 줄씩 저장
    for index, row in df.iterrows():
        try:
            # 진행 상황 표시
            print(f"Translating row {index + 1}/{len(df)}: {row['description']}")

            # API 호출하여 description 번역
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a translation assistant."},
                    {"role": "user", "content": f"Translate the following English text to Korean: '{row['description']}'"}
                ],
                max_tokens=60,
                temperature=0.3
            )
            # 번역 결과를 변수에 저장
            translation = response['choices'][0]['message']['content'].strip()
            print(f"Translated {row['description']} to {translation}")

        except Exception as e:
            print(f"Error translating row {index}: {e}")
            translation = 'Translation Error'

        # 번역된 행을 새 CSV 파일에 한 줄씩 추가
        writer.writerow([row['unicode'], row['emoji'], row['description'], translation])

        # API 호출 사이에 지연 시간 추가
        time.sleep(1)

print("번역 파일 저장 완료")