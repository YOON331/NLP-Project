import openai
import csv
import time
import re
import os 
from dotenv import load_dotenv

# ENV 파일 내용 가져오기
load_dotenv()
API_KEY = os.environ.get("API_KEY")

# OpenAI API Key 설정
openai.api_key = API_KEY

# 주제별 프롬프트 정의
topics = ["감정", "동물", "자연", "음식", "활동", "직업", "여행", "장소", "직업", "사물", "국가", "사계절", "일상", "경험"]


# CSV 파일 저장 경로
csv_filename = "test.csv"

# API 요청을 보내는 함수
def generate_sentences(topic, n=2):  # 각 토픽별로 n개 생성
    prompt = (
        f"{topic}에 대해 설명하는 완전하고 온전한 한국어 문장으로만 작성하세요. "
        "각 문장별 응답 형식: "
        f"(문장):(이모지 시퀀스):{topic}. "
        "응답 예시: "
        f"밤하늘을 보면서 평화로움을 느꼈어:🌙✨👀🧘‍♂️:{topic}. "
        f"눈이 너무 내려서 얼어 죽는줄 알았어:❄️🌨️🥶:{topic}. "
        "창의적이고 비유적이며 한 문장당 이모지 최소 2개에서 최대 20개의 응답이 생성됩니다. "
        f"이모지 사용시 {topic}과 관련된 카테고리의 이모지를 필수로 포함합니다."
    )

# CSV 파일을 append 모드로 열기
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:  
        writer = csv.writer(file)
        
        # 파일이 처음 생성되면 헤더 추가
        if file.tell() == 0:  # 파일의 위치가 0일 때 헤더 작성
            writer.writerow(["text", "emoji", "topic"])  # CSV 파일에 헤더 추가 (한번만)
        
        for _ in range(n):
            try:
                # GPT-4 모델을 사용한 문장 및 이모지 시퀀스 생성
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=170,
                    temperature=1.0,
                )
                
                # 응답 내용 확인 및 바로 저장 준비
                generated_text = response['choices'][0]['message']['content'].strip()
                print(f"{generated_text}")  # 응답을 콘솔에 출력
                
                # 문장이 여러 개일 경우 구분자 '.'으로 나누기
                sentences = generated_text.split('.')
                for sentence in sentences:
                    if ':' in sentence:
                        parts = sentence.split(':')
                        if len(parts) == 3:  # 정확하게 세 부분으로 분리되었는지 확인
                            text, emojis, topic = parts[0].strip(), parts[1].strip(), parts[2].strip()

                            if len(emojis) > 1:
                                writer.writerow([text, emojis, topic])  # 바로 CSV 파일에 저장
                                # print(f"Saved: {text}, {emojis}, {topic}")
                            else:
                                print(f"Skipping entry: Not enough emojis for topic {topic}")
                        else:
                            print(f"Skipping entry: Incorrect format in response for topic {topic}")
                    else:
                        print(f"Skipping entry: No proper format found in response for topic {topic}")
                    time.sleep(1)
            except Exception as e:
                print(f"Error generating text for topic {topic}: {e}")
                continue

# # 주제별로 돌아가면서 문장 생성
def generate_sentences_across_topics(topics, n_per_topic):
    total_generated = 0
    while total_generated < n_per_topic * len(topics):
        for topic in topics:
            generate_sentences(topic, n=1)  # 한 번에 각 토픽별로 하나씩 생성
            total_generated += 1
            if total_generated >= n_per_topic * len(topics):
                break

# # 문장 생성 실행
generate_sentences_across_topics(topics, 3)

print(f"Data saved to {csv_filename}")