import requests
import csv

# 데이터셋 다운로드
url = "https://www.unicode.org/Public/emoji/16.0/emoji-test.txt"
response = requests.get(url)
data = response.text
file_name = 'get_emoji_test.csv'

# 이모티콘, 유니코드, 설명을 저장할 리스트
emoji_data = []

# 각 줄을 처리하여 유니코드, 이모티콘과 설명을 추출
for line in data.splitlines():
    if not line.startswith('#') and line.strip():  # 주석 및 빈 줄 제외
        parts = line.split(';')
        if len(parts) > 1:
            unicode_code = parts[0].strip()  # 유니코드 시퀀스
            emoji = unicode_code.split()  # 공백으로 분리된 유니코드 시퀀스를 하나로 처리
            emoji = ''.join([chr(int(code, 16)) for code in emoji])  # 유니코드 변환
            description = parts[1].strip()  # 전체 설명
            # 설명에서 '#'과 버전 정보를 제거하고 그 뒤의 텍스트만 남김
            description = description.split('#')[-1].strip()  # '#' 뒤의 부분 추출
            description = ' '.join(description.split(' ')[2:])  # 버전 뒤의 텍스트 추출
            
            emoji_data.append((unicode_code, emoji, description))

# CSV 파일로 저장
with open(file_name, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['unicode', 'emoji', 'description'])  # 헤더 작성
    for unicode_code, emoji, description in emoji_data:
        writer.writerow([unicode_code, emoji, description])  # 유니코드, 이모티콘, 설명을 행으로 저장

print(f"데이터가 {file_name} 파일에 저장되었습니다.")


# CSV 파일 읽기
with open(file_name, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
