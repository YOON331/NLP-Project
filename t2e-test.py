import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from datasets import load_dataset
import pandas as pd
import os

def load_emoji_dict():
    """
    'omarkamali/emoji-map' 데이터셋을 로드하여 이모지와 설명의 딕셔너리를 생성합니다.
    """
    emoji_map_dataset = load_dataset('omarkamali/emoji-map')['train']
    emoji_dict = {}
    for row in emoji_map_dataset:
        emoji_char = row['emoji']
        desc = row['description_kor_Hang']
        if desc:
            emoji_dict[emoji_char] = desc
    return emoji_dict

def initialize_model(model_path):
    """
    주어진 경로에서 모델과 토크나이저를 로드하고, 토크나이저의 길이에 맞게 모델의 임베딩을 조정합니다.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer

def generate_emojis(model, tokenizer, input_text, max_length=64):
    """
    주어진 한국어 문장을 사용하여 이모지 시퀀스를 생성합니다.
    """
    inputs = "[TEXT2EMOJI] " + input_text
    input_ids = tokenizer.encode(inputs, return_tensors='pt').to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,               # 생성할 최대 토큰 수
        num_beams=5,                      # Beam Search의 빔 수
        early_stopping=True,             # 조기 종료 활성화
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,          # 동일한 2-gram 반복 방지
        length_penalty=1.2,               # 길이 패널티
    )
    
    emojis = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return emojis

def generate_text(model, tokenizer, emoji_sequence, emoji_dict, max_length=64):
    """
    주어진 이모지 시퀀스를 사용하여 한국어 문장을 생성합니다.
    """
    # 이모지에 대한 설명을 딕셔너리에서 가져옵니다.
    descriptions = [emoji_dict.get(e, "") for e in emoji_sequence.split()]
    descriptions = [desc for desc in descriptions if desc]  # 빈 설명 제거
    descriptions_str = " ".join(descriptions)
    
    inputs = "[EMOJI2TEXT] [EMOJI_DESC] " + descriptions_str
    input_ids = tokenizer.encode(inputs, return_tensors='pt').to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,               # 생성할 최대 토큰 수
        num_beams=5,                      # Beam Search의 빔 수
        early_stopping=True,             # 조기 종료 활성화
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,          # 동일한 2-gram 반복 방지
        length_penalty=1.2,               # 길이 패널티
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    # 모델 경로 설정
    model_a_path = './kobart-1214-4e5'                       # 모델 A 경로
    model_b_path = './kobart-with-emoji-map-dict2-4e5'       # 모델 B 경로
    
    # 이모지 딕셔너리 로드
    print("이모지 딕셔너리를 로드 중...")
    emoji_dict = load_emoji_dict()
    print("이모지 딕셔너리 로드 완료.")
    
    # 모델과 토크나이저 초기화
    print("모델 A 로드 중...")
    model_a, tokenizer_a = initialize_model(model_a_path)
    print("모델 A 로드 완료.")
    
    print("모델 B 로드 중...")
    model_b, tokenizer_b = initialize_model(model_b_path)
    print("모델 B 로드 완료.")
    
    print("\n모델 준비 완료. 한국어 문장을 입력하세요. 종료하려면 'exit', 'quit', 'q'를 입력하세요.\n")
    
    while True:
        # 사용자 입력 받기
        user_input = input("한국어 문장을 입력하세요 >>> ")
        if user_input.strip().lower() in ['exit', 'quit', 'q']:
            print("프로그램을 종료합니다.")
            break
        
        # 모델 A를 사용하여 이모지 시퀀스 생성
        emojis_a = generate_emojis(model_a, tokenizer_a, user_input)
        print("\nA 모델 이모지 시퀀스 결과:")
        print(emojis_a)
        
        # 모델 B를 사용하여 이모지 시퀀스 생성
        emojis_b = generate_emojis(model_b, tokenizer_b, user_input)
        print("\nB 모델 이모지 시퀀스 결과:")
        print(emojis_b)
        
        # 모델 A를 사용하여 이모지 시퀀스를 한국어 문장으로 번역
        text_a = generate_text(model_a, tokenizer_a, emojis_a, emoji_dict)
        print("\nA 모델의 이모지 시퀀스를 한국어 시퀀스로 번역한 결과 >>>")
        print(text_a)
        
        # 모델 B를 사용하여 이모지 시퀀스를 한국어 문장으로 번역
        text_b = generate_text(model_b, tokenizer_b, emojis_b, emoji_dict)
        print("\nB 모델의 이모지 시퀀스를 한국어 시퀀스로 번역한 결과 >>>")
        print(text_b)
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
