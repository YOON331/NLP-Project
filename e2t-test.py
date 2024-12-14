import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm
import os
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# -----------------------------
# 사용자 입력 처리 함수
# -----------------------------
def user_input_interactive(models, tokenizers, emoji_dict, device, max_length=64):
    while True:
        # 사용자 입력 받기
        user_input = input("이모지 시퀀스를 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            print("종료합니다.")
            break

        # 모델별로 결과 계산 및 출력
        for model_name, (model, tokenizer) in models.items():
            print(f"\n[{model_name}] 결과:")

            # Emoji→Text 변환
            emojis = user_input.split()
            emoji_to_text_input = "[EMOJI2TEXT] " + " ".join(emojis)
            emoji_descriptions = [emoji_dict.get(e, "") for e in emojis]
            if any(desc for desc in emoji_descriptions):
                emoji_to_text_input += " [EMOJI_DESC] " + " ".join(emoji_descriptions)

            emoji_to_text_enc = tokenizer(
                emoji_to_text_input,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=True,
            )

            emoji_to_text_output = model.generate(
                input_ids=emoji_to_text_enc["input_ids"].to(device),
                attention_mask=emoji_to_text_enc["attention_mask"].to(device),
                max_new_tokens=64,
                num_beams=5,
                num_beam_groups=5,
                diversity_penalty=0.7,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
                length_penalty=1.2,
            )
            emoji_to_text_result = tokenizer.decode(emoji_to_text_output[0], skip_special_tokens=True).strip()
            print(f"[번역된 한국어 문장]: {emoji_to_text_result}")

# -----------------------------
# 메인 실행 코드
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 및 토크나이저 불러오기
    models = {}

    for model_name, model_path in [
        ("kobart-with-emoji-map-dict2-4e5", './kobart-with-emoji-map-dict2-4e5'),
        ("kobart-1214-4e5", './kobart-1214-4e5')
    ]:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path).to(device)

        if tokenizer.eos_token is None:
            tokenizer.eos_token = '</s>'

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.decoder_start_token_id = tokenizer.bos_token_id

        models[model_name] = (model, tokenizer)

    # Emoji 사전 로드
    emoji_map_dataset = load_dataset('omarkamali/emoji-map')['train']
    emoji_dict = {}
    for row in emoji_map_dataset:
        if row['description_kor_Hang']:
            emoji_dict[row['emoji']] = row['description_kor_Hang']

    print("인터랙티브 모드를 시작합니다.")
    user_input_interactive(models, None, emoji_dict, device)

if __name__ == "__main__":
    main()
