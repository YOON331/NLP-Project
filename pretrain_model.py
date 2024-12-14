from datasets import load_dataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, TrainingArguments, Trainer
import torch
import random

# 1. 데이터셋 로드
dataset = load_dataset("omarkamali/emoji-map", split="train")
emoji_data = [{"emoji": row["emoji"], "descrition": row["description_kor_Hang"]} for row in dataset if row["description_kor_Hang"]]

# 2. 마스킹 함수 정의
def mask_descrition(descrition, mask_token="[MASK]", mask_prob=0.15, strategy="random"):
    words = descrition.split()
    if strategy == "random":
        masked_words = [mask_token if random.random() < mask_prob else word for word in words]
    elif strategy == "contiguous":
        if len(words) > 2:
            start_idx = random.randint(0, len(words) - 2)
            words[start_idx:start_idx + 2] = [mask_token]
        masked_words = words
    elif strategy == "full":
        masked_words = [mask_token]
    else:
        raise ValueError("Invalid masking strategy")
    return " ".join(masked_words)

# 3. 마스킹 적용
augmented_data = []
for row in emoji_data:
    masked_descrition = mask_descrition(row["descrition"], strategy="random")
    augmented_data.append({
        "emoji": row["emoji"],
        "masked_descrition": masked_descrition,
        "descrition": row["descrition"]
    })

# 4. 토크나이저 및 모델 준비
tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")

# 이모지를 분리하고 U+200D 제거
def split_emojis(emoji):
    return [char for char in emoji.replace("\u200d", "")]

# 새로운 이모지 추출
all_emojis = set()
for row in augmented_data:
    all_emojis.update(split_emojis(row["emoji"]))  # 개별 이모지를 모두 추가

# 토크나이저에 없는 이모지 확인
existing_tokens = set(tokenizer.get_vocab().keys())
new_emojis = [emoji for emoji in all_emojis if emoji not in existing_tokens]

# 스페셜 토큰 추가
special_tokens = {'additional_special_tokens': ['[EMOJI]', '[DESC]']}
tokenizer.add_special_tokens(special_tokens)

# 새로운 이모지 토큰 추가
tokenizer.add_tokens(new_emojis)
model.resize_token_embeddings(len(tokenizer))

# 5. 데이터 전처리 함수 정의
# Input: [EMOJI] + emoji + [DESC] + masked_descrition
# Label: descrition
def preprocess_function(examples):
    # 이모지 단순화
    split_emoji = " ".join(split_emojis(examples["emoji"]))
    inputs = "[EMOJI] " + split_emoji + " [DESC] " + examples["masked_descrition"]
    labels = examples["descrition"]

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=64)
    with tokenizer.as_target_tokenizer():
        label_inputs = tokenizer(labels, padding="max_length", truncation=True, max_length=64)

    model_inputs["labels"] = label_inputs["input_ids"]
    # pad 토큰의 라벨은 -100으로 설정해 모델이 loss 계산 시 무시하도록 함
    model_inputs["labels"] = [
        (lbl if lbl != tokenizer.pad_token_id else -100) 
        for lbl in model_inputs["labels"]
    ]
    return model_inputs

tokenized_dataset = list(map(preprocess_function, augmented_data))

# PyTorch Dataset 형식으로 변환
class EmojiDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.data[idx].items()}

train_dataset = EmojiDataset(tokenized_dataset)

# 6. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./kobart-1214",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs"
)

# 7. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 8. 학습 실행
trainer.train()

# 9. 모델 및 토크나이저 저장
trainer.save_model("./kobart-1214")
tokenizer.save_pretrained("./kobart-1214")

print("사전학습 완료 및 모델 저장 완료!")


# 새로운 이모지 토큰 추가 및 개수 확인
added_tokens_count = tokenizer.add_tokens(new_emojis)
model.resize_token_embeddings(len(tokenizer))

# 추가된 토큰 개수 출력
print(f"추가된 토큰 개수: {added_tokens_count}")
