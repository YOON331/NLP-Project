import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, AdamW, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def initialize_model_and_tokenizer(pretrained_model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name)
    # 스페셜 토큰 추가
    special_tokens = {'additional_special_tokens': ['[TEXT2EMOJI]', '[EMOJI2TEXT]', '[EMOJI_DESC]']}
    tokenizer.add_special_tokens(special_tokens)
    model = BartForConditionalGeneration.from_pretrained(pretrained_model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, device

class UnifiedTextEmojiDataset(Dataset):
    def __init__(self, data, tokenizer, emoji_dict, max_length=64):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.emoji_dict = emoji_dict
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = row['text']
        emoji_str = row['emoji']  # 이모지 시퀀스 문자열

        # 토크나이저로 이모지 문자열 토큰화
        # 여기서 이미 tokenizer에 이모지 토큰이 추가되어 있다고 가정
        emojis = self.tokenizer.tokenize(emoji_str)

        # 각 이모지에 대한 설명 매핑
        emoji_descriptions = [self.emoji_dict.get(e, "") for e in emojis]

        # Text-to-Emoji Task
        # Input: "[TEXT2EMOJI] {text}"
        # Label: "{emoji1} {emoji2} ... {emojiN}<EOS>"
        text_to_emoji_input = '[TEXT2EMOJI] ' + text
        text_to_emoji_target = " ".join(emojis) + self.tokenizer.eos_token

        # Emoji-to-Text Task
        # Input: "[EMOJI2TEXT] {emoji1} {emoji2} ... {emojiN} [EMOJI_DESC] {desc1} {desc2} ... {descN}"
        # Label: "{text}<EOS>"
        emoji_to_text_input = '[EMOJI2TEXT] ' + " ".join(emojis)
        if any(desc for desc in emoji_descriptions):
            emoji_to_text_input += ' [EMOJI_DESC] ' + " ".join(emoji_descriptions)
        emoji_to_text_target = text + self.tokenizer.eos_token

        # 토크나이징
        text_to_emoji_enc = self.tokenizer(
            text_to_emoji_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        text_to_emoji_dec = self.tokenizer(
            text_to_emoji_target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        emoji_to_text_enc = self.tokenizer(
            emoji_to_text_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        emoji_to_text_dec = self.tokenizer(
            emoji_to_text_target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        text_to_emoji_labels = text_to_emoji_dec['input_ids'].clone()
        text_to_emoji_labels[text_to_emoji_labels == self.tokenizer.pad_token_id] = -100

        emoji_to_text_labels = emoji_to_text_dec['input_ids'].clone()
        emoji_to_text_labels[emoji_to_text_labels == self.tokenizer.pad_token_id] = -100

        return {
            'text_to_emoji_input_ids': text_to_emoji_enc['input_ids'].squeeze(),
            'text_to_emoji_attention_mask': text_to_emoji_enc['attention_mask'].squeeze(),
            'text_to_emoji_labels': text_to_emoji_labels.squeeze(),
            'emoji_to_text_input_ids': emoji_to_text_enc['input_ids'].squeeze(),
            'emoji_to_text_attention_mask': emoji_to_text_enc['attention_mask'].squeeze(),
            'emoji_to_text_labels': emoji_to_text_labels.squeeze(),
        }

def evaluate_model(model, val_loader, device, tokenizer):
    model.eval()
    text_to_emoji_loss_total = 0.0
    emoji_to_text_loss_total = 0.0
    text_to_emoji_correct = 0
    emoji_to_text_correct = 0
    text_to_emoji_tokens = 0
    emoji_to_text_tokens = 0
    text_to_emoji_seq_correct = 0
    emoji_to_text_seq_correct = 0
    import numpy as np

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            text_to_emoji_input_ids = batch['text_to_emoji_input_ids'].to(device)
            text_to_emoji_attention_mask = batch['text_to_emoji_attention_mask'].to(device)
            text_to_emoji_labels = batch['text_to_emoji_labels'].to(device)

            text_to_emoji_outputs = model(
                input_ids=text_to_emoji_input_ids,
                attention_mask=text_to_emoji_attention_mask,
                labels=text_to_emoji_labels
            )
            text_to_emoji_loss = text_to_emoji_outputs.loss
            text_to_emoji_loss_total += text_to_emoji_loss.item()

            text_to_emoji_logits = text_to_emoji_outputs.logits
            text_to_emoji_predictions = torch.argmax(text_to_emoji_logits, dim=-1)
            text_to_emoji_correct += ((text_to_emoji_predictions == text_to_emoji_labels) & (text_to_emoji_labels != -100)).sum().item()
            text_to_emoji_tokens += (text_to_emoji_labels != -100).sum().item()

            emoji_to_text_input_ids = batch['emoji_to_text_input_ids'].to(device)
            emoji_to_text_attention_mask = batch['emoji_to_text_attention_mask'].to(device)
            emoji_to_text_labels = batch['emoji_to_text_labels'].to(device)

            emoji_to_text_outputs = model(
                input_ids=emoji_to_text_input_ids,
                attention_mask=emoji_to_text_attention_mask,
                labels=emoji_to_text_labels
            )
            emoji_to_text_loss = emoji_to_text_outputs.loss
            emoji_to_text_loss_total += emoji_to_text_loss.item()

            emoji_to_text_logits = emoji_to_text_outputs.logits
            emoji_to_text_predictions = torch.argmax(emoji_to_text_logits, dim=-1)
            emoji_to_text_correct += ((emoji_to_text_predictions == emoji_to_text_labels) & (emoji_to_text_labels != -100)).sum().item()
            emoji_to_text_tokens += (emoji_to_text_labels != -100).sum().item()

            # 시퀀스 단위 정확도 계산
            seq_mask = (text_to_emoji_labels != -100)
            correct_predictions = (text_to_emoji_predictions == text_to_emoji_labels) | (~seq_mask)
            seq_correct = correct_predictions.all(dim=1)
            text_to_emoji_seq_correct += seq_correct.sum().item()

            seq_mask = (emoji_to_text_labels != -100)
            correct_predictions = (emoji_to_text_predictions == emoji_to_text_labels) | (~seq_mask)
            seq_correct = correct_predictions.all(dim=1)
            emoji_to_text_seq_correct += seq_correct.sum().item()

    avg_text_to_emoji_loss = text_to_emoji_loss_total / len(val_loader)
    avg_emoji_to_text_loss = emoji_to_text_loss_total / len(val_loader)

    text_to_emoji_perplexity = np.exp(avg_text_to_emoji_loss) if avg_text_to_emoji_loss < 20 else float('inf')
    emoji_to_text_perplexity = np.exp(avg_emoji_to_text_loss) if avg_emoji_to_text_loss < 20 else float('inf')

    text_to_emoji_accuracy = text_to_emoji_correct / text_to_emoji_tokens if text_to_emoji_tokens > 0 else 0.0
    emoji_to_text_accuracy = emoji_to_text_correct / emoji_to_text_tokens if emoji_to_text_tokens > 0 else 0.0

    text_to_emoji_seq_accuracy = text_to_emoji_seq_correct / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
    emoji_to_text_seq_accuracy = emoji_to_text_seq_correct / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0

    return (
        avg_text_to_emoji_loss,
        text_to_emoji_perplexity,
        text_to_emoji_accuracy,
        text_to_emoji_seq_accuracy,
        avg_emoji_to_text_loss,
        emoji_to_text_perplexity,
        emoji_to_text_accuracy,
        emoji_to_text_seq_accuracy,
    )

def train_unified_model(model, tokenizer, train_loader, val_loader, optimizer, num_epochs, beta, gamma, device, save_path, patience=3, warmup_steps=500):
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            # Text-to-Emoji Task
            text_to_emoji_input_ids = batch['text_to_emoji_input_ids'].to(device)
            text_to_emoji_attention_mask = batch['text_to_emoji_attention_mask'].to(device)
            text_to_emoji_labels = batch['text_to_emoji_labels'].to(device)

            text_to_emoji_outputs = model(
                input_ids=text_to_emoji_input_ids,
                attention_mask=text_to_emoji_attention_mask,
                labels=text_to_emoji_labels
            )
            text_to_emoji_loss = text_to_emoji_outputs.loss

            # Emoji-to-Text Task
            emoji_to_text_input_ids = batch['emoji_to_text_input_ids'].to(device)
            emoji_to_text_attention_mask = batch['emoji_to_text_attention_mask'].to(device)
            emoji_to_text_labels = batch['emoji_to_text_labels'].to(device)

            emoji_to_text_outputs = model(
                input_ids=emoji_to_text_input_ids,
                attention_mask=emoji_to_text_attention_mask,
                labels=emoji_to_text_labels
            )
            emoji_to_text_loss = emoji_to_text_outputs.loss

            total_batch_loss = beta * text_to_emoji_loss + gamma * emoji_to_text_loss
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += total_batch_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

        # Validation
        (
            val_text_to_emoji_loss,
            val_text_to_emoji_perplexity,
            val_text_to_emoji_accuracy,
            val_text_to_emoji_seq_accuracy,
            val_emoji_to_text_loss,
            val_emoji_to_text_perplexity,
            val_emoji_to_text_accuracy,
            val_emoji_to_text_seq_accuracy
        ) = evaluate_model(model, val_loader, device, tokenizer)

        print(f"Epoch {epoch+1}, Validation Results:")
        print(f"  Text-to-Emoji Loss: {val_text_to_emoji_loss:.4f}, PPL: {val_text_to_emoji_perplexity:.4f}, Token-Acc: {val_text_to_emoji_accuracy:.4f}, Seq-Acc: {val_text_to_emoji_seq_accuracy:.4f}")
        print(f"  Emoji-to-Text Loss: {val_emoji_to_text_loss:.4f}, PPL: {val_emoji_to_text_perplexity:.4f}, Token-Acc: {val_emoji_to_text_accuracy:.4f}, Seq-Acc: {val_emoji_to_text_seq_accuracy:.4f}")

        combined_val_loss = val_text_to_emoji_loss + val_emoji_to_text_loss
        if combined_val_loss < best_val_loss:
            best_val_loss = combined_val_loss
            no_improvement_count = 0
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Best model saved with combined validation loss: {best_val_loss:.4f}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("No improvement for {} epochs. Early stopping.")
                break

        # 모델 예측 예시 출력
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # text→emoji 
                input_ids = batch['text_to_emoji_input_ids'][0].unsqueeze(0).to(device)
                attention_mask = batch['text_to_emoji_attention_mask'][0].unsqueeze(0).to(device)
                generated = model.generate(input_ids, attention_mask=attention_mask, max_length=20)
                gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                src_text = tokenizer.decode(batch['text_to_emoji_input_ids'][0], skip_special_tokens=True)
                print(f"Text2Emoji Example: Input: {src_text}, Generated: {gen_text}")

                # emoji→text
                input_ids = batch['emoji_to_text_input_ids'][0].unsqueeze(0).to(device)
                attention_mask = batch['emoji_to_text_attention_mask'][0].unsqueeze(0).to(device)
                generated = model.generate(input_ids, attention_mask=attention_mask, max_length=20)
                gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                src_text = tokenizer.decode(batch['emoji_to_text_input_ids'][0], skip_special_tokens=True)
                print(f"Emoji2Text Example: Input: {src_text}, Generated: {gen_text}")
                
                # 한 번만 출력하고 break
                break

def check_dataset(dataset, emoji_dict, tokenizer):
    print("Checking Dataset Samples...")
    for idx in range(3):  # 샘플 3개 출력
        sample = dataset[idx]

        # Text-to-Emoji Task 확인
        text_to_emoji_input = tokenizer.decode(sample['text_to_emoji_input_ids'], skip_special_tokens=False)
        text_to_emoji_label_ids = sample['text_to_emoji_labels']
        text_to_emoji_label_ids = torch.where(text_to_emoji_label_ids == -100, torch.tensor(tokenizer.pad_token_id), text_to_emoji_label_ids)
        text_to_emoji_label = tokenizer.decode(text_to_emoji_label_ids, skip_special_tokens=False)
        print(f"[Text-to-Emoji] Input: {text_to_emoji_input}")
        print(f"[Text-to-Emoji] Label: {text_to_emoji_label}")

        # Emoji-to-Text Task 확인
        emoji_to_text_input = tokenizer.decode(sample['emoji_to_text_input_ids'], skip_special_tokens=False)
        emoji_to_text_label_ids = sample['emoji_to_text_labels']
        emoji_to_text_label_ids = torch.where(emoji_to_text_label_ids == -100, torch.tensor(tokenizer.pad_token_id), emoji_to_text_label_ids)
        emoji_to_text_label = tokenizer.decode(emoji_to_text_label_ids, skip_special_tokens=False)
        print(f"[Emoji-to-Text] Input: {emoji_to_text_input}")
        print(f"[Emoji-to-Text] Label: {emoji_to_text_label}")

        # 이모지 정보 출력
        splitted = emoji_to_text_input.split()
        # "[EMOJI2TEXT]" 다음부터 "[EMOJI_DESC]" 이전까지 이모지 시퀀스
        if "[EMOJI_DESC]" in splitted:
            desc_index = splitted.index("[EMOJI_DESC]")
            emojis = splitted[1:desc_index]
            descs = splitted[desc_index+1:]
        else:
            emojis = splitted[1:]
            descs = []

        print(f"Emoji Sequence: {emojis}")
        print(f"Descriptions: {descs}")
        print("-" * 50)

def main():
    set_seed(42)
    pretrained_model_name = './kobart-1214'
    model, tokenizer, device = initialize_model_and_tokenizer(pretrained_model_name)

    # 학습 및 검증 데이터 로드
    train_data = pd.read_csv("final_train_dataset.csv")
    val_data = pd.read_csv("final_val_dataset.csv")

    # text 열에서 쌍 따옴표 제거
    train_data['text'] = train_data['text'].str.replace('"', '', regex=False)

    # 이모지 딕셔너리 생성
    emoji_map_dataset = load_dataset('omarkamali/emoji-map')['train']
    emoji_dict = {}
    for row in emoji_map_dataset:
        emoji_char = row['emoji']
        desc = row['description_kor_Hang']
        if desc:
            emoji_dict[emoji_char] = desc

    # 데이터셋 초기화
    train_dataset = UnifiedTextEmojiDataset(train_data, tokenizer, emoji_dict=emoji_dict)
    val_dataset = UnifiedTextEmojiDataset(val_data, tokenizer, emoji_dict=emoji_dict)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 데이터셋 샘플 확인
    check_dataset(train_dataset, emoji_dict, tokenizer)

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=4e-5)
    save_path = './kobart-1214-4e5'

    # 모델 학습
    train_unified_model(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=5,
        beta=1.0,
        gamma=1.0,
        device=device,
        save_path=save_path,
        patience=3,
        warmup_steps=500
    )

if __name__ == "__main__":
    main()
