import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm
import pandas as pd
import os
import math
from torch.nn import CrossEntropyLoss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
from bert_score import score as bert_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# -----------------------------
# 1. 테스트 데이터셋 클래스 정의 (학습 포맷과 동일하게)
# -----------------------------
class TextEmojiTestDataset(Dataset):
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
        emoji_str = row['emoji']

        # 이모지 토크나이징
        emojis = self.tokenizer.tokenize(emoji_str)
        emoji_descriptions = [self.emoji_dict.get(e, "") for e in emojis]

        # Text→Emoji 입력 및 라벨
        text_to_emoji_input = "[TEXT2EMOJI] " + text
        text_to_emoji_enc = self.tokenizer(
            text_to_emoji_input,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )
        text_to_emoji_target = " ".join(emojis) + self.tokenizer.eos_token
        text_to_emoji_dec = self.tokenizer(
            text_to_emoji_target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        text_to_emoji_labels = text_to_emoji_dec["input_ids"].clone()
        text_to_emoji_labels[text_to_emoji_labels == self.tokenizer.pad_token_id] = -100

        # Emoji→Text 입력 및 라벨
        emoji_to_text_input = "[EMOJI2TEXT] " + " ".join(emojis)
        if any(desc for desc in emoji_descriptions):
            emoji_to_text_input += " [EMOJI_DESC] " + " ".join(emoji_descriptions)
        emoji_to_text_enc = self.tokenizer(
            emoji_to_text_input,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )
        emoji_to_text_target = text + self.tokenizer.eos_token
        emoji_to_text_dec = self.tokenizer(
            emoji_to_text_target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        emoji_to_text_labels = emoji_to_text_dec["input_ids"].clone()
        emoji_to_text_labels[emoji_to_text_labels == self.tokenizer.pad_token_id] = -100

        return {
            "text": text,
            "emoji": emoji_str,
            "text_to_emoji_input_ids": text_to_emoji_enc["input_ids"].squeeze(),
            "text_to_emoji_attention_mask": text_to_emoji_enc["attention_mask"].squeeze(),
            "text_to_emoji_labels": text_to_emoji_labels.squeeze(),
            "emoji_to_text_input_ids": emoji_to_text_enc["input_ids"].squeeze(),
            "emoji_to_text_attention_mask": emoji_to_text_enc["attention_mask"].squeeze(),
            "emoji_to_text_labels": emoji_to_text_labels.squeeze(),
        }

# -----------------------------
# 2. 평가 함수 정의
# -----------------------------
def evaluate_model(model, tokenizer, test_loader, device, output_dir):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Text-to-Emoji Task 예측
            text_to_emoji_input_ids = batch["text_to_emoji_input_ids"].to(device)
            text_to_emoji_attention_mask = batch["text_to_emoji_attention_mask"].to(device)

            emoji_outputs = model.generate(
                input_ids=text_to_emoji_input_ids,
                attention_mask=text_to_emoji_attention_mask,
                max_new_tokens=15,
                num_beams=5,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=2,
                length_penalty=1.2,
                top_k=30,
                top_p=0.9,
            )
            text_to_emoji_preds = tokenizer.batch_decode(
                emoji_outputs,
                skip_special_tokens=True,
            )

            # Emoji-to-Text Task 예측
            emoji_to_text_input_ids = batch["emoji_to_text_input_ids"].to(device)
            emoji_to_text_attention_mask = batch["emoji_to_text_attention_mask"].to(device)

            text_outputs = model.generate(
                input_ids=emoji_to_text_input_ids,
                attention_mask=emoji_to_text_attention_mask,
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
            emoji_to_text_preds = tokenizer.batch_decode(
                text_outputs,
                skip_special_tokens=True
            )

            for i in range(len(text_to_emoji_preds)):
                text_to_emoji_pred = text_to_emoji_preds[i].strip()
                emoji_to_text_pred = emoji_to_text_preds[i].strip()

                result = {
                    "text": batch["text"][i],
                    "emoji": batch["emoji"][i],
                    "text_to_emoji_pred": text_to_emoji_pred,
                    "emoji_to_text_pred": emoji_to_text_pred,
                }
                results.append(result)

    # 결과 저장
    results_df = pd.DataFrame(results)
    text_to_emoji_results = results_df[['text', 'text_to_emoji_pred', 'emoji']]
    emoji_to_text_results = results_df[['emoji', 'emoji_to_text_pred', 'text']]

    text_to_emoji_results.to_csv(os.path.join(output_dir, "text_to_emoji_results.csv"), index=False, encoding="utf-8-sig")
    emoji_to_text_results.to_csv(os.path.join(output_dir, "emoji_to_text_results.csv"), index=False, encoding="utf-8-sig")
    print(f"Results saved to {output_dir}")

    return results_df

def calculate_perplexity_for_task(model, tokenizer, test_loader, device, input_ids_key, attention_mask_key, labels_key):
    model.eval()
    loss_fn = CrossEntropyLoss(ignore_index=-100)  # 라벨에서 -100이 ignore index
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Calculating Perplexity for {input_ids_key}"):
            input_ids = batch[input_ids_key].to(device)
            attention_mask = batch[attention_mask_key].to(device)
            labels = batch[labels_key].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            active_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens

    if total_tokens > 0:
        overall_perplexity = math.exp(total_loss / total_tokens)
    else:
        overall_perplexity = float('inf')

    return overall_perplexity

def calculate_bleu_for_task(results_df, reference_col, hypothesis_col):
    bleu_scores = []
    smoothing_function = SmoothingFunction().method4

    for _, row in results_df.iterrows():
        reference = row[reference_col].split()
        hypothesis = row[hypothesis_col].split()

        if not hypothesis or not reference:
            bleu_scores.append(0)
            continue

        try:
            score = sentence_bleu([reference], hypothesis, smoothing_function=smoothing_function)
            bleu_scores.append(score)
        except Exception:
            bleu_scores.append(0)

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if len(bleu_scores) > 0 else 0
    return avg_bleu

def calculate_bertscore_for_task(results_df, reference_col, hypothesis_col, model_type='bert-base-multilingual-cased'):
    references = results_df[reference_col].tolist()
    candidates = results_df[hypothesis_col].tolist()

    # BERTScore 계산
    P, R, F1 = bert_score(candidates, references, model_type=model_type)
    return float(P.mean()), float(R.mean()), float(F1.mean())

# -----------------------------
# 3. 메인 테스트 코드
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = './kobart-with-emoji-map-dict2-4e5'
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)

    if tokenizer.eos_token is None:
        tokenizer.eos_token = '</s>'

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    test_data = pd.read_csv("final_test_dataset.csv")
    if 'text' not in test_data.columns or 'emoji' not in test_data.columns:
        raise ValueError("Test data must contain 'text' and 'emoji' columns.")

    emoji_map_dataset = load_dataset('omarkamali/emoji-map')['train']
    emoji_dict = {}
    for row in emoji_map_dataset:
        if row['description_kor_Hang']:
            emoji_dict[row['emoji']] = row['description_kor_Hang']

    output_dir = './kobart-with-emoji-map-dict2-4e5'
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = TextEmojiTestDataset(test_data, tokenizer, emoji_dict=emoji_dict)
    test_loader = DataLoader(test_dataset, batch_size=16)

    results_df = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
    )

    # Perplexity 계산
    avg_perplexity_text2emoji = calculate_perplexity_for_task(
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        device=device,
        input_ids_key="text_to_emoji_input_ids",
        attention_mask_key="text_to_emoji_attention_mask",
        labels_key="text_to_emoji_labels"
    )

    avg_perplexity_emoji2text = calculate_perplexity_for_task(
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        device=device,
        input_ids_key="emoji_to_text_input_ids",
        attention_mask_key="emoji_to_text_attention_mask",
        labels_key="emoji_to_text_labels"
    )

    # BLEU 계산
    text2emoji_bleu = calculate_bleu_for_task(results_df, reference_col='emoji', hypothesis_col='text_to_emoji_pred')
    emoji2text_bleu = calculate_bleu_for_task(results_df, reference_col='text', hypothesis_col='emoji_to_text_pred')

    # BERTScore 계산
    # Emoji->Text 태스크(참조: text, 예측: emoji_to_text_pred)에 대해 BERTScore 계산
    # BERTScore는 자연어 비교에 적합하므로 Emoji->Text에 특히 유용할 것임
    p_emoji2text, r_emoji2text, f1_emoji2text = calculate_bertscore_for_task(
        results_df,
        reference_col='text',
        hypothesis_col='emoji_to_text_pred',
        model_type='bert-base-multilingual-cased'
    )

    # Text->Emoji 태스크(참조: emoji, 예측: text_to_emoji_pred)에 대한 BERTScore 계산 시도
    # 이모지 시퀀스에 대한 의미적 평가 한계는 있지만, 참고용으로 계산
    p_text2emoji, r_text2emoji, f1_text2emoji = calculate_bertscore_for_task(
        results_df,
        reference_col='emoji',
        hypothesis_col='text_to_emoji_pred',
        model_type='bert-base-multilingual-cased'
    )

    print(f"Final Average Perplexity (Text->Emoji): {avg_perplexity_text2emoji}")
    print(f"Final Average Perplexity (Emoji->Text): {avg_perplexity_emoji2text}")
    print(f"Final Average BLEU (Text->Emoji): {text2emoji_bleu}")
    print(f"Final Average BLEU (Emoji->Text): {emoji2text_bleu}")

    print("[BERTScore - Emoji->Text]")
    print(f"Precision: {p_emoji2text:.4f}, Recall: {r_emoji2text:.4f}, F1: {f1_emoji2text:.4f}")

    print("[BERTScore - Text->Emoji]")
    print(f"Precision: {p_text2emoji:.4f}, Recall: {r_text2emoji:.4f}, F1: {f1_text2emoji:.4f}")

if __name__ == "__main__":
    main()
