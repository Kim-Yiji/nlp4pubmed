# 1. 라이브러리 임포트
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments,
    DataCollatorForTokenClassification, EarlyStoppingCallback
)
from torch.utils.data import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch import nn
from torchcrf import CRF
from sklearn.utils.class_weight import compute_class_weight
from transformers.trainer_callback import PrinterCallback

import numpy as np


# Optionally, set matplotlib backend for headless environments
import matplotlib
# matplotlib.use("Agg")  # Uncomment if running on a headless server

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for debugging

# 3. 라벨 정의 및 매핑
label_list = [
    "O", "B-INGREDIENT", "I-INGREDIENT", "B-DOSAGE", "I-DOSAGE",
    "B-SYMPTOM", "I-SYMPTOM", "B-SENSITIVE_CONDITION", "I-SENSITIVE_CONDITION",
    "B-PERSONAL_INFO", "I-PERSONAL_INFO"
]
label_to_id = {label: i for i, label in enumerate(label_list)}

# 4. 토크나이저
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# 6. 인코딩 + 레이블 정렬 함수
def encode_and_align_labels(data):
    tokens = [d['tokens'] for d in data]
    labels = [[label_to_id[tag] for tag in d['labels']] for d in data]
    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
    encodings.pop("offset_mapping")

    labels_aligned = []
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                if label[word_idx] % 2 == 1:  # I-tag
                    label_ids.append(label[word_idx])
                else:
                    if label[word_idx] + 1 < len(label_list):
                        label_ids.append(label[word_idx] + 1)
                    else:
                        label_ids.append(label[word_idx])  # fallback: use as is
            previous_word_idx = word_idx
        labels_aligned.append(label_ids)
    return encodings, labels_aligned

# 7. Dataset 정의
class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item
    def __len__(self):

        return len(self.labels)

# 8. CRF 모델 정의
class BioBERT_CRF(nn.Module):
    def __init__(self, model_name, num_labels, class_weights=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)
        if labels is not None:
            if self.class_weights is not None:
                # 가중치 기반 손실 수동 적용 (수정된 버전)
                loss_mask = attention_mask.bool()
                crf_mask = loss_mask
                weights = self.class_weights.to(device)
                weights_per_token = torch.zeros_like(labels, dtype=torch.float, device=labels.device)
                valid_mask = labels != -100
                weights_per_token[valid_mask] = weights[labels[valid_mask]]
                weights_per_token = weights_per_token * loss_mask
                weights_per_token = weights_per_token * loss_mask
                labels_fixed = labels.clone()
                labels_fixed[labels_fixed == -100] = 0  # Replace -100 with 0
                log_likelihood = self.crf(emissions, labels_fixed, mask=crf_mask, reduction='none')  # [B]
                loss = -(log_likelihood.mean())
            else:
                labels_fixed = labels.clone()
                labels_fixed[labels_fixed == -100] = 0
                loss = -self.crf(emissions, labels_fixed, mask=attention_mask.bool(), reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            pred = self.crf.decode(emissions, mask=attention_mask.bool())
            return {"logits": pred}

# 9. 평가 메트릭 정의
def compute_metrics(p):
    predictions, labels = p
    if isinstance(predictions[0][0], int):  # CRF decode 결과
        decoded_preds = predictions
    else:
        decoded_preds = predictions.argmax(axis=-1)

    true_labels = []
    true_preds = []
    for pred, label in zip(decoded_preds, labels):
        pred_label = []
        true_label = []
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                pred_label.append(label_list[p_i])
                true_label.append(label_list[l_i])
        true_preds.append(pred_label)
        true_labels.append(true_label)

    print("\n" + classification_report(true_labels, true_preds))

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds)
    }

if __name__ == "__main__":
    import csv

    log_file = "./modeling/Database/NER/training_log.csv"
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Precision", "Recall", "F1"])
    # 2. 데이터 로드
    with open("./modeling/Database/NER/ner_bio_format_results.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 5. train/test 분할
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 5-2. 소수 클래스 증강 (MASK 기반 증강)
    import random
    from copy import deepcopy

    def mask_augment(example, mask_prob=0.15):
        new_example = deepcopy(example)
        for i in range(len(new_example['tokens'])):
            if random.random() < mask_prob and new_example['labels'][i] != 'O':
                new_example['tokens'][i] = '[MASK]'
        return new_example

    rare_labels = {"B-PERSONAL_INFO", "I-PERSONAL_INFO", "B-SENSITIVE_CONDITION", "I-SENSITIVE_CONDITION"}
    augmented_data = []
    for d in train_data:
        if any(tag in rare_labels for tag in d["labels"]):
            augmented = mask_augment(d)
            augmented_data.append(augmented)
    print(f"🔁 희소 클래스 포함 문장 {len(augmented_data)}개 증강됨 (MASK 기반)")
    train_data.extend(augmented_data)

    # 5-1. 클래스 가중치 계산
    all_train_labels = [label_to_id[label] for d in train_data for label in d["labels"]]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(label_to_id.values())),
        y=all_train_labels
    )
    try:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    except RuntimeError as e:
        print("⚠️ class_weights를 GPU에 올리는 중 오류 발생. CPU로 전환합니다.")
        print(e)
        device = torch.device("cpu")
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("📊 클래스 가중치:", class_weights_tensor)

    train_encodings, train_labels = encode_and_align_labels(train_data)
    test_encodings, test_labels = encode_and_align_labels(test_data)

    train_dataset = NERDataset(train_encodings, train_labels)
    test_dataset = NERDataset(test_encodings, test_labels)

    if class_weights_tensor.device.type == "cpu":
        device = torch.device("cpu")
    model = BioBERT_CRF("dmis-lab/biobert-base-cased-v1.1", num_labels=len(label_list), class_weights=class_weights_tensor)
    print("🔍 Debug - Model device move starting...")
    for name, param in model.named_parameters():
        print(f"{name} - requires_grad: {param.requires_grad} | shape: {param.shape} | device: {param.device}")
    try:
        model.to(device)
    except RuntimeError as e:
        print("🔥 Error during model.to(device):", e)
        raise

    # 10. 학습 설정 (early stopping 포함)
    training_args = TrainingArguments(
        output_dir="./modeling/biobert_ner_output",
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./modeling/logs",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_safetensors=False
    )

    # 11. Trainer 정의 + EarlyStopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 12. 학습 및 저장
    # Enable logging of training loss every epoch
    training_args.logging_strategy = "epoch"
    training_args.logging_first_step = True
    trainer.add_callback(PrinterCallback())
    trainer.train()
    # Save training loss from state.log_history
    training_loss_log = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry and "eval_loss" not in entry]
    print("📉 Training Loss Log:", training_loss_log)
    # Save training loss log to file
    with open("./modeling/Database/NER/training_loss_log.txt", "w") as f:
        for i, loss in enumerate(training_loss_log):
            f.write(f"Epoch {i + 1}: Training Loss = {loss:.6f}\n")
    print("📁 training_loss_log.txt 저장 완료")
    # Extract and save metrics from trainer.state.log_history
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        for entry in trainer.state.log_history:
            if "eval_loss" in entry or "loss" in entry:
                writer.writerow([
                    entry.get("epoch", ""),
                    entry.get("loss", ""),
                    entry.get("eval_loss", ""),
                    entry.get("eval_precision", ""),
                    entry.get("eval_recall", ""),
                    entry.get("eval_f1", "")
                ])
    trainer.save_model("./modeling/biobert_ner_model")
    print("✅ 학습 및 모델 저장 완료")

    # 13. 예측 후처리 예시
    predictions = trainer.predict(test_dataset).predictions
    decoded_preds = predictions if isinstance(predictions[0][0], int) else predictions.argmax(axis=-1)

    print("\n🎯 예측 결과 예시")
    for i in range(3):
        print(f"\n[문장 {i+1}]")
        result = []
        for p_i, l_i in zip(decoded_preds[i], test_labels[i]):
            if l_i != -100:
                result.append(f"PRED: {label_list[p_i]:<25} | TRUE: {label_list[l_i]}")
        print("\n".join(result))

# import matplotlib.pyplot as plt

# # 에폭별 로그 데이터
# epochs = list(range(1, 10))
# train_loss = [None, None, None, 19.206800, 19.206800, 19.206800, 1.961100, 1.961100, 1.961100]
# val_loss = [22.482336, 20.139780, 18.260933, 22.373323, 25.521544, 27.804270, 31.009399, 34.640724, 40.719376]
# precision = [0.805605, 0.834903, 0.876247, 0.878585, 0.876222, 0.884798, 0.875803, 0.886179, 0.883676]
# recall =    [0.795459, 0.864822, 0.888061, 0.885755, 0.890367, 0.891077, 0.894447, 0.889480, 0.885400]
# f1 =        [0.800500, 0.849599, 0.882115, 0.882155, 0.883238, 0.887926, 0.885027, 0.887826, 0.884537]

# # 그래프 크기 설정
# plt.figure(figsize=(12, 6))

# # Loss 그래프
# plt.subplot(1, 2, 1)
# # plt.plot(epochs[3:], train_loss[3:], label='Train Loss', marker='o')
# plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Validation Loss Over Epoch")
# plt.legend()
# plt.grid(True)

# # Precision / Recall / F1 그래프
# plt.subplot(1, 2, 2)
# plt.plot(epochs, precision, label='Precision', marker='o')
# plt.plot(epochs, recall, label='Recall', marker='o')
# plt.plot(epochs, f1, label='F1 Score', marker='o')
# plt.xlabel("Epoch")
# plt.ylabel("Score")
# plt.title("Precision / Recall / F1 over Epochs")
# plt.ylim(0.75, 0.95)
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# all_labels = list(set([int(label) for batch in train_dataset.labels for label in batch]))
# print(sorted(all_labels))

# flat_train_labels = [label for sent in train_labels for label in sent if label != -100]
# print("✅ max label:", max(flat_train_labels))
# print("✅ min label:", min(flat_train_labels))
# print("✅ num_labels:", len(label_list))

# all_labels_in_data = set(l for d in train_data for l in d["labels"])
# print("📌 전체 라벨 종류:", sorted(all_labels_in_data))
# for label in all_labels_in_data:
#     if label not in label_to_id:
#         print(f"⚠️ 존재하지 않는 라벨: {label}")

# ## The following are not valid Python syntax and should be run in your terminal if needed:
# # pip install seqeval pytorch-crf

