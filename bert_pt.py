import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# 원본 데이터 불러오기
df = pd.read_csv('C:/kdt_jss/workspace/m5_빅데이터/2차 프로젝트2/train_pt.csv')
x = list(df['sentence'].values)
y = list(df['label'].values)

# 학습 데이터와 검증 데이터로 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(x, y, test_size = 0.2, random_state = 11)

# 학습 데이터와 검증 데이터를 토크나이징하여 인코딩
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# 학습 데이터와 검증 데이터를 텐서로 변환
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              torch.tensor(train_labels))
val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                            torch.tensor(val_encodings['attention_mask']),
                            torch.tensor(val_labels))

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 16)

# BERT 모델 불러오기
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels = 2)
model.to(device)

# 옵티마이저 및 스케줄러 설정
optimizer = AdamW(model.parameters(), lr = 5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_loader)*3)

# 손실 함수 설정
criterion = torch.nn.CrossEntropyLoss()

# 학습 함수 정의
def train(model, train_loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(train_loader)

# 검증 함수 정의
def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim = 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(val_loader), accuracy

# 학습 및 검증
num_epochs = 8
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, scheduler, criterion)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss - {train_loss:.4f}, Val Loss - {val_loss:.4f}, Val Accuracy - {val_accuracy:.4f}")

# 학습된 모델 저장
torch.save(model.state_dict(), 'bert_pt_model.pth')

# 저장된 모델 불러오기
model_path = 'C:/kdt_jss/workspace/m5_빅데이터/2차 프로젝트2/bert_pt_model.pth'
model.load_state_dict(torch.load(model_path))
model.to(device)