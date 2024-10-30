from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import torch
from torch.optim import AdamW
from data_utils import get_data_loaders
from evaluate_model import evaluate

# Định nghĩa Focal Loss với alpha cho từng lớp
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha else [1, 1, 1]  # Đặt alpha mặc định cho từng lớp nếu không có
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Sử dụng alpha cho từng lớp
        at = torch.tensor(self.alpha, device=inputs.device)[targets]
        focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Load PhoBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Sử dụng AdamW optimizer
optimizer = AdamW(model.parameters(), lr=5e-6)  # Giảm learning rate

# Cấu hình FocalLoss với alpha điều chỉnh cho lớp neutral
criterion = FocalLoss(alpha=[0.5, 2.0, 1.0], gamma=2.5)  # Tăng trọng số cho lớp neutral

# Hàm huấn luyện
def train(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Khởi tạo DataLoader và huấn luyện
train_loader, test_loader = get_data_loaders()
train(model, train_loader, optimizer, criterion, epochs=5)

# Đánh giá mô hình sau khi huấn luyện xong
evaluate(model, test_loader)

# Lưu mô hình sau khi huấn luyện xong
model_save_path = r"D:\ABSAPhoBert\train_model"
model.save_pretrained(model_save_path)
print(f"Mô hình đã được lưu vào {model_save_path}")
