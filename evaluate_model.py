from sklearn.metrics import accuracy_score, classification_report
import torch

# Hàm đánh giá mô hình
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = attention_mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Đánh giá kết quả
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["negative", "neutral", "positive"])
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(report)
    
    return accuracy  # Trả về độ chính xác để sử dụng trong huấn luyện
