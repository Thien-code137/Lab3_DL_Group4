# 02 Kiến trúc mô hình

## Exercise 1

Checkpoint `distilbert/distilbert-base-uncased-finetuned-sst-2-english` đã được fine-tune cho phân tích cảm xúc. Exercise này chỉ sử dụng tokenizer, forward pass và softmax để suy luận, không cập nhật trọng số.

## Exercise 2

Checkpoint `distilbert/distilbert-base-uncased` là encoder pretrained. `AutoModelForSequenceClassification(num_labels=2)` gắn đầu phân loại cho hai lớp IMDb. Khi fine-tune, encoder và đầu phân loại đều có thể cập nhật trọng số.

Luồng xử lý: văn bản → tokenizer → `input_ids`/`attention_mask` → DistilBERT → logits 2 lớp → cross-entropy (train) hoặc softmax/argmax (dự đoán).

## Lý do lựa chọn

DistilBERT nhẹ hơn BERT-base, phù hợp với bài thực hành cần fine-tune trên CPU. Đổi lại, kết quả có thể bị giới hạn bởi kích thước mẫu nhỏ, số epoch và độ dài chuỗi.
