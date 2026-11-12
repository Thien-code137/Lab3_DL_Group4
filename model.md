# Mô tả mô hình – Lab 3

## 1. Phạm vi

Lab 3 có hai bài thực hành liên quan đến phân tích cảm xúc, nhưng mục đích sử dụng mô hình ở hai bài không giống nhau:

- **Exercise 1 – Sentiment Analysis:** sử dụng mô hình `distilbert-base-uncased-finetuned-sst-2-english` đã được fine-tune sẵn trên SST-2 để suy luận nhãn cảm xúc. Bài này chủ yếu thực hiện inference, không huấn luyện lại mô hình.
- **Exercise 2 – Fine-tuning:** sử dụng `distilbert-base-uncased` làm mô hình pretrained ban đầu, sau đó fine-tune cho bài toán phân loại cảm xúc nhị phân trên dữ liệu IMDb.

Cần giữ riêng kết quả và mục tiêu của hai bài, không gộp chúng thành một thí nghiệm duy nhất.

## 2. Kiến trúc DistilBERT

DistilBERT là mô hình Transformer encoder được rút gọn từ BERT. So với BERT-base, DistilBERT-base dùng 6 Transformer encoder layers, hidden size 768 và 12 attention heads. Mô hình dùng cơ chế self-attention hai chiều để tạo biểu diễn ngữ cảnh cho các token trong câu.

Luồng xử lý khái quát:

1. Văn bản được tokenizer tách thành token.
2. Tokenizer chuyển token thành `input_ids` và tạo `attention_mask` để phân biệt token thật với vị trí padding.
3. Các embedding được đưa qua các Transformer encoder layers.
4. Với bài toán phân loại câu, biểu diễn tại token đầu `[CLS]` được đưa vào classification head.
5. Classification head tạo logits cho hai lớp `NEGATIVE` và `POSITIVE`.

## 3. Tokenizer và đầu vào

Exercise 2 sử dụng tokenizer đi kèm checkpoint `distilbert-base-uncased`. Văn bản được chuyển thành `input_ids` và `attention_mask`. Các review được padding và cắt theo độ dài tối đa được khai báo trong notebook. Khi báo cáo thông số độ dài tối đa, cần lấy đúng giá trị từ cell cấu hình của notebook, không tự suy đoán.

Nhãn trong bài IMDb được ánh xạ như sau:

- `0`: `NEGATIVE`
- `1`: `POSITIVE`

## 4. Thiết lập cho từng bài

### 4.1. Exercise 1 – Inference với SST-2

- Checkpoint: `distilbert-base-uncased-finetuned-sst-2-english`.
- Mục tiêu: dự đoán cảm xúc cho câu/review đầu vào bằng mô hình đã được fine-tune sẵn.
- Không thực hiện fine-tuning lại trong Exercise 1.

### 4.2. Exercise 2 – Fine-tuning trên IMDb

- Checkpoint khởi tạo: `distilbert-base-uncased`.
- Bài toán: phân loại review IMDb thành hai lớp `NEGATIVE` và `POSITIVE`.
- Quy trình: tokenize dữ liệu, tạo tập train/validation/test theo notebook, fine-tune bằng Hugging Face Trainer, sau đó đánh giá trên test set.
- Notebook đã lưu kết quả đánh giá và dự đoán trong `outputs/exercise2/`.

Các thông số như learning rate, batch size, số epoch, max length và seed phải được trích từ cell cấu hình trong `Lab3_Exercise2_FineTunning.ipynb`. Nếu có thay đổi so với lần chạy đã tạo ra kết quả hiện tại, cần ghi rõ thay đổi và không gán kết quả cũ cho cấu hình mới.

## 5. Kiểm tra số lượng tham số

Không nên ghi một con số tham số ước lượng nếu chưa kiểm tra đúng checkpoint và classification head đang dùng. Có thể kiểm tra trực tiếp bằng đoạn code sau sau khi khởi tạo model trong notebook:

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
```

Khi ghi kết quả vào báo cáo, cần nêu rõ đang đếm model nào (Exercise 1 hay Exercise 2), checkpoint nào và có tính classification head hay không.

## 6. Kiểm tra và lưu mô hình

Với Exercise 2, checkpoint được lựa chọn theo metric đã cấu hình trong Hugging Face Trainer. Báo cáo hiện tại ghi nhận checkpoint tốt nhất là `checkpoint-125` theo validation accuracy. Validation loss thấp nhất được ghi nhận ở Epoch 1 là `0.283972`; đây là một tiêu chí khác với tiêu chí chọn checkpoint, vì vậy không được diễn đạt rằng checkpoint được chọn theo validation loss nếu cấu hình thực tế chọn theo accuracy.

Bản ZIP được rà soát trước đó không chứa thư mục checkpoint/model weights đã fine-tune. Vì vậy, nếu cần tái sử dụng mô hình đã fine-tune, cần tìm lại checkpoint trên máy hoặc nơi lưu trữ của lần chạy trước. Không thể khôi phục trọng số chỉ từ file CSV/JSON metric.

## 7. Giới hạn

- DistilBERT là mô hình pretrained; kết quả phụ thuộc vào checkpoint, tokenizer, cách chia dữ liệu và cấu hình fine-tuning.
- PCA trong notebook EDA/PCA là bước khám phá biểu diễn TF-IDF của review, không phải một phần của kiến trúc DistilBERT và không cập nhật trọng số mô hình.
- Các kết quả trong `outputs/exercise2/` chỉ nên được coi là kết quả của lần chạy đã tạo ra chúng. Nếu thay đổi dữ liệu hoặc cấu hình, cần ghi nhận riêng và không trộn các kết quả giữa những lần chạy.
