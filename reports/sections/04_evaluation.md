# 04 Đánh giá

## Chỉ số

- Accuracy: tỷ lệ dự đoán đúng.
- Precision lớp Positive: tỷ lệ nhãn Positive dự đoán đúng trong các dự đoán Positive.
- Recall lớp Positive: tỷ lệ mẫu Positive thật được mô hình tìm thấy.
- F1-score: trung bình điều hòa giữa precision và recall.
- Confusion matrix: bảng các dự đoán đúng và sai theo hai nhãn.

## Kết quả thực nghiệm đã lưu

Notebook Exercise 2 gốc có kết quả trên 200 mẫu: accuracy 89%, precision Positive 89,38%, recall Positive 90,99%, F1 Positive 90,18%. Confusion matrix gồm 77 TN, 12 FP, 10 FN và 101 TP.

Đây là kết quả của notebook đã chạy, không phải kết quả tự động của pipeline trong `src/train.py`. Để có chỉ số của pipeline module, chạy `python main.py` và lấy `outputs/new_run/test_metrics.json`.

## Phân tích giới hạn

Tập test con 200 mẫu có thể tạo sai lệch do lấy mẫu; dữ liệu IMDb cũng không đại diện cho mọi loại bình luận. Không nên sử dụng kết quả của test để quyết định learning rate, epoch hoặc checkpoint.
