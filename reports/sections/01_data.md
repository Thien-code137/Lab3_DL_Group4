# 01 Dữ liệu và tiền xử lý

## Bộ dữ liệu IMDb

Nhóm sử dụng `stanfordnlp/imdb` qua thư viện `datasets`. Hai nhãn: 0 = NEGATIVE và 1 = POSITIVE. Dataset gồm 25.000 văn bản train có nhãn và 25.000 văn bản test có nhãn.

Kết quả EDA trên train: 12.500 mẫu mỗi lớp, 0 missing, 0 empty, 96 dòng có nội dung lặp. Phân tích PCA và độ dài văn bản được thực hiện trong `notebooks/01_eda_pca.ipynb`.

## Tách dữ liệu

1. Chia 25.000 train gốc thành train pool và validation pool theo tỷ lệ 90/10, có phân tầng nhãn, seed 42.
2. Từ train pool chọn 500 mẫu mỗi lớp, tổng 1.000 mẫu.
3. Từ validation pool chọn 100 mẫu mỗi lớp, tổng 200 mẫu.
4. Chọn 200 mẫu từ split test gốc, giữ độc lập với train và validation.

Phần `src/data.py` kiểm tra kích thước các tập và so sánh nội dung giữa train/validation đã lấy mẫu. Dòng trùng trong tập gốc cần được kiểm tra thêm nếu mở rộng số lượng train. Không dùng test để điều chỉnh siêu tham số.

## Tokenization

Tokenizer DistilBERT sinh `input_ids` và `attention_mask`. Dùng `truncation=True`, `max_length=256`; padding động theo batch với `DataCollatorWithPadding`. Giới hạn token giúp giảm RAM nhưng có thể cắt đoạn cuối của review dài.
