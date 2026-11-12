# RULE.md — Quy tắc thực hiện Lab 3

Tài liệu này ghi lại các quy ước cần tuân thủ khi đọc, chạy, kiểm tra và bổ sung Lab 3. Nội dung được đối chiếu với hai notebook hiện có, `README.md`, `reports/Lab3_Report.md` và notebook EDA/PCA bổ sung.

## 1. Phạm vi bài thực hành

Lab 3 gồm hai nội dung chính:

- **Exercise 1 – Sentiment Analysis:** sử dụng mô hình DistilBERT đã fine-tune trên SST-2 để suy luận cảm xúc. Exercise này không huấn luyện lại mô hình.
- **Exercise 2 – Fine-tuning:** fine-tune mô hình DistilBERT pretrained trên dữ liệu IMDb để phân loại cảm xúc thành hai lớp.
- **EDA/PCA:** khám phá tập train IMDb và trực quan hóa TF-IDF bằng PCA. Đây là bước phân tích dữ liệu, không phải bước huấn luyện bộ phân loại.

Không gộp kết quả Exercise 1 và Exercise 2 thành một thí nghiệm duy nhất vì chúng sử dụng mô hình và mục đích khác nhau.

## 2. Nguồn dữ liệu và nhãn

- Dataset: `stanfordnlp/imdb`.
- Nhãn `0`: `NEGATIVE`.
- Nhãn `1`: `POSITIVE`.
- EDA/PCA chỉ sử dụng split `train` gốc; không sử dụng split `test` để khám phá hoặc lựa chọn đặc trưng.
- Trong PCA, lấy mẫu phân tầng cố định 500 review mỗi lớp (tổng 1.000 review), với random seed `42`.
- Không tự ý đổi nguồn dữ liệu, ánh xạ nhãn hoặc số lượng mẫu trong báo cáo nếu chưa cập nhật notebook và ghi lại thay đổi.

## 3. Quy tắc tiền xử lý văn bản

- Giữ nguyên văn bản gốc để có thể truy vết về dữ liệu nguồn.
- Khi tính thống kê EDA, tạo bản làm sạch riêng: giải mã HTML entities, thay thẻ xuống dòng/HTML bằng khoảng trắng và chuẩn hóa khoảng trắng.
- Ghi rõ cách xử lý missing values và duplicate reviews; không âm thầm xóa mẫu mà không ghi lại số lượng trước/sau.
- Với fine-tuning Exercise 2, dùng tokenizer tương ứng với `distilbert/distilbert-base-uncased`.
- Độ dài tối đa đầu vào của Exercise 2 là 256 token; padding theo batch (dynamic padding) theo collator trong notebook.
- Không dùng tập test để fit tokenizer, chọn checkpoint, điều chỉnh siêu tham số hoặc quyết định cấu hình.

## 4. Quy tắc EDA và PCA

- Báo cáo số lượng mẫu, phân bố nhãn, missing values, duplicate texts và thống kê độ dài review.
- Biểu đồ cần có tiêu đề, nhãn trục, chú giải khi cần và ghi rõ split dữ liệu được sử dụng.
- PCA hiện tại chạy trên đặc trưng TF-IDF của 1.000 review train đã lấy mẫu cân bằng; không mô tả đây là embedding của DistilBERT.
- Cấu hình TF-IDF: `max_features=1000`, `min_df=3`, `max_df=0.95`, `stop_words="english"`, `sublinear_tf=True`.
- PCA dùng 2 thành phần chính và random seed `42`.
- Khi diễn giải PCA, ghi tỷ lệ phương sai giải thích của PC1, PC2 và tổng hai thành phần theo output thực tế. Không kết luận hai lớp tách biệt hoàn toàn nếu biểu đồ không chứng minh điều đó.

## 5. Quy tắc fine-tuning Exercise 2

Các giá trị dưới đây là cấu hình của lần thực nghiệm đã được lưu trong notebook/báo cáo; không tự xem chúng là cấu hình bắt buộc cho mọi lần chạy mới.

- Base model: `distilbert/distilbert-base-uncased`.
- Số nhãn: `2` (`NEGATIVE`, `POSITIVE`).
- Train thực tế truyền vào Trainer: 1.000 mẫu; validation: 200 mẫu.
- Test cuối: 200 mẫu lấy từ split test gốc; phân bố nhãn của kết quả đã lưu là 89 NEGATIVE và 111 POSITIVE.
- Max length: `256`.
- Epoch: `2`.
- Batch size train/evaluation: `8` / `8`.
- Learning rate: `2e-5`.
- Weight decay: `0.01`.
- Optimizer: `adamw_torch` (AdamW của PyTorch).
- Evaluation và save theo epoch.
- Tiêu chí chọn checkpoint: validation accuracy, ưu tiên giá trị cao hơn (`greater_is_better=True`).
- Seed trong cấu hình chia dữ liệu/TrainingArguments: `42`.

Lưu ý: notebook có cell thay đổi một số biến dữ liệu sau khi đã tạo tokenized train/validation. Vì vậy phải căn cứ vào dataset thực sự được truyền vào Trainer và output đã lưu, không lấy các biến tạm ở cell sau làm kích thước train/validation cuối cùng. Khi chỉnh sửa notebook, nên làm rõ tên biến và thứ tự để tránh nhầm lẫn.

## 6. Đánh giá và chọn checkpoint

- Chỉ đánh giá cuối cùng trên test sau khi đã chọn checkpoint bằng validation.
- Báo cáo ít nhất Accuracy, Precision, Recall, F1 và Confusion Matrix.
- Ghi rõ lớp dương khi tính Precision/Recall/F1 nhị phân là `POSITIVE` (label `1`).
- Phân biệt F1 của lớp POSITIVE, macro-F1 và weighted-F1; không dùng các tên này thay thế cho nhau.
- Kết quả hiện có: validation accuracy epoch 1 là 0.905 và epoch 2 là 0.895; validation loss lần lượt là 0.283972 và 0.299779.
- Checkpoint `checkpoint-125` được chọn theo validation accuracy. Không mô tả checkpoint này là được chọn theo validation loss.
- Kết quả test đã lưu: 200 mẫu, accuracy 0.89; các metric chi tiết lấy từ `outputs/exercise2/test_metrics.json` và `classification_report.txt`.
- Không khẳng định kết quả tổng quát cho toàn bộ IMDb hoặc miền dữ liệu khác từ tập test con 200 mẫu.

## 7. Quy tắc lưu file và bằng chứng

- Giữ nguyên hai notebook gốc Exercise 1 và Exercise 2; không xóa output đã có.
- Lưu notebook EDA/PCA sau khi chạy bằng `Ctrl+S`, giữ output và biểu đồ trong notebook.
- Lưu các bảng/biểu đồ EDA/PCA dưới `outputs/eda_pca/`.
- Lưu metrics, confusion matrix và predictions Exercise 2 dưới `outputs/exercise2/`.
- Ghi nguồn của mọi con số trong README/báo cáo (notebook cell, CSV, JSON hoặc log tương ứng).
- Không tạo số liệu giả để lấp chỗ trống. Nếu không có log/checkpoint gốc thì ghi rõ chưa có bằng chứng.
- Không đưa `.venv`, cache Python hoặc dữ liệu lớn/checkpoint chưa được nhóm thống nhất lên Git.

## 8. Quy tắc môi trường và chạy lại

- Dùng một môi trường Python riêng cho Lab 3 và ghi lại phiên bản Python cùng các thư viện thực tế đã dùng.
- Không cài thư viện trực tiếp vào Python hệ thống nếu đã có môi trường ảo.
- Notebook EDA/PCA có thể chạy độc lập sau khi cài các thư viện cần thiết; không cần chạy lại Trainer.
- Chỉ chạy lại fine-tuning khi nhóm yêu cầu hoặc khi cần tái tạo checkpoint; trước khi chạy phải kiểm tra CPU/RAM, thời gian và nơi lưu checkpoint.
- Sau khi thay đổi code/notebook, chạy lại phần bị ảnh hưởng, lưu output và cập nhật audit log tương ứng.

## 9. Nguyên tắc báo cáo

- Phân biệt rõ: (1) nội dung đã có trong notebook, (2) kết quả đã được lưu từ lần thực nghiệm trước, và (3) nội dung vừa bổ sung/chạy lại.
- Không ghi “đã chạy thành công” nếu chỉ mới đọc code hoặc chưa có output.
- Với PCA, nêu rõ TF-IDF là biểu diễn đầu vào; không gọi là embedding DistilBERT.
- Với model checkpoint, phân biệt checkpoint được chọn theo validation accuracy với giá trị best validation loss.
- Mọi sửa đổi cần có mô tả ngắn trong `audit_logs/` để thành viên khác truy vết được.
