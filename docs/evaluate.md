# EVALUATE.PY – ĐÁNH GIÁ VÀ XUẤT KẾT QUẢ MÔ HÌNH

## 1. Mục đích

File `src/evaluate.py` đảm nhiệm việc đánh giá khả năng phân loại cảm xúc của mô hình trên dữ liệu validation và test.

Các chỉ số được sử dụng gồm Accuracy, Precision, Recall và F1-score. Ngoài ra, chương trình còn xuất Confusion Matrix, Classification Report và kết quả dự đoán từng review để phục vụ việc phân tích hiệu suất mô hình.

## 2. Các thư viện sử dụng

- `csv`: Ghi kết quả dự đoán và ma trận nhầm lẫn ra file CSV.
- `pathlib.Path`: Quản lý đường dẫn lưu kết quả.
- `numpy`: Chuyển dữ liệu thành mảng và lấy nhãn dự đoán từ logits.
- `sklearn.metrics`: Tính Accuracy, Precision, Recall, F1-score, Classification Report và Confusion Matrix.
- `matplotlib.pyplot`: Trực quan hóa Confusion Matrix.
- `src.utils.save_json`: Lưu kết quả đánh giá dưới định dạng JSON.

## 3. Các hàm chính

### 3.1. Hàm scores(labels, predictions)

**Mục đích:**

Tính các chỉ số đánh giá dựa trên nhãn thực tế và nhãn dự đoán của mô hình.

**Input:**

- `labels`: Danh sách nhãn thực tế.
- `predictions`: Danh sách nhãn do mô hình dự đoán.

**Quy trình thực hiện:**

1. Chuyển nhãn thực tế và nhãn dự đoán thành NumPy Array.
2. Kiểm tra kích thước hai mảng có bằng nhau hay không.
3. Sử dụng `accuracy_score` để tính Accuracy.
4. Sử dụng `precision_recall_fscore_support` để tính Precision, Recall và F1-score.
5. Chỉ số được tính theo phương pháp Binary Classification, trong đó nhãn 1 (POSITIVE) được xem là lớp dương.
6. Trường hợp một lớp không có dự đoán phù hợp, `zero_division=0` giúp tránh lỗi chia cho 0.

**Output:**

Hàm trả về một dictionary gồm:

- `accuracy`: Tỷ lệ dự đoán chính xác trên tổng số mẫu.
- `precision_positive`: Độ chính xác của các dự đoán thuộc lớp POSITIVE.
- `recall_positive`: Khả năng tìm được các mẫu POSITIVE thực tế.
- `f1_positive`: Trung bình điều hòa giữa Precision và Recall của lớp POSITIVE.

### 3.2. Hàm compute_metrics(eval_prediction)

**Mục đích:**

Tính các chỉ số đánh giá trong quá trình Validation khi sử dụng Hugging Face Trainer.

**Input:**

`eval_prediction` là đối tượng chứa:
- `predictions`: Logits do mô hình tạo ra.
- `label_ids`: Nhãn thực tế tương ứng với dữ liệu đầu vào.

**Quy trình thực hiện:**

1. Sử dụng `np.argmax(..., axis=-1)` để chọn lớp có logit lớn nhất làm nhãn dự đoán.
2. Lấy nhãn thực tế từ `eval_prediction.label_ids`.
3. Gọi hàm `scores()` để tính các chỉ số Accuracy, Precision, Recall và F1-score.

**Output:**

Dictionary chứa các chỉ số đánh giá để Hugging Face Trainer sử dụng trong quá trình Validation.

Các tên metrics được Trainer thêm tiền tố `eval_` khi ghi nhận kết quả đánh giá.

### 3.3. Hàm evaluate_and_save(dataset, logits, output_dir, best_checkpoint, best_val_loss)

**Mục đích:**

Thực hiện đánh giá cuối cùng trên tập Test và lưu lại các kết quả để phục vụ phân tích.

**Input:**

- `dataset`: Tập dữ liệu Test, bao gồm văn bản và nhãn thực tế.
- `logits`: Điểm số đầu ra của mô hình.
- `output_dir`: Đường dẫn lưu kết quả.
- `best_checkpoint`: Thông tin checkpoint được lựa chọn.
- `best_val_loss`: Giá trị Validation Loss tốt nhất được cung cấp từ quá trình huấn luyện.

**Quy trình thực hiện:**

Bước 1: Chuyển nhãn thực tế từ dataset thành NumPy Array.

Bước 2: Sử dụng `argmax` trên logits để xác định nhãn dự đoán.

Bước 3: Gọi hàm `scores()` để tính các chỉ số đánh giá.

Bước 4: Tổng hợp số lượng mẫu test, các metrics, Best Validation Loss và checkpoint được sử dụng.

Bước 5: Lưu kết quả tổng hợp vào `test_metrics.json` thông qua hàm `save_json()`.

Bước 6: Tính Confusion Matrix với thứ tự nhãn NEGATIVE và POSITIVE, sau đó lưu thành `confusion_matrix.csv`.

Bước 7: Tạo Classification Report, bao gồm Precision, Recall, F1-score và Support cho từng lớp, rồi lưu thành `classification_report.txt`.

Bước 8: Ghi từng mẫu dự đoán vào `test_predictions.csv`, bao gồm văn bản, nhãn thực tế, nhãn dự đoán và trạng thái dự đoán đúng hoặc sai.

Bước 9: Sử dụng Matplotlib để vẽ Confusion Matrix, hiển thị số lượng dự đoán tại từng ô và lưu hình ảnh dưới dạng `confusion_matrix.png`.

Cuối cùng, hàm trả về dictionary kết quả đánh giá.

## 4. Các file kết quả được tạo

Sau khi gọi `evaluate_and_save()`, chương trình tạo các file:

| File | Nội dung |
|---|---|
| `test_metrics.json` | Tổng hợp các chỉ số đánh giá |
| `confusion_matrix.csv` | Ma trận nhầm lẫn dạng CSV |
| `classification_report.txt` | Báo cáo chi tiết theo từng lớp |
| `test_predictions.csv` | Kết quả dự đoán trên từng review |
| `confusion_matrix.png` | Hình ảnh trực quan hóa ma trận nhầm lẫn |

## 5. Ý nghĩa các chỉ số đánh giá

**Accuracy:** Tỷ lệ tổng số dự đoán đúng trên tổng số mẫu đánh giá.

**Precision:** Trong những mẫu được dự đoán là POSITIVE, có bao nhiêu mẫu thực sự POSITIVE.

**Recall:** Trong những mẫu thực sự POSITIVE, mô hình nhận diện được bao nhiêu mẫu.

**F1-score:** Trung bình điều hòa giữa Precision và Recall, giúp đánh giá cân bằng hơn giữa hai chỉ số.

**Confusion Matrix:** Ma trận thể hiện số lượng dự đoán đúng và nhầm lẫn giữa hai lớp NEGATIVE và POSITIVE.

## 6. Liên kết với các module khác

- `data.py`: Cung cấp tập dữ liệu Test và nhãn thực tế.
- `model.py`: Xây dựng mô hình và tạo logits.
- `train.py`: Cung cấp thông tin quá trình huấn luyện, Validation Loss và checkpoint tốt nhất.
- `utils.py`: Cung cấp hàm `save_json()` để lưu kết quả đánh giá.

File `evaluate.py` không trực tiếp huấn luyện hay cập nhật trọng số mô hình mà tập trung vào tính toán và lưu trữ các kết quả đánh giá.

## 7. Kiểm chứng

Chương trình có một số bước kiểm tra nhằm bảo đảm kết quả đánh giá hợp lệ:

- Kiểm tra kích thước nhãn thực tế và nhãn dự đoán trong hàm `scores()`.
- Sử dụng `argmax` để chuyển logits thành nhãn dự đoán.
- Cố định thứ tự nhãn NEGATIVE và POSITIVE trong Confusion Matrix.
- Lưu kết quả dưới nhiều định dạng để thuận tiện đối chiếu.
- Lưu trạng thái đúng/sai của từng mẫu nhằm hỗ trợ phân tích các trường hợp mô hình dự đoán chưa chính xác.

## 8. Kết luận

File `evaluate.py` là thành phần chịu trách nhiệm đánh giá hiệu suất của mô hình phân tích cảm xúc sau quá trình huấn luyện.

Thông qua Accuracy, Precision, Recall, F1-score, Confusion Matrix và các kết quả dự đoán chi tiết, nhóm có thể đánh giá khả năng phân loại của mô hình trên dữ liệu Test và xác định những trường hợp mô hình còn dự đoán sai.
