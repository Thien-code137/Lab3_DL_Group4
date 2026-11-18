
# Audit Summary - Lab 3

## 1. Mục đích

Tài liệu ghi lại quá trình kiểm tra dữ liệu, thực hiện
thực nghiệm và đánh giá kết quả Lab 3.

## 2. Các nội dung đã thực hiện

- Exercise 1: Sử dụng DistilBERT đã fine-tune trên SST-2
  để phân tích cảm xúc.
- Exercise 2: Fine-tune DistilBERT pretrained trên dữ liệu IMDb.
- EDA: Khảo sát phân bố nhãn, độ dài review, dữ liệu thiếu
  và dữ liệu trùng lặp.
- PCA: Chuyển văn bản thành TF-IDF và giảm chiều dữ liệu
  xuống 2 thành phần chính để trực quan hóa.

## 3. Kết quả thực nghiệm hiện có

Kết quả Exercise 2 được tổng hợp từ notebook và báo cáo
thực nghiệm đã lưu:

- Training samples: 1.000.
- Validation samples: 200.
- Test samples: 200.
- Test Accuracy: 89%.

Validation loss:
- Epoch 1: 0.283972.
- Epoch 2: 0.299779.

Validation loss thấp nhất được ghi nhận tại Epoch 1.

Checkpoint được lựa chọn theo Validation Accuracy,
không phải theo Validation Loss.

## 4. Kiểm tra kết quả

Các file được sử dụng để đối chiếu:

- outputs/exercise2/test_metrics.json
- outputs/exercise2/classification_report.txt
- outputs/exercise2/confusion_matrix.csv
- outputs/exercise2/test_predictions.csv
- reports/Lab3_Report.md

## 5. Bổ sung EDA và PCA

Notebook Lab3_EDA_PCA_fixed.ipynb được bổ sung
để thực hiện khám phá dữ liệu trên tập train.

Các kết quả được lưu trong outputs/eda_pca/.

Không sử dụng tập test trong bước EDA/PCA.

## 6. Ghi chú

Các kết quả huấn luyện được giữ từ lần thực nghiệm trước.
Không thực hiện fine-tuning lại khi bổ sung tài liệu
và phần EDA/PCA.

Các thông tin về môi trường và kết quả cần được đối chiếu
với log gốc trước khi sử dụng làm bằng chứng chính thức.
