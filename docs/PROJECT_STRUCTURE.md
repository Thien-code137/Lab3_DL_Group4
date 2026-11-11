# Cấu trúc dự án Lab 3

| Thư mục | Mục đích |
|---|---|
| `configs/` | Siêu tham số, seed, dataset và tiêu chí chọn |
| `src/data.py` | Chia train/validation không chồng lấn và lấy mẫu cân bằng |
| `src/train.py` | Fine-tuning và đánh giá đúng thứ tự |
| `src/model.py` | Tải tokenizer và kiến trúc phân loại hai lớp |
| `src/evaluate.py` | Metrics, confusion matrix và dự đoán test |
| `src/visualize.py` | Lịch sử huấn luyện và biểu đồ loss |
| `src/utils.py` | Kiểm tra cấu hình và ghi JSON |
| `src/sentiment.py` | Exercise 1: suy luận pretrained, không train |
| `src/predict.py` | Suy luận từ checkpoint đã xuất |
| `notebooks/01_eda_pca.ipynb` | EDA và PCA trước huấn luyện |
| `reports/` | Báo cáo, HTML và DOCX |
| `outputs/exercise2/` | Các kết quả thực nghiệm đã lưu |
| `tests/` | Test cấu hình không cần tải dataset |

Chạy `python -m unittest discover -s tests` để kiểm tra cấu hình. Chạy `python main.py` để bắt đầu huấn luyện.
