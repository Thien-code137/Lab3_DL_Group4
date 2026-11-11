# Lab 3 – Phân tích cảm xúc với Hugging Face

**Học phần:** Deep Learning  
**Nhóm:** 4

## Nội dung

Lab gồm hai bài tập:

- **Exercise 1:** Sử dụng mô hình phân tích cảm xúc đã được huấn luyện trên Hugging Face Hub để tokenize và dự đoán nhãn cho câu tiếng Anh.
- **Exercise 2:** Fine-tune DistilBERT trên IMDb để phân loại đánh giá phim thành hai lớp `NEGATIVE` và `POSITIVE`.

## Cấu trúc thư mục

```text
Lab3_DL_Group4-main/
├── configs/default.json
├── data/
├── docs/
│   ├── ASSIGNMENT_LAB3.md
│   ├── BEST_VAL_LOSS.md
│   ├── EDA_PCA.md
│   └── PROJECT_STRUCTURE.md
├── notebooks/
│   ├── 01_eda_pca.ipynb
│   ├── 02_exercise1_sentiment.ipynb
│   └── 03_exercise2_finetuning.ipynb
├── outputs/exercise2/
├── reports/
│   └── Lab3_Report.md
├── src/
│   ├── data.py
│   ├── predict.py
│   ├── sentiment.py
│   ├── model.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── utils.py
│   └── train.py
├── tests/
├── main.py
├── model.md
├── requirements.txt
└── rules.md
```

Hai notebook `Lab3_Exercise1_Sentiment_Analysis.ipynb` và `Lab3_Exercise2_FineTunning.ipynb` ở thư mục gốc được giữ để tham khảo kết quả đã thực hiện.

## Cài đặt

```bash
python -m venv .venv
```

Trên PowerShell, kích hoạt môi trường ảo và cài thư viện:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Khi mở notebook trong VS Code, chọn kernel trỏ đến `.venv\Scripts\python.exe`. Chạy `python -m pip` từ terminal; chỉ dùng `%pip` bên trong notebook.

Kiểm tra mã nguồn trước khi train:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Thứ tự thực hiện

1. Chạy `notebooks/01_eda_pca.ipynb` để xem phân phối nhãn, độ dài văn bản và PCA trên TF-IDF.
2. Chạy `notebooks/02_exercise1_sentiment.ipynb` cho Exercise 1.
3. Chạy `notebooks/03_exercise2_finetuning.ipynb` cho Exercise 2.
4. Lưu các biểu đồ, chỉ số và checkpoint sau khi huấn luyện.

Có thể huấn luyện bằng lệnh:

```bash
python main.py --config configs/default.json --output outputs/new_run
```

Lệnh trên sẽ tải IMDb và pretrained model nếu chưa có cache; thời gian chạy tùy cấu hình máy.

## Kết quả

Trong lần chạy được lưu ở notebook Exercise 2 ban đầu, mô hình đạt **89% accuracy trên 200 mẫu test**. Các số liệu và biểu đồ của lần chạy đó nằm tại `outputs/exercise2/`.

Pipeline trong `src/` sử dụng cấu hình lựa chọn checkpoint theo validation loss. Khi chạy pipeline này, kết quả mới được xuất ra thư mục chỉ định bởi `--output` và cần được đánh giá riêng.

Sau khi huấn luyện bằng `main.py`, thư mục `outputs/new_run/` được tạo với:

| File | Nội dung |
|---|---|
| `data_summary.json` | Kích thước và phân phối nhãn của ba tập |
| `history.csv` | Loss train, loss/accuracy/F1 validation qua các epoch |
| `loss_curves.png` | Biểu đồ train và validation loss |
| `test_metrics.json` | Accuracy, precision, recall, F1 và best validation loss |
| `confusion_matrix.csv`, `.png` | Ma trận nhầm lẫn |
| `classification_report.txt` | Precision, recall, F1 của từng lớp |
| `test_predictions.csv` | Dự đoán từng mẫu và nhãn thật |
| `best_model/` | Mô hình và tokenizer đã lưu |
| `run_info.json` | Thời gian, phiên bản, seed và cấu hình |

Các file này chỉ xuất hiện khi chạy thành công `python main.py`; kết quả ở `outputs/exercise2/` là của notebook đã lưu trước đó.

## Báo cáo

- [Báo cáo Lab 3](reports/Lab3_Report.md)
- [Yêu cầu bài tập](docs/ASSIGNMENT_LAB3.md)
- [EDA và PCA](docs/EDA_PCA.md)
- [Mô hình](model.md)
- [Phân tích kỹ thuật theo từng mục](reports/sections/01_data.md)

## Tài liệu tham khảo

- [Hugging Face Fine-tuning](https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model)
- [Sentiment Analysis with Python](https://huggingface.co/blog/sentiment-analysis-python)
- [Kaggle Sentiment Analysis using Hugging Face](https://www.kaggle.com/code/gauravduttakiit/sentiment-analysis-using-hugging-face)
- [Kaggle Fine-tuning BERT](https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification)
