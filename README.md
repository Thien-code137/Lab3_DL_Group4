# Lab 3 — Get started with Hugging Face

**Học phần:** Deep Learning | **Nhóm:** 4

Project đầy đủ gồm hai Exercise bắt buộc: (1) Sentiment Analysis bằng mô hình pretrained và (2) fine-tuning DistilBERT cho phân loại cảm xúc nhị phân trên IMDb. Phần EDA/PCA là nội dung bổ sung, không phải đầu vào của DistilBERT.

## Cấu trúc

```text
Lab3_DL_Group4-main/
├── configs/default.json       # Cấu hình thực nghiệm
├── data/README.md             # Nguồn dữ liệu và cách tải
├── docs/                      # Đề bài, giải thích kỹ thuật, đối chiếu kết quả
├── notebooks/
│   ├── practice_3.ipynb       # Notebook chính: toàn bộ bài theo thứ tự đề
│   ├── 01_exercise1_sentiment.ipynb
│   ├── 02_exercise2_finetuning.ipynb
│   ├── README.md
│   └── archive/              # Notebook gốc có output, chỉ tham khảo
├── outputs/exercise2/         # Kết quả lịch sử (89%)
├── reports/                   # Báo cáo Markdown, DOCX, HTML và các mục
├── src/                       # Module data/model/train/evaluate/predict/...
├── tests/                     # Unit test
├── Lab3_EDA_PCA_fixed.ipynb    # Notebook EDA/PCA gốc
├── main.py                    # Entry point huấn luyện bằng terminal
├── requirements.txt
├── README.md
└── ...                       # Biểu đồ, JSON, CSV, tài liệu gốc của nhóm
```

## 1. Cài đặt trên Windows / VS Code

Mở PowerShell tại thư mục gốc. Khuyến nghị Python 3.12, GPU nếu có; cần Internet để tải model và dataset lần đầu.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Mở `notebooks/practice_3.ipynb`, chọn kernel `.venv`, chạy từ trên xuống (Run All). Nếu cập nhật thư viện trong notebook, restart kernel rồi chạy lại. Có thể upload notebook độc lập lên Google Colab và chọn GPU.

## 2. Workflow theo đề

| Exercise | Các bước                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1        | Cài Transformers → tải pretrained sentiment model → tokenize câu → dự đoán cảm xúc                                                    |
| 2        | Cài thư viện → IMDb (EDA/PCA và chia tập) → model/tokenizer → preprocess → TrainingArguments → Trainer và fine-tuning → đánh giá test |

Notebook chính đã bao gồm cả hai Exercise, không cần chạy lại hai notebook riêng nếu đã hoàn thành nó. Mặc định `QUICK_TEST=False`, `RUN_TRAIN=True`; cấu hình 1.000/200/200 mẫu, 2 epoch, batch size 8, max length 256, learning rate 2e-5, seed 42. Xem [hướng dẫn notebook](notebooks/README.md).

## 3. Đường chạy module (tùy chọn)

Từ thư mục gốc, dùng Python của môi trường đã cài thư viện:

```powershell
.\.venv\Scripts\python.exe -m src.sentiment --text "I love this movie!"
.\.venv\Scripts\python.exe main.py --config configs/default.json --output outputs/new_run
.\.venv\Scripts\python.exe -m src.predict --checkpoint outputs/new_run/best_model --text "The film was excellent."
```

Lệnh cuối chỉ chạy sau khi có `best_model/`. Module gốc và notebook mới có khác biệt trong cách lấy mẫu và làm sạch dữ liệu; không gộp kết quả của hai đường chạy.

## 4. Kết quả

- `outputs/exercise2/`: 89% accuracy từ thực nghiệm lịch sử A, được giữ nguyên cùng confusion matrix, prediction, classification report.
- `notebooks/archive/practice_3_original.ipynb`: 86,5% accuracy từ thực nghiệm lịch sử B; giữ nguyên output.
- Notebook: Sau khi chạy, kết quả được lưu ở `outputs/practice3_<timestamp>/` gồm cấu hình, EDA/PCA, data summary, history, loss curves, test metrics, classification report, confusion matrix, dự đoán và checkpoint.
- Pipeline module xuất `outputs/new_run/` với dữ liệu chia tập, history, metrics, confusion matrix, predictions, run info và best model.

Xem [đối chiếu các lần chạy](docs/EXPERIMENTS.md). Không thay thế kết quả mới bằng số liệu lịch sử. Không có full dataset hay trọng số fine-tuned sẵn trong ZIP. Kiểm tra cấu trúc và cú pháp notebook, 6 unit test đạt; chưa chạy lại fine-tuning với Hugging Face.

## 5. Tài liệu

- [Cấu trúc dự án](docs/PROJECT_STRUCTURE.md)
- [EDA/PCA lịch sử](docs/EDA_PCA.md)
- [Báo cáo Lab 3](reports/Lab3_Report.md)
- [Báo cáo DOCX gốc](reports/Lab3_Presentation.docx)
- [Hugging Face — Fine-tuning](https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model)
- [Hugging Face — Sentiment Analysis](https://huggingface.co/blog/sentiment-analysis-python)
