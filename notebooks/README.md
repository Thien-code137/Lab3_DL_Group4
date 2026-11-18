# Practice 3 — Get started with Hugging Face | Nhóm 4

Thư mục notebooks được tổ chức lại từ `Lab3_DL_Group4-main(3).zip` và notebook tổng hợp đã dựng. Mỗi notebook tự chứa code, không phụ thuộc `src/`, `configs/` hoặc checkpoint từ project gốc.

## 1. Cấu trúc thư mục

```text
notebooks/
├── practice_3.ipynb                  # Notebook chính: chạy toàn bộ bài
├── 01_exercise1_sentiment.ipynb       # Exercise 1 riêng
├── 02_exercise2_finetuning.ipynb      # Exercise 2 riêng, gồm EDA/PCA
└── README.md                         # Hướng dẫn chạy và đối chiếu đề
```

**Khi nộp hoặc trình bày:** ưu tiên `practice_3.ipynb`. Hai notebook còn lại dùng để chạy và kiểm tra từng exercise độc lập; không cần chạy cả ba nếu đã chạy notebook chính.

## 2. Flow theo slide đề bài

| Yêu cầu | Nội dung                                             | Mục trong notebook chính         |
| ------- | ---------------------------------------------------- | -------------------------------- |
| EX1.1   | Install transformers                                 | 1.1                              |
| EX1.2   | Use pretrained sentiment model from Hugging Face Hub | 1.2                              |
| EX1.3   | Tokenize a sample sentence                           | 1.3                              |
| EX1.4   | Perform sentiment analysis                           | 1.4                              |
| EX2.1   | Install transformers, datasets, evaluate             | 2.1 (dùng thư viện đã cài ở 1.1) |
| EX2.2   | Load binary text classification dataset              | 2.2 (IMDb, EDA/PCA, chia tập)    |
| EX2.3   | Load pretrained model and tokenizer                  | 2.3                              |
| EX2.4   | Preprocess the dataset                               | 2.4                              |
| EX2.5   | Define TrainingArguments                             | 2.5                              |
| EX2.6   | Create Trainer and fine-tune                         | 2.6                              |
| EX2.7   | Evaluate the fine-tuned model                        | 2.7                              |

EDA/PCA, kiểm tra trùng lặp, biểu đồ loss, phân tích dự đoán sai và lưu model là nội dung bổ sung. PCA không phải đầu vào của DistilBERT.

## 3. Cách chạy

### Google Colab (khuyến nghị GPU)

1. Giải nén ZIP, upload `practice_3.ipynb` lên Colab.
2. Chọn Runtime > Change runtime type > GPU (nếu có).
3. Chạy cell cài thư viện, restart session nếu cần và chạy toàn bộ từ đầu.
4. Kiểm tra kết quả cuối notebook và tải thư mục `outputs/` về trước khi kết thúc phiên Colab nếu cần lưu lâu dài.

### VS Code / Jupyter trên Windows

1. Cài Python 3.10–3.12 và extension Python, Jupyter trên VS Code.
2. Mở thư mục vừa giải nén; có thể tạo môi trường riêng:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip ipykernel
```

3. Mở `practice_3.ipynb`, chọn kernel `.venv` và chạy cell cài thư viện bằng `%pip`.
4. Restart Kernel nếu cần, sau đó Run All. Cần Internet để tải model và IMDb lần đầu. CPU vẫn có thể chạy nhưng fine-tuning có thể chậm.

## 4. Cấu hình mặc định (dựa trên configs/default.json gốc)

| Tham số                      | Giá trị                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| Dataset                      | stanfordnlp/imdb                                           |
| EX1 model                    | distilbert/distilbert-base-uncased-finetuned-sst-2-english |
| EX2 model                    | distilbert/distilbert-base-uncased                         |
| Train / Validation / Test    | 1000 / 200 / 200                                           |
| Epochs / Batch size          | 2 / 8                                                      |
| Learning rate / Weight decay | 2e-5 / 0.01                                                |
| Max length / Seed            | 256 / 42                                                   |
| Best checkpoint              | eval_loss thấp nhất                                        |

Mặc định `QUICK_TEST = False`, `RUN_TRAIN = True`. Nếu cần kiểm tra nhanh flow, đặt `QUICK_TEST = True` (100/40/40, 1 epoch). Đặt `RUN_TRAIN = False` chỉ chuẩn bị dữ liệu và mô hình, không hoàn thành yêu cầu fine-tuning/evaluation. Khi nộp kết quả chính thức, dùng chế độ mặc định.

## 5. Kết quả sau khi chạy

Notebook tự tìm thư mục gốc project (nếu đang chạy trong repo); nếu chạy độc lập thì dùng working directory. Notebook sinh thư mục `outputs/`, mỗi lần chạy có thư mục riêng theo thời gian. Notebook chính lưu dự đoán Exercise 1, EDA/PCA, thống kê chia tập, lịch sử huấn luyện, loss, test metrics, confusion matrix, classification report, dự đoán test và `best_model/` (nếu train thành công). Hai notebook riêng chỉ tạo đầu ra liên quan đến exercise tương ứng.

Không có checkpoint hay output huấn luyện mới đóng gói sẵn. Kết quả 89% trong ZIP gốc là của lần chạy cũ, không phải kết quả của phiên bản này. Không dùng kết quả cũ để thay thế output mới. Đã kiểm tra cấu trúc notebook và cú pháp Python; chưa chạy lại fine-tuning thực tế trong môi trường tạo ZIP.

## 6. Tham khảo theo slide

- [Hugging Face — Fine-tuning](https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model)
- [Hugging Face — Sentiment Analysis Python](https://huggingface.co/blog/sentiment-analysis-python)
- [Kaggle — Sentiment Analysis using Hugging Face](https://www.kaggle.com/code/gauravduttakiit/sentiment-analysis-using-hugging-face)
- [Kaggle — Fine-tuning BERT](https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification)

## 7. Lưu trữ và đối chiếu

Notebook gốc có output được giữ tại `archive/practice_3_original.ipynb`. Xem [đối chiếu kết quả](../docs/EXPERIMENTS.md) và [README project](../README.md).
