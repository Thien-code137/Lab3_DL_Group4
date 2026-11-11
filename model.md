# MODEL — Lab 3

- Exercise 1: suy luận với `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, không fine-tune.
- Exercise 2: fine-tune `distilbert/distilbert-base-uncased`, hai lớp 0 NEGATIVE và 1 POSITIVE.
- Pipeline: `src/data.py`, `src/train.py`, `src/predict.py`, `configs/default.json`.
- Cấu hình dữ liệu: train 1.000 cân bằng, validation 200 cân bằng, test 200. Train và validation được tách trước khi lấy mẫu.
- Max length 256, 2 epoch, batch 8, LR 2e-5, weight decay 0.01, seed 42.
- **Chọn checkpoint:** Pipeline `src/train.py` dùng `eval_loss` nhỏ nhất; notebook Exercise 2 ở thư mục gốc dùng validation accuracy.
- **Kết quả notebook Exercise 2:** test accuracy 89%, checkpoint `checkpoint-125`, validation accuracy epoch 1 = 90,5%.
- **Kết quả pipeline:** Chỉ số được lưu trong thư mục `outputs/new_run/` sau khi chạy huấn luyện.

## Quy trình huấn luyện

1. `src/data.py`: chia train và validation theo nhãn, sau đó lấy mẫu cân bằng với seed 42.
2. `src/model.py`: tải tokenizer và DistilBERT pretrained. Đầu phân loại hai lớp được huấn luyện cho IMDb.
3. `src/train.py`: tokenize, tạo `DataCollatorWithPadding`, cấu hình `TrainingArguments` và `Trainer`.
4. Cuối mỗi epoch, Trainer tính validation loss. Checkpoint tốt nhất theo `eval_loss` được khôi phục bằng `load_best_model_at_end=True`.
5. `src/evaluate.py`: đánh giá test một lần sau selection; `src/visualize.py` lưu learning curve.

## Lưu ý về sự khác nhau giữa hai lần chạy

Notebook `Lab3_Exercise2_FineTunning.ipynb` ở thư mục gốc chọn checkpoint theo validation accuracy và có metrics đã lưu. Pipeline `src/train.py` dùng validation loss nên checkpoint cuối có thể khác. Hai kết quả cần được báo cáo theo đúng lần thực nghiệm và không ghép lại với nhau.
