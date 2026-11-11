# Notebook Lab 3

| Notebook | Nội dung | Kết quả cần quan sát |
|---|---|---|
| `01_eda_pca.ipynb` | EDA tập train IMDb, TF-IDF và PCA | Phân phối nhãn, độ dài review, PCA 2D và explained variance |
| `02_exercise1_sentiment.ipynb` | Tokenize câu và suy luận bằng model pretrained | Tokens, nhãn, điểm softmax |
| `03_exercise2_finetuning.ipynb` | Cấu hình Trainer, fine-tune và đánh giá | Train log, best checkpoint và metrics test |

Nên chạy lần lượt theo thứ tự từ trên xuống trong cùng một môi trường Python. Chọn kernel đã cài các thư viện trong `requirements.txt`.

`03_exercise2_finetuning.ipynb` có cell `trainer.train()` nên sẽ thực hiện huấn luyện khi chạy; không nên dùng nút Run All nếu chưa chuẩn bị thời gian và tài nguyên.
