# Yêu cầu bài thực hành Practice 3

Lab 3 gồm Exercise 1: Sentiment Analysis và Exercise 2: Fine-tuning a Pretrained Model for Binary Text Classification.

| # | Nội dung | File thực hiện |
|---|---|
| EX1.1 | Install Hugging Face transformers | `requirements.txt`, `02_exercise1_sentiment.ipynb` |
| EX1.2 | Use a pretrained sentiment model from Hugging Face Hub | `src/sentiment.py` |
| EX1.3 | Tokenize a sample sentence | `src/sentiment.py`, notebook EX1 |
| EX1.4 | Perform sentiment analysis on the sentence | `src/sentiment.py`, notebook EX1 |
| EX2.1 | Install transformers, datasets, evaluate | `requirements.txt`, notebook EX2 |
| EX2.2 | Load a simple binary dataset | `src/data.py` (IMDb) |
| EX2.3 | Load a pretrained model and tokenizer | `src/train.py` |
| EX2.4 | Preprocess dataset | `src/data.py`, `src/train.py` |
| EX2.5 | Define training arguments | `configs/default.json`, `src/train.py` |
| EX2.6 | Create Trainer and fine-tune | `src/train.py` |
| EX2.7 | Evaluate fine-tuned model | `src/train.py` |

Kết quả thực nghiệm trong `outputs/exercise2/` được lưu từ notebook `Lab3_Exercise2_FineTunning.ipynb`. Với pipeline `src/train.py`, kết quả được lưu theo từng lần chạy ở thư mục output riêng.

## Nguồn tham khảo từ đề

1. [Hugging Face documentation — Fine-tuning](https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model).
2. [Hugging Face blog — Sentiment Analysis Python](https://huggingface.co/blog/sentiment-analysis-python).
3. [Kaggle — Sentiment Analysis using Hugging Face](https://www.kaggle.com/code/gauravduttakiit/sentiment-analysis-using-hugging-face).
4. [Kaggle — Fine-tuning BERT for Text Classification](https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification).

