# Notebook — Practice 3

| Phần | Nội dung |
|---|---|
| A | Chuẩn bị môi trường, tải IMDb, thống kê, EDA và PCA trên TF-IDF |
| B / Exercise 1 | Pretrained sentiment, tokenize câu, dự đoán và hiển thị score |
| C / Exercise 2 | Tải model, chia data, tokenize, TrainingArguments, Trainer, train, test, checkpoint, prediction |
| D | Đối chiếu các kết quả cũ, kiểm tra file đầu ra và kết luận |

- Cài `python -m pip install -r requirements.txt` trong terminal ở thư mục gốc.
- Mở notebook, chọn đúng Python Kernel, chạy từ cell đầu tiên đến cuối.
- Ở Codespaces thông thường notebook tải dữ liệu/model qua mạng và chạy huấn luyện. Chế độ `LAB3_OFFLINE_VERIFY=1` chỉ kiểm tra file đã lưu, không train.
- Output được lưu trong notebook ở lần bàn giao này là **kiểm chứng ngoại tuyến**: 20/20 code cells chạy thành công, không phải bằng chứng đã fine-tune lại.
- Notebook cũ nằm ở `legacy/notebooks/` nhằm giữ lại bài làm/bằng chứng cũ.

Đọc thêm [docs/README.md](../docs/README.md) để hiểu từng bước.
