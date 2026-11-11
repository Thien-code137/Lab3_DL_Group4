# Dữ liệu IMDb

Dataset được tải qua `datasets.load_dataset("stanfordnlp/imdb")` từ Hugging Face Hub. Repo không lưu bản sao toàn bộ dataset.

- Train gốc: 25.000 mẫu.
- Test gốc: 25.000 mẫu.
- Nhãn: `0 = NEGATIVE`, `1 = POSITIVE`.

Hàm chia và lấy mẫu nằm trong `src/data.py`. Cache tải dữ liệu do Hugging Face quản lý trên máy chạy chương trình.