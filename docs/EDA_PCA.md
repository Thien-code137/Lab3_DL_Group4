# EDA và PCA trước fine-tuning Lab 3

Notebook `notebooks/01_eda_pca.ipynb` tải **train split IMDb** và thực hiện:

1. Đếm số lượng mỗi nhãn, độ dài văn bản, thiếu/chuỗi rỗng, văn bản trùng.
2. Vẽ histogram độ dài và phân phối lớp.
3. Chọn tối đa 600 văn bản từ train, TF-IDF tối đa 1.500 features rồi PCA 2D để xem mức độ chồng lấn giữa hai nhãn.
4. Không lấy thông tin từ validation/test để quyết định mô hình.

## Kết quả kiểm tra tập train

Kết quả đã ghi nhận khi chạy notebook trên tập train IMDb:

| Nội dung | Giá trị |
|---|---:|
| Tổng số văn bản | 25.000 |
| Negative | 12.500 |
| Positive | 12.500 |
| Giá trị thiếu | 0 |
| Văn bản rỗng | 0 |
| Dòng trùng nội dung | 96 |

Tỉ lệ trùng là `96 / 25.000 = 0,384%`. Chưa tự động xóa các dòng này vì cần kiểm tra thêm trường hợp cùng nội dung nhưng khác nhãn. Các biểu đồ độ dài và PCA được sinh khi chạy các cell tương ứng; không có số liệu PCA cố định trước khi chạy.

## Diễn giải biểu đồ

- **Label distribution:** Hai lớp cân bằng ở tập train gốc, thuận lợi cho đánh giá phân loại nhị phân.
- **Review length histogram:** Dùng để cân nhắc `max_length=256` và nguy cơ cắt văn bản dài.
- **PCA scatter:** Hai điểm gần nhau chỉ có biểu diễn TF-IDF tương đối giống trong phép chiếu hai chiều, không đồng nghĩa có cùng nhãn.
- **Explained variance:** Tổng tỉ lệ phương sai của PC1 và PC2 cho biết mức độ thông tin được giữ lại trong phép chiếu; nếu thấp, không nên diễn giải hình 2D như mô tả đầy đủ dữ liệu.

PCA trên TF-IDF là phép chiếu phục vụ trực quan hóa, không phải embedding DistilBERT hay bước biến đổi trong pipeline fine-tuning. Notebook tải dataset IMDb từ Hugging Face khi thực thi lần đầu.
