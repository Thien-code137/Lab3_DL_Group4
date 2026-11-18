# DATA.PY – CHUẨN BỊ VÀ CHIA DỮ LIỆU IMDb

## 1. Mục đích

File `src/data.py` chịu trách nhiệm tải dataset IMDb và chuẩn bị dữ liệu cho quá trình huấn luyện, kiểm định và đánh giá mô hình phân tích cảm xúc.

Dữ liệu được chia thành ba tập gồm Training, Validation và Testing. Việc chia dữ liệu được thực hiện với random seed cố định để bảo đảm khả năng tái lập kết quả.

## 2. Các thư viện sử dụng

- `collections.Counter`: Kiểm tra số lượng mẫu thuộc từng nhãn.
- `datasets.load_dataset`: Tải dataset từ thư viện Hugging Face Datasets.
- `datasets.concatenate_datasets`: Kết hợp các tập dữ liệu sau khi lấy mẫu từ từng lớp.

## 3. Các hàm chính

### 3.1. Hàm balanced_subset(dataset, n, seed)

Hàm này lấy một tập con có số lượng mẫu cân bằng giữa hai lớp cảm xúc.

**Input:**

- `dataset`: Dataset đầu vào.
- `n`: Tổng số mẫu cần lấy.
- `seed`: Giá trị random seed để bảo đảm tính tái lập.

**Quy trình thực hiện:**

1. Kiểm tra số lượng mẫu cần lấy có phải số chẵn hay không. Nếu là số lẻ, chương trình thông báo lỗi.
2. Lọc dữ liệu theo hai nhãn 0 và 1.
3. Xáo trộn dữ liệu của từng lớp bằng seed cố định.
4. Lấy đúng n/2 mẫu từ mỗi lớp.
5. Kết hợp hai tập con bằng `concatenate_datasets`.
6. Xáo trộn tập dữ liệu sau khi kết hợp.
7. Sử dụng `Counter` để kiểm tra số lượng mẫu của từng nhãn.

**Output:**

Một Dataset mới có tổng cộng n mẫu, trong đó số lượng mẫu thuộc hai lớp bằng nhau.

### 3.2. Hàm prepare_splits(config)

Hàm này chịu trách nhiệm chuẩn bị ba tập dữ liệu phục vụ cho quá trình thực nghiệm.

**Input:**

Đối tượng `config` chứa các thông tin cấu hình:

- `dataset`: Tên dataset cần tải.
- `seed`: Random seed.
- `train_size`: Số lượng mẫu Training.
- `validation_size`: Số lượng mẫu Validation.
- `test_size`: Số lượng mẫu Testing.

**Quy trình thực hiện:**

Bước 1: Tải dataset bằng `load_dataset(config["dataset"])`.

Bước 2: Chia tập train gốc thành hai phần Training và Validation bằng `train_test_split`, với tỷ lệ validation là 10%, sử dụng random seed và stratification theo nhãn.

Bước 3: Sử dụng `balanced_subset` để lấy đúng số lượng mẫu Training và bảo đảm hai lớp có số mẫu bằng nhau.

Bước 4: Tiếp tục lấy mẫu Validation từ phần dữ liệu đã tách riêng, cũng bảo đảm số lượng mẫu hai lớp cân bằng.

Bước 5: Lấy dữ liệu Testing từ split `test` gốc của IMDb, xáo trộn và chọn đúng số lượng mẫu được cấu hình.

Bước 6: Kiểm tra dữ liệu Training và Validation không có văn bản trùng nhau.

Bước 7: Kiểm tra số lượng mẫu thực tế của cả ba tập có đúng với cấu hình hay không.

**Output:**

Hàm trả về ba tập dữ liệu:

- `train`: Dữ liệu huấn luyện.
- `validation`: Dữ liệu kiểm định.
- `test`: Dữ liệu đánh giá cuối cùng.

## 4. Quy tắc chia dữ liệu

Trong quá trình chuẩn bị dữ liệu, chương trình áp dụng các quy tắc:

- Sử dụng dataset IMDb từ Hugging Face.
- Tập train gốc được chia thành Training và Validation trước khi lấy mẫu.
- Training và Validation được cân bằng giữa hai nhãn.
- Testing được lấy riêng từ split test gốc.
- Random seed được cố định theo cấu hình.
- Kiểm tra số lượng mẫu và tránh trùng văn bản giữa Training và Validation.

Cách chia này giúp tách biệt dữ liệu sử dụng để học tham số, kiểm định mô hình và đánh giá cuối cùng.

## 5. Liên kết với các module khác

`data.py` là bước chuẩn bị dữ liệu đầu vào cho pipeline Lab3.

- `train.py`: Sử dụng dữ liệu Training và Validation trong quá trình fine-tuning.
- `evaluate.py`: Sử dụng dữ liệu Testing để đánh giá mô hình.
- `model.py`: Nhận dữ liệu đã được mã hóa thành tensor hoặc input phù hợp để xử lý.

Việc tokenization và tạo attention mask được thực hiện ở các bước xử lý dữ liệu tiếp theo, không phải nhiệm vụ trực tiếp của `data.py`.

## 6. Kiểm chứng

Trong code, chương trình sử dụng các câu lệnh `assert` để kiểm tra:

- Số lượng mẫu của Training đúng với cấu hình.
- Số lượng mẫu của Validation đúng với cấu hình.
- Số lượng mẫu của Testing đúng với cấu hình.
- Văn bản Training và Validation không bị trùng lặp.
- Số lượng mẫu của hai lớp trong tập con cân bằng được kiểm tra bằng `Counter`.

Nếu một điều kiện không thỏa mãn, chương trình sẽ phát sinh lỗi để tránh tiếp tục sử dụng dữ liệu không đúng yêu cầu.

## 7. Kết luận

File `data.py` đảm nhiệm việc tải và phân chia dữ liệu IMDb thành ba tập Training, Validation và Testing. Trong đó, Training và Validation được lấy mẫu cân bằng giữa hai lớp cảm xúc, đồng thời sử dụng random seed cố định để hỗ trợ tái lập thực nghiệm.

Đây là bước đầu tiên trong pipeline phân tích cảm xúc, tạo dữ liệu đầu vào cho các bước tiền xử lý, huấn luyện và đánh giá mô hình tiếp theo.
