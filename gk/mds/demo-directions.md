# Các hướng làm demo ANN

## Hướng được chọn: `make_moons`

`make_moons` là lựa chọn chính vì nó có hai feature và tạo ra hai lớp với ranh giới phi tuyến. Người xem có thể nhìn trực tiếp dữ liệu, decision boundary và sự khác nhau giữa model tuyến tính với MLP.

### Vì sao phù hợp?

- Dữ liệu hai chiều nên dễ vẽ.
- Logistic regression tạo baseline rõ ràng.
- MLP có cơ hội thể hiện tác dụng của hidden layer.
- Không cần kể một câu chuyện nghiệp vụ giả hoặc thêm thông tin người dùng không có.
- Dễ thay đổi noise, số neuron và activation để tạo câu hỏi tương tác.

### Kịch bản thuyết trình

1. Vẽ dữ liệu `make_moons` và hỏi người xem có thể tách hai lớp bằng một đường thẳng không.
2. Chạy logistic regression, cho xem decision boundary tuyến tính.
3. Chạy MLP, cho xem decision boundary thay đổi.
4. Mở loss curve để giải thích model cập nhật tham số qua quá trình train.
5. Đổi `activation` hoặc `hidden_layer_sizes` một lần để minh họa ảnh hưởng của kiến trúc.
6. Kết luận bằng giới hạn của demo: dataset nhỏ, cấu hình cụ thể và vẫn cần đánh giá trên test set.

## Hướng phụ 1: XOR

XOR phù hợp để mở đầu phần lý thuyết:

- Chỉ có bốn điểm nên có thể vẽ và giải thích nhanh.
- Perceptron đơn không tạo được đường thẳng phân tách hai lớp.
- MLP cho thấy vì sao hidden layer mở rộng khả năng biểu diễn.

XOR nên là ví dụ ngắn, không nhất thiết là toàn bộ notebook. Một demo chỉ có bốn điểm ít cho thấy quá trình huấn luyện trên dữ liệu có nhiễu.

## Hướng phụ 2: `make_circles`

`make_circles` tạo ranh giới dạng vòng. Hướng này tiếp tục làm nổi bật sự khác nhau giữa model tuyến tính và model có hidden layer.

Nên dùng `make_circles` như phần mở rộng sau `make_moons`. Nếu đưa cả hai dataset vào kịch bản chính, thời gian giải thích decision boundary và tham số sẽ dài hơn.

## Hướng phụ 3: dataset chẩn đoán khối u

Dataset chẩn đoán khối u của scikit-learn có thể dùng cho phần mở rộng hoặc phần so sánh với nhóm làm decision tree. Tuy nhiên không nên chọn nó làm demo ANN chính vì:

- Có nhiều feature nên không thể vẽ toàn bộ decision boundary trực tiếp.
- Câu chuyện dễ chuyển thành so sánh accuracy, trong khi mục tiêu chính là giải thích ANN hoạt động và được train ra sao.
- Nhóm khác đã dùng dataset này cho decision tree, nên `make_moons` giúp phần ANN có hình ảnh và câu hỏi riêng.

Nếu dùng dataset này sau đó, cần nói rõ preprocessing, cách chia dữ liệu, metric và kết quả thực tế sau khi chạy. Không ghi trước accuracy trong tài liệu.

## So sánh nhanh

| Hướng | Điểm mạnh | Hạn chế | Vai trò đề xuất |
| --- | --- | --- | --- |
| `make_moons` | Dễ vẽ và thấy ranh giới phi tuyến | Là dữ liệu tổng hợp | Demo chính |
| XOR | Giải thích giới hạn perceptron rất ngắn | Quá ít điểm cho một thí nghiệm đầy đủ | Mở đầu lý thuyết |
| `make_circles` | Decision boundary trực quan | Có thể lặp lại ý tưởng `make_moons` | Phần mở rộng |
| Dataset chẩn đoán khối u | Gần bài toán thực tế | Nhiều chiều, khó minh họa bằng hình | Phần so sánh sau này |

## Hướng chuyển thành web app sau notebook

Sau khi notebook chạy ổn định, có thể chuyển cùng logic sang Streamlit. App chỉ cần tập trung vào các tương tác phục vụ bài nói:

- Chọn dataset: `make_moons`, XOR hoặc `make_circles`.
- Chọn activation: `tanh` hoặc `relu`.
- Chọn số neuron trong hidden layer.
- Chọn learning rate và giới hạn lặp.
- Nhấn nút train.
- Hiển thị dữ liệu và decision boundary.
- Hiển thị loss curve khi model hỗ trợ thuộc tính này.
- Hiển thị metric trên test set cùng cấu hình đã dùng.

### Trạng thái cần dự kiến cho app

Nếu triển khai web app, cần có các trạng thái rõ ràng:

- Chưa train: hướng dẫn người dùng chọn cấu hình và nhấn train.
- Đang train: hiển thị trạng thái xử lý.
- Đã train: hiển thị biểu đồ và metric.
- Lỗi cấu hình: báo tham số không hợp lệ hoặc model chưa hội tụ.

Không thêm nút hoặc menu không có hành vi thật. Không đưa claim về tốc độ, độ chính xác hoặc khả năng tổng quát nếu chưa có phép đo và nguồn tương ứng.

## Thứ tự ưu tiên triển khai

1. Hoàn thành notebook `make_moons`.
2. Thêm XOR vào phần giải thích.
3. Thử `make_circles` nếu còn thời gian.
4. Chỉ sau đó mới chuyển sang Streamlit.
5. Dùng dataset chẩn đoán khối u như một phần mở rộng, không thay thế demo trực quan chính.

