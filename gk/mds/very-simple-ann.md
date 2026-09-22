# ANN trong một ví dụ rất đơn giản

## Tiếp nối phần trước:

- Ta có 1 model tuyến tính đơn giản: `z = w_1*x_1 + w_2*x_2 + b`. Sau khi tính `z` thì được 1 con số, vd `0.4`, `-0.5`, ... Nhưng mục tiêu cần là "Đậu" hoặc "Rớt". Lúc này phải viết thêm logic: `if z >= 0 then "Đậu" else "Rớt"`.

- Bước đầu của ANN (1 neuron / perceptron) là: lấy `z`, rồi đưa qua 1 **activation function** để ra tín hiệu gắn với "Đậu"/"Rớt". Step function chính là if-else viết thành hàm; Sigmoid/Softmax thì ra xác suất, lúc lấy lớp cứng vẫn cần ngưỡng hoặc `argmax`.

- 2 lớp Đậu/Rớt chỉ là ví dụ. Có thể là `["học sinh yếu", "học sinh trung bình", "học sinh giỏi"]` (**3 lớp**). Phần này mới chỉ nói 1 neuron; tiếp: [`very-simple-n-neurons.md`](./very-simple-n-neurons.md).

## Ý tưởng:

- Từ `z` ta cần map sang nhãn (hoặc xác suất các lớp). Activation function làm việc đó ở **lớp cuối**; ở **lớp ẩn** thì activation chủ yếu tạo phi tuyến, không nhất thiết ra class.

- Vậy activation function là gì?

- Activation function là hàm biến đổi đầu ra của neuron (`z = w·x + b`) thành tín hiệu mới. Ở lớp cuối có thể giúp suy ra class / xác suất; ở lớp ẩn thì giúp mạng học được quan hệ phi tuyến.

- Activation function thường được sử dụng:
  - **Step function**: cắt cứng theo ngưỡng. Ví dụ `z >= 0` → Đậu (1), ngược lại → Rớt (0). Không có xác suất, chỉ có 0 hoặc 1.
  - **Sigmoid**: ép `z` về khoảng `(0, 1)`. Có thể đọc như xác suất "Đậu". Thường dùng khi chỉ có **2 lớp**.
  - **Tanh**: ép `z` về khoảng `(-1, 1)`. Ít dùng làm output Đậu/Rớt; hay dùng ở **lớp ẩn** để tín hiệu có dương/âm.
  - **ReLU**: `max(0, z)`. Không phải hàm phân loại Đậu/Rớt; dùng ở **lớp ẩn** vì đơn giản, học nhanh.
  - **Softmax**: nhận nhiều điểm số (mỗi lớp một số), trả về bộ xác suất cộng lại = 1. Dùng khi có **nhiều lớp** (yếu / TB / giỏi, ...).
