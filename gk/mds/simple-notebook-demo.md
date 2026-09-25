# Thiết kế notebook demo ANN

Notebook dùng dữ liệu Iris để minh họa một ANN tự viết bằng NumPy. `scikit-learn` chỉ cung cấp dữ liệu qua `load_iris`; notebook không dùng model ANN có sẵn.

## Mục tiêu notebook

Notebook cần trả lời câu hỏi:

> Một ANN gồm những layer nào, dữ liệu đi qua từng layer ra sao, và weights học được bằng cách nào?

Model có cấu hình:

```text
4 input -> 8 hidden neurons with tanh -> 3 output neurons with softmax
```

Model tự cài đặt:

- forward pass;
- `tanh` và `softmax`;
- cross-entropy loss;
- backpropagation;
- gradient descent;
- trace output của từng layer trên một mẫu hoa.

## Thứ tự cell

1. Import `numpy`, `matplotlib` và `load_iris`.
2. Đọc dữ liệu, chia train/validation/test và chuẩn hóa bằng mean/std của training set.
3. Chuyển nhãn thành one-hot.
4. Định nghĩa `SimpleANN` với `input_to_hidden_weights`, `hidden_layer_biases`, `hidden_to_output_weights` và `output_layer_biases`.
5. Chạy training, theo dõi validation loss/accuracy và lưu `loss_history`.
6. In shape của weights và bias để xác nhận cấu trúc `4 -> 8 -> 3`.
7. Chạy một mẫu qua `trace()` và hiển thị input, hidden output, output probabilities.
8. Vẽ training/validation loss curve và confusion matrix trên test set.
9. Hiển thị giao diện demo với ba khu vực Input layer, Hidden layer và Output layer.

Notebook chạy thật nằm ở [`ann_from_scratch_demo.ipynb`](./ann_from_scratch_demo.ipynb). File này giữ vai trò checklist để notebook không quay lại dùng `MLPClassifier` hoặc pipeline model có sẵn; phần giải thích chi tiết vẫn nằm trong [`demo-directions.md`](./demo-directions.md).

## Output bắt buộc khi chạy demo

Với một mẫu hoa, notebook phải hiện:

```text
Raw input:                  (1, 4)
Hidden layer input:         (1, 4)
Hidden layer output:        (1, 8)
Output layer input:         (1, 8)
Output layer output:        (1, 3)
Predicted species:          một trong ba loài Iris
```

Phần giao diện có thể dùng bảng hoặc bar chart. Mỗi neuron ở hidden layer cần có một giá trị output để người xem thấy layer này thật sự đang tạo ra biểu diễn trung gian.

## Giới hạn của demo

Demo nhằm giải thích cách ANN hoạt động và được train. Validation dùng trong lúc phát triển model; test chỉ dùng ở bước đánh giá cuối. Accuracy phụ thuộc vào cách chia dữ liệu, weights khởi tạo, learning rate và số epoch. Không ghi sẵn accuracy trước khi chạy notebook.
