# ANN hoạt động như thế nào?

Một ANN feed-forward nhận input ở đầu vào, truyền qua các hidden layer rồi tạo output. Trong một lượt dự đoán, dữ liệu chỉ đi theo chiều từ trước ra sau. Lượt này gọi là forward propagation hoặc forward pass.

## 1. Forward pass của một neuron

Với vector input `x`, vector trọng số `w` và bias `b`, neuron tính:

```text
z = w^T*x + b
a = f(z)
```

`w^T*x` là tích vô hướng. `f` là activation function. Một neuron không ghi nhớ toàn bộ dữ liệu đầu vào. Nó chỉ dùng các tham số `w`, `b` và quy tắc `f` để biến input thành output.

## 2. Forward pass qua hidden layer

Giả sử mạng có input `x`, một hidden layer và một output layer:

```text
x -> hidden -> output
```

Hidden layer có thể viết bằng dạng ma trận:

```text
h = f(W_1*x + b_1)
y_hat = g(W_2*h + b_2)
```

Trong đó:

- `W_1`, `b_1` là trọng số và bias nối input với hidden layer.
- `h` là biểu diễn trung gian do hidden layer tạo ra.
- `W_2`, `b_2` nối hidden layer với output layer.
- `f` là activation của hidden layer.
- `g` là activation của output layer.
- `y_hat` là giá trị dự đoán được.

Khi có nhiều hidden layer, mẫu tính toán lặp lại:

```text
h_1 = f_1(W_1*x + b_1)
h_2 = f_2(W_2*h_1 + b_2)
...
y_hat = g(W_L*h_(L-1) + b_L)
```

`7-nn1-intro.ppt.pdf` mô tả neuron theo đúng ba bước: tính weighted sum, cộng bias và đưa qua activation. `mit15_773_s24_lec01.pdf` dùng một mạng nhỏ để tính output của từng hidden node bằng ReLU trước khi tính output cuối.

## 3. Vì sao hidden layer giúp tạo ranh giới phi tuyến?

Một perceptron có thể tạo một đường thẳng hoặc siêu phẳng. Hidden layer tạo ra các biến đổi mới từ input. Output layer sau đó kết hợp các biến đổi này.

Với một hidden layer:

- Mỗi hidden neuron tạo một phản hồi khác nhau từ cùng input.
- Activation làm cho phản hồi không còn chỉ là phép biến đổi tuyến tính.
- Output layer kết hợp các phản hồi để tạo decision boundary mới.

Đây là lý do MLP có thể xử lý các mẫu như XOR hoặc `make_moons`, trong khi một model tuyến tính gặp giới hạn.

Không nên diễn giải mỗi hidden neuron như một đặc trưng có ý nghĩa rõ ràng với con người. Hidden layer tạo ra một biểu diễn trung gian hữu ích cho toàn mạng, nhưng một neuron riêng lẻ không nhất thiết tương ứng với một khái niệm đơn giản.

## 4. Activation function thay đổi điều gì?

Activation function nhận một giá trị `z` sau weighted sum và bias, rồi trả về output của neuron. Nó quyết định neuron giữ nguyên, giới hạn, cắt bỏ hoặc biến đổi giá trị đầu vào như thế nào.

Ngưỡng hoạt động là mức dùng để nói neuron có "fire" hay không. Step và sign dùng ngưỡng rõ ràng, thường tại `z = 0`. Sigmoid và tanh không bật hoặc tắt đột ngột. Chúng thay đổi dần theo `z`, nhưng ta vẫn có thể dùng một ngưỡng ở bước cuối để biến output thành nhãn.

Nếu mọi layer chỉ dùng activation tuyến tính, phép hợp thành của các layer vẫn là một phép tuyến tính. Activation phi tuyến giúp mạng biểu diễn các ranh giới phức tạp hơn.

### Tanh

```text
tanh(z) nằm trong khoảng (-1, 1)
```

Tanh thường dễ minh họa trong lớp vì output có cả giá trị âm và dương.

Ví dụ, `tanh(0) = 0`. Khi `z` rất dương, output tiến gần `1`; khi `z` rất âm, output tiến gần `-1`.

### Sigmoid

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

Sigmoid đưa output về khoảng `(0, 1)`, phù hợp với trực giác xác suất trong một số bài toán phân loại nhị phân.

Ví dụ, `sigmoid(0) = 0.5`. Một quy tắc phân loại thường dùng ngưỡng `0.5`, nhưng ngưỡng này là quy tắc chuyển xác suất thành nhãn, không phải bản thân sigmoid.

### ReLU

```text
ReLU(z) = max(0, z)
```

ReLU giữ phần dương và triệt tiêu phần âm. `mit15_773_s24_lec01.pdf` dùng ReLU trong ví dụ forward pass của hidden layer.

Ví dụ, `ReLU(-2) = 0` và `ReLU(2) = 2`.

## 5. Liên hệ với `make_moons`

`make_moons` tạo hai cụm điểm có hình dạng giống hai cung trăng. Mỗi điểm có hai feature để ta vẽ trực tiếp trên mặt phẳng.

Notebook sẽ tạo ba hình ảnh chính:

1. Dữ liệu gốc với màu biểu diễn lớp.
2. Decision boundary của logistic regression.
3. Decision boundary của MLP.

Nếu logistic regression không bao quanh được hai cung trăng bằng một đường thẳng, đó là giới hạn của ranh giới tuyến tính. Nếu MLP học được vùng phân chia cong hơn, hình ảnh cho thấy hidden layer đã giúp model biểu diễn quan hệ phi tuyến.

Hình vẽ chỉ minh họa hành vi của model trên dataset và cấu hình cụ thể. Không dùng hình này để kết luận MLP luôn tốt hơn mọi model khác.

## 6. Từ dự đoán sang học

Forward pass chỉ tạo ra dự đoán. Để học, mạng cần biết dự đoán sai bao nhiêu. Ta dùng loss function:

```text
loss = loss(y, y_hat)
```

Sau đó backpropagation tính ảnh hưởng của từng trọng số đến loss. Gradient descent dùng các gradient đó để cập nhật trọng số theo hướng làm loss giảm.

Chuỗi đầy đủ là:

```text
input
  -> forward pass
  -> prediction
  -> loss
  -> backpropagation
  -> update weights and biases
  -> lặp lại
```

Phần chi tiết về loss, gradient descent và backpropagation nằm trong [`how-to-train.md`](./how-to-train.md).

## Nguồn đối chiếu

- `7-nn1-intro.ppt.pdf`, phần Artificial Neuron Model, Bias và Activation Functions.
- `chap4_ann.pdf`, slide 11 đến 19, về MLP, activation, gradient descent và backpropagation.
- `lecture-21.pdf`, trang 1 đến 8, về công thức của mạng feed-forward và activation.
- `mit15_773_s24_lec01.pdf`, các slide 94 đến 104, về cấu hình và tính output của một mạng nhỏ.
