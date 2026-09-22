# ANN là gì?

Artificial Neural Network (ANN) là một mô hình học máy gồm nhiều đơn vị tính toán đơn giản, thường gọi là neuron, nối với nhau bằng các trọng số. Mỗi neuron nhận input, tính tổng có trọng số, cộng bias rồi đưa kết quả qua một activation function.

Một mục tiêu của ANN là học một hàm ánh xạ từ input sang output. Với bài toán phân loại, output thường là nhãn hoặc xác suất của nhãn. Với bài toán hồi quy, output thường là một hoặc nhiều giá trị số.

## Đọc trước (ví dụ Đậu/Rớt)

Hai file này đi từ model tuyến tính tới 1 neuron, làm bằng tay trước khi đọc phần tổng quan dưới đây:

1. [`very-simple-liner-model.md`](./very-simple-liner-model.md): `z = w_1*x_1 + w_2*x_2 + b`, vì sao cần `b`, và `if z >= 0` để ra Đậu/Rớt.
2. [`very-simple-ann.md`](./very-simple-ann.md): thay bước if bằng **activation function**; đây mới chỉ là 1 neuron / perceptron.
3. [`very-simple-n-neurons.md`](./very-simple-n-neurons.md): **chiều rộng** — n neuron cạnh nhau (Softmax nhiều lớp hoặc 1 hidden).
4. [`very-simple-depth.md`](./very-simple-depth.md): **chiều sâu** — xếp nhiều layer nối tiếp; vì sao cần activation phi tuyến giữa các layer.

File này (`overview.md`) bước tiếp: MLP đầy đủ hơn, phi tuyến, so sánh logistic regression, và giới hạn cần nói khi thuyết trình.

## Một neuron nhân tạo

Với input `x_1, x_2, ..., x_d`, trọng số `w_1, w_2, ..., w_d` và bias `b`, neuron tính:

```text
z = w_1*x_1 + w_2*x_2 + ... + w_d*x_d + b
a = f(z)
```

Trong đó:

- `z` là tổng đầu vào trước activation.
- `b` là bias, giúp neuron dịch ngưỡng hoạt động.
- `f` là activation function.
- `a` là output của neuron.

Nếu bỏ activation phi tuyến và chỉ dùng các phép biến đổi tuyến tính, nhiều layer nối tiếp vẫn có thể rút gọn thành một phép biến đổi tuyến tính. Vì vậy activation phi tuyến là thành phần quan trọng khi ANN cần học ranh giới phi tuyến.

### Phi tuyến là gì?

Một quan hệ tuyến tính có thể mô tả bằng một đường thẳng trong mặt phẳng, hoặc bằng một siêu phẳng khi có nhiều feature. Một hàm phi tuyến không bị giới hạn bởi dạng đường thẳng đó. Khi vẽ ra, ranh giới của nó có thể cong, gấp khúc hoặc có nhiều vùng khác nhau.

Ví dụ:

- `y = 2*x` là hàm tuyến tính.
- `y = 2*x + 1` là hàm affine, vẫn tạo ra một đường thẳng và thường được gọi chung là biến đổi tuyến tính trong machine learning.
- `y = x^2` là hàm phi tuyến vì đồ thị không phải đường thẳng.
- `ReLU(x) = max(0, x)` cũng là phi tuyến dù từng đoạn của nó là đường thẳng.

Trong ANN, tính phi tuyến đến từ activation function. Nếu xếp nhiều layer tuyến tính mà không có activation phi tuyến, toàn bộ mạng vẫn chỉ tạo ra một phép biến đổi tuyến tính. Vì vậy nhiều layer không tự động làm model phi tuyến.

## Perceptron

Perceptron là dạng đơn giản của ANN, thường có một neuron đầu ra và activation dạng ngưỡng hoặc sign. Nó học một quyết định tuyến tính:

```text
y_hat = sign(w^T*x + b)
```

Ranh giới quyết định của perceptron là một đường thẳng trong hai chiều hoặc một siêu phẳng trong nhiều chiều. Vì lý do này, perceptron không thể phân tách hoàn hảo các dữ liệu như XOR bằng một neuron đơn.

`chap4_ann.pdf`, slide 2 đến 10, trình bày neuron, perceptron, luật cập nhật trọng số và ví dụ XOR.

## Mạng nhiều lớp (MLP)

Multi-Layer Perceptron (MLP) là mạng feed-forward có một hoặc nhiều hidden layer. Dữ liệu đi theo chiều:

```text
input layer -> hidden layer -> output layer
```

Mỗi hidden layer nhận activation từ layer trước, thực hiện phép biến đổi có trọng số và activation, rồi gửi kết quả sang layer tiếp theo. Nhiều hidden layer cho phép mạng tạo ra biểu diễn trung gian phức tạp hơn.

Mạng đơn giản có một hidden layer đã đủ để minh họa một điểm quan trọng của ANN: kết hợp nhiều neuron phi tuyến có thể tạo ra decision boundary phi tuyến.

## Activation function

Activation function là hàm được áp dụng sau weighted sum và bias:

```text
z = w^T*x + b
a = f(z)
```

Activation biến `z` thành output `a` của neuron. Nó có thể giới hạn khoảng giá trị, tạo ngưỡng hoặc thêm tính phi tuyến cho mạng.

### Ngưỡng hoạt động

Ngưỡng hoạt động là mức mà tại đó ta quyết định neuron được xem là đang hoạt động hoặc đang "fire". Với step function đơn giản, ngưỡng thường là `0`:

```text
step(z) = 1 nếu z >= 0
          0 nếu z < 0
```

Nếu `z` vượt ngưỡng, neuron trả về lớp hoặc trạng thái hoạt động. Nếu `z` thấp hơn ngưỡng, neuron trả về trạng thái không hoạt động. Trong thực tế, bias làm thay đổi vị trí ngưỡng vì điều kiện hoạt động là `w^T*x + b >= 0`.

Không nên nhầm hai cách dùng từ "ngưỡng":

- Ngưỡng của activation step là một phần của hàm neuron.
- Ngưỡng `0.5` khi chuyển xác suất sigmoid thành nhãn `0` hoặc `1` là quy tắc ra quyết định ở bước phân loại.

### Các activation thường gặp

| Activation | Công thức hoặc quy tắc | Hiểu đơn giản | Ghi chú |
| --- | --- | --- | --- |
| Linear | `f(z) = z` | Giữ nguyên giá trị đầu vào | Không tạo thêm tính phi tuyến; thường dùng ở output hồi quy |
| Step | `1` nếu `z >= 0`, ngược lại `0` | Bật hoặc tắt theo ngưỡng | Dễ giải thích nhưng hàm không trơn nên không thuận tiện cho gradient descent |
| Sign | `1` nếu `z >= 0`, ngược lại `-1` | Giống Step nhưng dùng hai giá trị `-1` và `1` | Thường xuất hiện trong ví dụ perceptron |
| Sigmoid | `1 / (1 + exp(-z))` | Ép giá trị vào khoảng `0` đến `1` | Có thể diễn giải như điểm xác suất trong bài toán nhị phân |
| Tanh | `(exp(z) - exp(-z)) / (exp(z) + exp(-z))` | Ép giá trị vào khoảng `-1` đến `1` | Output trung tâm tại `0`, thường dễ minh họa trong hidden layer |
| ReLU | `max(0, z)` | Giữ phần dương, bỏ phần âm | Đơn giản, thường được dùng trong mạng nhiều lớp |
| Softmax | ép nhiều điểm số thành xác suất cộng = 1 | Phân phối xác suất trên các lớp | Thường dùng ở **output** khi có nhiều lớp |

Chi tiết ngắn và ví dụ Đậu/Rớt: xem lại [`very-simple-ann.md`](./very-simple-ann.md).

Ví dụ với `z = -2`, `0` và `2`:

| `z` | Linear | Step | Sign | Sigmoid gần đúng | Tanh gần đúng | ReLU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -2 | -2 | 0 | -1 | 0.12 | -0.96 | 0 |
| 0 | 0 | 1 | 1 | 0.50 | 0 | 0 |
| 2 | 2 | 1 | 1 | 0.88 | 0.96 | 2 |

Các giá trị gần đúng trong bảng chỉ giúp hình dung hình dạng của hàm. Khi làm bài toán thật, không chọn activation chỉ vì một bảng ví dụ nhỏ.

Trong phần demo, `tanh` và `relu` được dùng để người xem quan sát ảnh hưởng của activation. Không nên kết luận một activation luôn tốt hơn activation khác chỉ từ một dataset nhỏ.

## Logistic regression và neural network

Logistic regression tính một weighted sum rồi đưa qua sigmoid để tạo xác suất cho lớp dương. Nếu nhìn theo cách tổ chức của neural network, logistic regression có thể xem là mạng đơn giản không có hidden layer.

Điểm so sánh này phù hợp với demo:

- Logistic regression tạo ranh giới tuyến tính.
- MLP thêm hidden layer và activation phi tuyến.
- Cả hai đều học tham số từ dữ liệu, nhưng MLP có không gian mô hình linh hoạt hơn.

`lecture-21.pdf`, phần mở đầu, và `mit15_773_s24_lec01.pdf`, các slide về logistic regression và hidden layer, dùng cùng trực giác này.

## Phân loại và hồi quy

ANN có thể dùng cho nhiều loại output:

- Phân loại nhị phân: một output biểu diễn xác suất hoặc điểm số cho lớp.
- Phân loại nhiều lớp: nhiều output hoặc cơ chế mã hóa phù hợp với số lớp.
- Hồi quy: output tuyến tính biểu diễn giá trị số.

Trong notebook của project, bài toán là phân loại nhị phân trên dữ liệu hai chiều `make_moons`. Việc chọn dữ liệu hai chiều giúp decision boundary dễ vẽ và dễ giải thích trong lớp.

## Ưu điểm và giới hạn

ANN có thể học quan hệ phi tuyến và tự học trọng số cho các input. Mạng nhiều lớp cũng có thể tạo ra các biểu diễn trung gian hữu ích.

Tuy nhiên, ANN không tự động giải quyết mọi vấn đề. Các giới hạn cần nói trong thuyết trình gồm:

- Model lớn có thể overfit.
- Kết quả phụ thuộc vào dữ liệu, kiến trúc và tham số huấn luyện.
- Dữ liệu nhiễu hoặc thiếu giá trị có thể làm việc học khó hơn.
- Gradient descent có thể cần nhiều bước và việc chọn cấu hình không phải lúc nào cũng rõ ràng.
- Decision boundary đẹp trên một dataset nhỏ không chứng minh model luôn tốt trên dữ liệu mới.

`chap4_ann.pdf`, slide 20 đến 22, nêu các vấn đề về thiết kế mạng, overfitting, noise và chi phí huấn luyện.

## Ý chính cần nhớ

ANN không phải là một phép tính bí ẩn. Chuỗi trực giác từ hai file ví dụ đơn giản:

1. Tính `z = w·x + b` (model tuyến tính).
2. Áp dụng activation (1 neuron).
3. Xếp nhiều neuron / hidden layer → MLP, học được ranh giới phi tuyến.
4. So sánh dự đoán với nhãn qua loss.
5. Điều chỉnh trọng số để loss giảm.

Chi tiết forward / train: [`how-it-works.md`](./how-it-works.md), [`how-to-train.md`](./how-to-train.md).
