# n neuron trong một ví dụ rất đơn giản

## Tiếp nối phần trước:

- 1 neuron: `z = w_1*x_1 + w_2*x_2 + b`, rồi `a = f(z)`. Đủ cho bài **2 lớp** (Đậu/Rớt) nếu dùng step/sigmoid ở output.
- Khi cần **nhiều lớp**, hoặc cần vài “cảm biến” trung gian trước khi quyết định cuối → dùng **n neuron** (cùng nhận input `x`, mỗi neuron có bộ `w`, `b` riêng).

## Ý tưởng:

- Neuron thứ `i` vẫn làm đúng một việc:

```text
z_i = w_i1*x_1 + w_i2*x_2 + b_i
a_i = f(z_i)
```

- n neuron cạnh nhau = 1 **layer** gồm n output `a_1, a_2, ..., a_n`.
- Hai chỗ hay gặp n neuron:
  - **Output nhiều lớp**: mỗi neuron “điểm” cho một lớp; Softmax biến các điểm thành xác suất, rồi `argmax` lấy lớp thắng.
  - **Hidden layer**: các `a_i` là tín hiệu trung gian (thường dùng ReLU/Tanh), rồi layer sau mới ra Đậu/Rớt.

## Ví dụ 1: 3 lớp = 3 neuron output

Bài toán: xếp loại `["yếu", "TB", "giỏi"]`. Cùng input sinh viên `x_1 = 0.8`, `x_2 = 0.6`.

Mỗi lớp một neuron (số giả định để tính tay):

```text
z_yếu  = 0.2*0.8 + 0.1*0.6 - 1.0 = -0.78
z_TB   = 0.5*0.8 + 0.4*0.6 - 0.5 = 0.14
z_giỏi = 0.9*0.8 + 0.8*0.6 - 0.3 = 0.90
```

- Softmax trên `(z_yếu, z_TB, z_giỏi)` → ba xác suất cộng = 1.
- `argmax` → lớp **giỏi** (vì `z_giỏi` lớn nhất). Không cần Softmax cũng chọn được lớp nếu chỉ cần nhãn cứng; Softmax cho thêm “chắc chắn bao nhiêu”.

## Ví dụ 2: 2 neuron ẩn, rồi 1 neuron quyết định

Vẫn bài Đậu/Rớt. Layer ẩn có 2 neuron (ReLU), layer ra có 1 neuron (step/sigmoid):

```text
# hidden
z_h1 = w11*x_1 + w12*x_2 + b_h1
a_h1 = ReLU(z_h1)

z_h2 = w21*x_1 + w22*x_2 + b_h2
a_h2 = ReLU(z_h2)

# output: input của neuron cuối là (a_h1, a_h2), không còn là (x_1, x_2)
z_out = w1*a_h1 + w2*a_h2 + b_out
a_out = sigmoid(z_out)   # hoặc step(z_out)
```

- Mỗi neuron ẩn “nhìn” `x` theo một cách khác (bộ `w` khác nhau).
- Neuron cuối chỉ kết hợp các tín hiệu ẩn → đây là MLP siêu nhỏ (2 hidden + 1 output).

## Cần nhớ:

- **n neuron ≠ n lớp mạng**. n neuron thường là **chiều rộng** của một layer; **số layer** mới là độ sâu → [`very-simple-depth.md`](./very-simple-depth.md).
- 1 neuron = 1 đường thẳng / siêu phẳng. Nhiều neuron + activation phi tuyến mới ghép được ranh giới cong (XOR, `make_moons`, ...).
- Phần sau (`overview.md`, `how-it-works.md`) nói rõ hơn về MLP và forward pass trên cả mạng.
