# Hidden layer trong web demo (từ logistic regression)

Bạn đã biết logistic regression = **1 neuron**:

```text
z = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b
ŷ = sigmoid(z)     → xác suất 1 lớp
```

ANN trong `gk/web` = xếp nhiều neuron kiểu đó thành 2 tầng: **hidden** rồi **output**.

Cấu hình mặc định: `4 → 8 → 3`

---

## Input (không phải neuron)

4 số đo (sau chuẩn hóa mean/std của train):

```text
x = [x1, x2, x3, x4]
```

Tầng này chỉ đưa data vào, không có weight để học.

---

## Hidden layer: đúng 8 neuron

Mỗi neuron hidden = **1 logistic nhỏ** (cùng công thức `z = w·x + b`, chỉ khác activation `f`).

### Neuron H1

```text
z1 = w11*x1 + w12*x2 + w13*x3 + w14*x4 + b1
h1 = f(z1)
```

### Neuron H2

```text
z2 = w21*x1 + w22*x2 + w23*x3 + w24*x4 + b2
h2 = f(z2)
```

### Neuron H3

```text
z3 = w31*x1 + w32*x2 + w33*x3 + w34*x4 + b3
h3 = f(z3)
```

### Neuron H4

```text
z4 = w41*x1 + w42*x2 + w43*x3 + w44*x4 + b4
h4 = f(z4)
```

### Neuron H5

```text
z5 = w51*x1 + w52*x2 + w53*x3 + w54*x4 + b5
h5 = f(z5)
```

### Neuron H6

```text
z6 = w61*x1 + w62*x2 + w63*x3 + w64*x4 + b6
h6 = f(z6)
```

### Neuron H7

```text
z7 = w71*x1 + w72*x2 + w73*x3 + w74*x4 + b7
h7 = f(z7)
```

### Neuron H8

```text
z8 = w81*x1 + w82*x2 + w83*x3 + w84*x4 + b8
h8 = f(z8)
```

Trong đó `f` là 1 trong: `tanh`, `sigmoid`, `relu`, `leaky_relu`, `softplus`, `identity`.

Sau hidden layer ta có:

```text
h = [h1, h2, h3, h4, h5, h6, h7, h8]
```

- Mỗi neuron có **bộ weight riêng** → nhìn cùng `x` theo cách khác nhau.
- Output `hi` **không phải nhãn class**, chỉ là tín hiệu trung gian.
- Số 8 **không tính ra từ data** — chọn tay cho demo (UI chỉnh được 1–32).

Trong code (`ann_core.py`):

```text
input_to_hidden_weights  shape (4, 8)   ← cột i = weight của neuron Hi
hidden_layer_biases      shape (1, 8)   ← bias b1…b8
```

---

## Output layer: đúng 3 neuron (3 class)

Không nhìn `x` nữa, nhìn `h`.

### Neuron O0 (class 0)

```text
s0 = v01*h1 + v02*h2 + v03*h3 + v04*h4
   + v05*h5 + v06*h6 + v07*h7 + v08*h8 + c0
```

### Neuron O1 (class 1)

```text
s1 = v11*h1 + v12*h2 + v13*h3 + v14*h4
   + v15*h5 + v16*h6 + v17*h7 + v18*h8 + c1
```

### Neuron O2 (class 2)

```text
s2 = v21*h1 + v22*h2 + v23*h3 + v24*h4
   + v25*h5 + v26*h6 + v27*h7 + v28*h8 + c2
```

Rồi cả 3 điểm số qua **softmax**:

```text
P(class 0), P(class 1), P(class 2)   (cộng lại = 1)
```

Chọn class có P lớn nhất.

Trong code:

```text
hidden_to_output_weights  shape (8, 3)
output_layer_biases       shape (1, 3)
```

---

## So sánh nhanh với logistic

| | Logistic bạn học | Web demo |
|---|---|---|
| Số neuron | 1 | 8 hidden + 3 output |
| Input | `x` | hidden nhìn `x`, output nhìn `h` |
| Output | P(1 lớp) qua sigmoid | P(3 lớp) qua softmax |
| Học gì | 1 bộ `w, b` | mọi `w, b` của 11 neuron |

Một neuron luôn chỉ làm việc này:

```text
nhận số vào → nhân weight + cộng bias → qua hàm f → ra 1 số
```
