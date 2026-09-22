# 1 Mạng máy học đơn giản.

## Bài toán: sinh viên có qua môn không?

Ta muốn dùng hai thông tin để dự đoán kết quả:

```text
x_1: mức độ học bài, từ 0 đến 1
x_2: mức độ hoàn thành bài tập, từ 0 đến 1
```

Ví dụ một sinh viên có:

```text
x_1 = 0.8
x_2 = 0.6
```

Kết quả cần dự đoán:

```text
y = 1: qua môn
y = 0: không qua môn
```

Các giá trị `x_1`, `x_2` là dữ liệu của từng sinh viên. Sinh viên khác có thể có giá trị khác.

## Bước 1: Trọng số nói input đóng góp bao nhiêu

Giả sử hai thông tin có vai trò ngang nhau. Ta đặt:

```text
w_1 = 1
w_2 = 1
```

Khi đó phần điểm do sinh viên đóng góp là:

```text
w_1*x_1 + w_2*x_2
= 1*0.8 + 1*0.6
= 1.4
```

`w_1` đi với `x_1`, còn `w_2` đi với `x_2`. Nếu feature nào quan trọng hơn, model có thể học trọng số lớn hơn cho feature đó.

## Bước 2: Vì sao cần `b`?

Ta đặt ra quy tắc của bài toán:

> Tổng hai điểm (z) phải đạt ít nhất `1` thì sinh viên mới qua môn. | z >= 1

Điều kiện này đang dùng ngưỡng `1`, nhưng nhiều thuật toán máy học như ReLU, tanh, sigmoid được xây dựng dựa trên ngưỡng `0`. Nên ta cần `tịnh tiến` ngưỡng `1` về ngưỡng `0` bằng cách trừ `1` cho `z`.

```text
tổng điểm = z >= 1
<=> z - 1 >= 0

đặt b = -1

z - b >= 0 # Đây là dạng chuẩn được quy ước rộng rãi trong học máy

```

Vậy trong ví dụ này, `b` không phải con số được đưa vào tùy ý. `b = -1` xuất hiện vì bài toán yêu cầu tổng điểm phải đạt `1`.

Có thể nhìn trên một trục số:

```text
tổng điểm:       0 ------ 0.5 ------ 1 ------ 1.5
                                     ^ đạt từ đây

z = tổng điểm-1: -1 ----- -0.5 ----- 0 ------ 0.5
                                     ^ step chỉ cần kiểm tra từ 0
```

Nếu ngưỡng qua môn là 3 thì `z >= 3 <=> z - 3 >= 0` => `b = -3`.

## Bước 3: Tính một sinh viên cụ thể

Với sinh viên có `x_1 = 0.8`, `x_2 = 0.6`:

```text
z = 1*0.8 + 1*0.6 - 1
  = 0.4
```

Vì `z = 0.4 >= 0`, neuron dự đoán sinh viên qua môn.

Một sinh viên khác có `x_1 = 0.2`, `x_2 = 0.3`:

```text
z = 1*0.2 + 1*0.3 - 1
  = -0.5
```

Vì `z = -0.5 < 0`, neuron dự đoán sinh viên không qua môn.

## Vậy `b` sinh ra từ đâu, làm sao để tính được?

`b` cũng là 1 tham số tương tự `w_1`, `w_2`.
`b` được huấn luyện cùng với `w_1`, `w_2`. VD có thể chọn 'b', 'w\_1', 'w\_2' ban đầu là 0 rồi dự đoán, sau đó tính ra sai số rồi cập nhật lại 'b', 'w\_1', 'w\_2' cho tốt hơn, tiếp tục thử và tính sai số cho đến khi sai số nhỏ hơn một ngưỡng nào đó. => Đây là quá trình huấn luyện một model.






## Kết quả:

```text
model: z = w_1*x_1 + w_2*x_2 + b

if z >= 0 then y = 1
else y = 0

```

