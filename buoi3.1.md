## Linear Regression

- Linear có thể gồm "Regression" hoặc "Classification"
- Linear Regression: dự đoán giá trị liên tục
- Linear Classification: dự đoán giá trị rời rạc

### Linear Regression

- X1: giá nhà
- X2: diện tích
- X3:....
=> X1, X2, X3, ... là CÁC FEATURES hay input variables, attributes

- Y: Gái nhà, tuổi nhà (kể từ thời điểm xây nhà đến nay)....
    - Nếu Y liên tục => Hổi quy
    - Nếu Y rời rạc => Classification


```


                    Y = Giá nhà
                    ^
                    |
                    |
                    |      #
                    |            #
                    |     #
            5Tỷ     |_____________________#
                    |
                    |                     |
                    |____________________________________> X (diện tích)
                    0                     70m^2



```

![Linear Regression](./imgs/buoi_3.1_vd_01.HEIC)

Từ 1 tập data như vậy => huẩn luyện mô hình từ diện tích để dự đoán giá nhà

- Tập data train không có nhà 20m^2 => model sẽ dự đoán được giá nhà (Có thể sai nếu data quá ít.)


### Bước 1: Định nghỉa model

- 1 simple linear regression model:
    - f(x) = y = b0+ b1*X1
    - b0: intercept (giá trị của y khi X = 0)
    - b1: slope (độ dốc của đường thẳng)
    - X1: diện tích
    => Học b0 và b1. (Tìm b0 và b1)

- Tìm best value of B0 và B1 sao cho đường thẳng Y đi qua nhiều x nhất.

- Vậy thế nào là best value?
    - Lost function: hàm mất mát. 
        - Trong trường hợp này lost function sẽ là tổng khoản cách của các tọa độ tạo thành từ x,y dến đường thẳng Y = b0 + b1*X1
        - Tìm cập b0, b1 để tổng khoản cách là nhỏ nhất.

        - VD: Có 1 tập data : (x,y) = (1,2), (2,3), (3,4), (4,5), (5,6)
        - nẾU b0 = 0 và b1 = 1 thì đường thẳng Y = 0 + 1*X1 = X1
            - Tổng khảon cachs là A1
        - nếu b0 = 1 và b1 = 1 thì đường thẳng Y = 1 + 1xX1 = 1
            - Tổng khảon cachs là A2
        - Nếu A2< A1 => cập b0, b1 = (1,1) là tốt nhất.
        - Nếu A2> A1 => cập b0, b1 = (0,1) là tốt nhất.


- 1 model phức tạp hơn:
    - x1: diện tích
    - x2: số phòng
    - x3: số tầng

    - y = b0 + b1*x1 + b2*x2 + b3*x3
    -> Tìm b0, b1, b2, b3 sao cho tổng khoản cách là nhỏ nhất.


- Chuyển vị là gì?

Chuyển vị (transpose): đổi hàng ↔ cột, ký hiệu Aᵀ.
Aᵀ[j, i] = A[i, j]

```

Ví dụ:
A = | 1 2 3 |        Aᵀ = | 1 4 |
    | 4 5 6 |             | 2 5 |
                          | 3 6 |

```

(Không nhầm với ma trận đơn vị I = diag(1,1,...,1).)

- Tại sao cần Xᵀ trong linear regression?

ŷ = Xb, loss L = ||y - Xb||²
min L ⇒ (Xᵀ X) b = Xᵀ y  ⇒  b = (Xᵀ X)⁻¹ Xᵀ y

Xᵀ giúp tạo XᵀX vuông để giải đồng thời b0, b1, b2, b3
(không tách thành các model 1 biến độc lập).


- Tại sao cần ma trận chuyển vị?

Mục tiêu không phải min(y), mà min tổng bình phương sai số:

L(b) = Σ (yᵢ - (b0 + b1·x1ᵢ + b2·x2ᵢ + b3·x3ᵢ))²

Viết dạng ma trận: ŷ = Xb, với mỗi hàng X = [1, x1, x2, x3].

Đạo hàm L theo b, cho = 0 → phương trình chuẩn:

  (Xᵀ X) b = Xᵀ y
  ⇒ b = (Xᵀ X)⁻¹ Xᵀ y

Xᵀ dùng để:
1. Đổi X (n×4) thành Xᵀ (4×n)
2. Tạo XᵀX (4×4) vuông → nghịch đảo được
3. Giải **đồng thời** b0, b1, b2, b3 (không tách 3 model độc lập,
   vì các feature thường tương quan; XᵀX có phần tử ngoài đường chéo)

Chuyển vị: Aᵀ[j,i] = A[i,j] (đổi hàng ↔ cột), không phải ma trận đơn vị chọn từng biến.

### VD đơn giản: 2 biến (diện tích + số phòng)

y = b0 + b1·x1 + b2·x2
- x1: diện tích (m²)
- x2: số phòng
- y: giá nhà (tỷ)

Data (3 mẫu):

| nhà | x1 | x2 | y |
|-----|----|----|---|
| 1   | 50 | 2  | 3 |
| 2   | 60 | 3  | 4 |
| 3   | 80 | 3  | 5 |

Ma trận X (cột 1 = 1 cho intercept), y:

```
X = | 1  50  2 |      y = | 3 |
    | 1  60  3 |          | 4 |
    | 1  80  3 |          | 5 |

Xᵀ = | 1   1   1  |
     | 50  60  80 |
     | 2   3   3  |


Công thức:
  b = (Xᵀ X)⁻¹ Xᵀ y
  với b = [b0, b1, b2]ᵀ

Ý: tìm 1 bộ (b0,b1,b2) sao cho
  L = (3-ŷ1)² + (4-ŷ2)² + (5-ŷ3)²  nhỏ nhất
  ŷi = b0 + b1·x1i + b2·x2i
```

Không tách thành model chỉ x1 rồi model chỉ x2 rồi cộng lại.


Đây là **quá trình train** (GD), không phải đáp án cuối.

Data: 3 nhà, bắt đầu `b = (0, 0, 0)`, learning rate `α = 0.00001`.

## Vòng 0 — chưa học gì

| nhà | x1 | x2 | y thật | ŷ đoán |
|-----|----|----|--------|--------|
| 1 | 50 | 2 | 3 | **0** |
| 2 | 60 | 3 | 4 | **0** |
| 3 | 80 | 3 | 5 | **0** |

Loss = \(3^2+4^2+5^2 = 50\) (sai nặng)

## Vòng 1 — sửa `b` lần đầu

Sau 1 bước GD:

\[
b \approx (0.00012,\ 0.0079,\ 0.00033)
\]

Dự đoán lại:

| nhà | ŷ |
|-----|------|
| 1 | 0.40 |
| 2 | 0.48 |
| 3 | 0.63 |

Loss ≈ **38.3** (đã giảm)

## Các vòng tiếp — `b` và ŷ đổi dần

| vòng | b0 | b1 | b2 | ŷ (3 nhà) | loss |
|------|------|------|------|-----------|------|
| 0 | 0 | 0 | 0 | 0, 0, 0 | 50.0 |
| 1 | 0.00012 | 0.0079 | 0.00033 | 0.40, 0.48, 0.63 | 38.3 |
| 2 | 0.00022 | 0.0148 | 0.00062 | 0.74, 0.89, 1.19 | 29.3 |
| 3 | 0.00032 | 0.0209 | 0.00087 | 1.05, 1.25, 1.67 | 22.4 |
| 5 | 0.00047 | 0.0308 | 0.00129 | 1.54, 1.85, 2.47 | 13.2 |
| 10 | 0.00071 | 0.0465 | 0.00195 | 2.33, 2.80, 3.73 | 3.5 |
| 50 | 0.00095 | 0.0630 | 0.00269 | 3.16, 3.79, 5.05 | 0.07 |

Thấy rõ: **ŷ bò dần về gần (3, 4, 5)**, loss tụt — đó là train.

## Mỗi vòng máy làm gì? (không thử vét số)

```
có b hiện tại
  → đoán ŷ cho cả 3 nhà
  → xem sai (ŷ - y)
  → chỉnh một chút cả b0,b1,b2 theo hướng giảm sai
  → lặp
```

## Đọc quá trình

- Đầu: đoán toàn 0 → sai to  
- Vài vòng: đoán ~1–2 → đỡ hơn  
- ~50 vòng: đoán ~3.2, 3.8, 5.0 → gần data  
- Thêm nhiều vòng nữa: tinh chỉnh chậm (vì `x1` lớn, GD chậm/lệch scale)

**Train = xem bảng trên chạy xuống**, không phải nhảy có `b` đẹp rồi thế vào.