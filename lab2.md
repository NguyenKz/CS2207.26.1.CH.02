# Ghi chú: Hypothesis Testing (Kiểm định giả thuyết)

Nội dung được chuyển từ bảng, đồng thời bổ sung các công thức còn thiếu để áp dụng vào bài tập Python.

## 1. Phân phối chuẩn (Normal Distribution)

Giả sử dữ liệu có phân phối chuẩn:

\\[ X\sim N(\mu,\sigma^2) \\]

Trong đó:

- \\(X\\): Biến ngẫu nhiên, ví dụ drug response.
- \\(\mu\\): Giá trị trung bình (mean).
- \\(\sigma^2\\): Phương sai (variance).
- \\(\sigma\\): Độ lệch chuẩn (standard deviation).
- \\(n\\): Số lượng mẫu.

Trong bài tập:

```
n = 100mu = 0std = 1x = np.random.normal(loc=mu, scale=std, size=(n, 1))
```

## 2. Hypothesis (Giả thuyết)

Đặt hai giả thuyết:

\\[ H_0:\mu=0 \\]

\\[ H_1:\mu\neq0 \\]

- \\(H_0\\): Null hypothesis (giả thuyết gốc).
- \\(H_1\\): Alternative hypothesis (giả thuyết đối).

Đây là kiểm định hai phía (two-sided test).

## 3. Test Statistic (Thống kê kiểm định)

Tính trung bình mẫu:

\\[ \bar X=\frac{1}{n}\sum\_{i=1}^{n}X_i \\]

Công thức thống kê kiểm định Z:

\\[ T=\frac{\bar X-\mu_0}{\sigma/\sqrt n} \\]

Trong đó \\(\mu_0=0\\) và \\(\sigma=1\\).

Dưới giả thuyết \\(H_0\\):

\\[ T\sim N(0,1) \\]

Áp dụng vào Python:

```
x_mean = np.mean(x)T = (x_mean - 0) / (std / np.sqrt(n))
```

Lưu ý: Vì bài tập cho biết độ lệch chuẩn tổng thể bằng 1, ta dùng Z-test. Nếu không biết độ lệch chuẩn tổng thể và phải ước lượng từ mẫu, thường dùng t-test với phân phối Student.

## 4. Compute p-value (Tính p-value)

Công thức trên bảng:

\\[ p\text{-value}=P\_{H_0}(|T|\geq|T^{obs}|) \\]

Với kiểm định hai phía:

\\[ \boxed{p=2(1-\Phi(|T^{obs}|))} \\]

Trong đó:

- \\(T^{obs}\\): Giá trị thống kê tính từ dữ liệu quan sát.
- \\(\Phi\\): Hàm phân phối tích lũy chuẩn.
- p-value: Xác suất nhận được giá trị thống kê ít nhất cực đoan như quan sát, giả sử \\(H_0\\) đúng.

-4-3-2-112340.10.20.30.4\\(z\\)\\(f(z)\\)z = 1.96

Two-sided: both shaded tails combine to p ≈ 0.0500

\\(z\\)

Sides

12

Give feedback

Trong Python:

```
p_value = 2 * norm.sf(abs(T))
```

`norm.sf()` tính xác suất đuôi phải của phân phối chuẩn.

## 5. Decision (Ra quyết định)

Chọn mức ý nghĩa:

\\[ \alpha=0.05 \\]

| Điều kiện      | Quyết định               |
| -------------- | ------------------------ |
| p-value < 0.05 | Reject \\(H_0\\)         |
| p-value ≥ 0.05 | Fail to reject \\(H_0\\) |

Lưu ý: Không bác bỏ \\(H_0\\) không đồng nghĩa với chứng minh \\(H_0\\) đúng.

## 6. False Positive Rate

False positive là trường hợp:

- \\(H_0\\) thực sự đúng.
- Nhưng kết quả kiểm định lại bác bỏ \\(H_0\\).

Công thức:

\\[ FPR=\frac{\text{Số lần bác bỏ }H_0}{\text{Tổng số lần kiểm định khi }H_0\text{ đúng}} \\]

Với \\(\alpha=0.05\\), tỷ lệ false positive kỳ vọng xấp xỉ 5%.

# Áp dụng vào 3 bài tập Python

Exercise 1: Tính thống kê kiểm định \\(T\\), sau đó tính p-value.

Exercise 2: Lặp lại nhiều lần và vẽ phân phối của p-value. Khi \\(H_0\\) đúng, p-value về lý thuyết có phân phối đều trên khoảng [0, 1].

Exercise 3: Chạy kiểm định 1.000 lần, tính tỷ lệ có `p_value < 0.05` để ước lượng False Positive Rate.

## Code hoàn chỉnh

```
import numpy as npimport matplotlib.pyplot as pltfrom scipy.stats import normdef run():    # Generate synthetic data    n = 100    mu = 0    std = 1    x = np.random.normal(loc=mu, scale=std, size=(n, 1))    # H0: mu = 0 vs H1: mu != 0    # Construct test-statistic    x_mean = np.mean(x)    T = (x_mean - 0) / (std / np.sqrt(n))    # Compute p-value under H0    p_value = 2 * norm.sf(abs(T))    return p_valueif __name__ == '__main__':    # Exercise 1: Compute p-value    p_value = run()    print("p-value:", p_value)    # Exercise 2: Plot distribution of p-values    p_values = [run() for _ in range(1000)]    plt.hist(p_values, bins=20, edgecolor='black')    plt.xlabel('p-value')    plt.ylabel('Frequency')
```

Kết quả kỳ vọng: Histogram ở Exercise 2 tương đối đều trong khoảng 0 đến 1. Exercise 3 cho tỷ lệ false positive gần 0.05 (5%). Vì dữ liệu được sinh ngẫu nhiên nên kết quả mỗi lần chạy có thể khác nhau.