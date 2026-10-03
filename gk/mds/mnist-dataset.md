# MNIST — Dataset chữ số viết tay

## Là gì?

**MNIST** (Modified NIST) — bộ ảnh số **0–9** viết tay, chuẩn trong dạy ML / thị giác máy tính.

Nguồn gốc: [Yann LeCun](https://yann.lecun.com/exdb/mnist/)

## Quy mô

### Tập gốc (MNIST)

| Tập   | Số mẫu |
| ----- | ------ |
| Train | 60 000 |
| Test  | 10 000 |

**Không có tập validation.**

### Cách nhóm chia

- Giữ nguyên **Test** (10 000)
- Chỉ chia **Train gốc** theo tỷ lệ **80 / 20** (seed 42):
  - **80% → Fit** = 48 000
  - **20% → Val** = 12 000

### Ba tập dùng khi train

| Tập  | Số mẫu | Tỷ lệ             | Vai trò                    |
| ---- | ------ | ----------------- | -------------------------- |
| Fit  | 48 000 | 80% của train gốc | Học trọng số               |
| Val  | 12 000 | 20% của train gốc | Theo dõi / early stopping  |
| Test | 10 000 | nguyên bản MNIST  | Đo cuối, không chỉnh model |

- Ảnh gốc: **28×28** pixel, xám (0–255)
- 10 lớp: chữ số `0` … `9`
- Số mẫu **theo lớp** (không đều tuyệt đối):

| Lớp  | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Fit  | 4738 | 5394 | 4766 | 4905 | 4674 | 4337 | 4734 | 5012 | 4681 | 4759 |
| Val  | 1185 | 1348 | 1192 | 1226 | 1168 | 1084 | 1184 | 1253 | 1170 | 1190 |
| Test | 980  | 1135 | 1032 | 1010 | 982  | 892  | 958  | 1028 | 974  | 1009 |



## Cách nhóm xử lý.

```text
MNIST / canvas → nhị phân + bỏ nhiễu → crop & căn giữa
  → resize 28×28 → scale 0..1 → flatten → ANN
```

- **28×28**: dùng đúng kích thước gốc của MNIST để hạn chế mất chi tiết nét chữ trước khi train ANN.
- **Thang mực 0→1**: dùng float để giữ lại mức xám sau area-average resize.
- **Augment nhẹ khi train**: dịch / xoay / nét → tổng quát hơn với nét tay thật

## Vì sao dùng MNIST?

Chuẩn, 10 lớp rõ, đủ lớn để học mà vẫn demo được với ANN flatten.

# Data Flow

> **MNIST 28×28 → chuẩn hóa, xóa vùng trống, resize 28×28 → train ANN → Predict nét vẽ tay.**
