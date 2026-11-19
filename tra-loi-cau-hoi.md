# Trả lời câu hỏi — Logistic Regression & MNIST

Tóm tắt theo đúng pipeline trong project (`train_digit_model.py`, `mnist-dataset.md`).

---

## 1. Train logistic regression như thế nào? Chỉ số `C` là gì?

### Cách train (trong project)

1. Load ảnh MNIST đã chuẩn hóa (28×28 → flatten thành 784 feature).
2. Chia tập train gốc thành **fit / validation** (80/20, stratified).
3. (Tuỳ chọn) augment chỉ trên fit set.
4. Fit `StandardScaler` trên dữ liệu fit, rồi scale fit / val / test.
5. **Tune `C`**: thử nhiều giá trị `C` trên tập con, chọn `C` cho validation accuracy cao nhất.
6. Train lại `LogisticRegression(C=best_C, solver="lbfgs", ...)` trên toàn bộ fit set.
7. Đánh giá cuối trên **test set** (không dùng để chọn `C`).

Code chính:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=C, max_iter=300, solver="lbfgs", random_state=42)
model.fit(fit_features, training_labels)
```

Các ứng viên `C` trong project:

```python
# np.logspace(-4, 2, 30)  → 30 giá trị từ 0.0001 đến 100
LOGISTIC_CANDIDATES = np.logspace(-4, 2, 30)
```

### `C` là gì?

`C` là **hệ số inverse regularization** của scikit-learn `LogisticRegression`:

| `C` | Ý nghĩa |
| --- | --- |
| **Nhỏ** | Regularization mạnh → model đơn giản hơn, ít overfit hơn |
| **Lớn** | Regularization yếu → model bám train nhiều hơn |

Công thức trực giác: `C ≈ 1 / λ` (λ = độ mạnh regularization).

Trong project, `C` được **tune trên validation**, không cố định tay một số.

---

## 2. Raw dataset có phải ảnh RGB 28×28 không?

**Không.**

- Raw MNIST là ảnh **xám (grayscale)**, kích thước **28×28**.
- Mỗi pixel: mực từ 0–255 (sau này scale về 0..1).
- **Không phải RGB** (không có 3 kênh màu).
- Mỗi ảnh → **784 feature** (= 28 × 28 × 1).

Nguồn: MNIST của Yann LeCun — chữ số viết tay 0–9.

---

## 3. Tỷ lệ dataset train / val / test?

MNIST gốc **không có validation**. Project chia như sau:

| Tập | Số mẫu | Cách lấy | Vai trò |
| --- | ---: | --- | --- |
| **Fit (train)** | 48 000 | 80% của train gốc (60 000) | Học trọng số |
| **Val** | 12 000 | 20% của train gốc | Tune `C` / theo dõi |
| **Test** | 10 000 | Giữ nguyên test MNIST | Đánh giá cuối |

Tóm tắt tỷ lệ:

- Từ **train gốc**: **80% fit / 20% val** (seed cố định, stratified).
- **Test**: giữ nguyên, **không lấy từ train**.

Nếu nói theo tổng ~70 000 mẫu: khoảng **68.6% / 17.1% / 14.3%** (fit / val / test).

---

## 4. Hàm loss là gì? Vì sao dùng Cross-entropy?

Đây là bài **phân loại nhiều lớp** (multi-class classification): ảnh chữ số → nhãn `0`–`9` (10 lớp).

**Loss** đo mức độ sai giữa nhãn thật và dự đoán. Train = tìm trọng số làm loss giảm.

Loss dùng là **Categorical Cross-entropy** (multi-class log-loss) — **không phải** Binary Cross-entropy.

| Loại | Khi nào dùng |
| --- | --- |
| **Binary CE** | 2 lớp (hoặc multi-label: nhiều nhãn độc lập, thường sigmoid) |
| **Categorical CE** | Nhiều lớp, chọn **một** nhãn — như MNIST 0–9 |

### Công thức (categorical / multi-class)

```text
L = - (1/n) * Σ_i Σ_k y_ik * log(ŷ_ik)
```

- `y_ik`: one-hot nhãn thật (chỉ 1 lớp đúng = 1, còn lại = 0).
- `ŷ_ik`: xác suất model gán cho lớp `k` (sau softmax / logistic).
- Vì one-hot, thực tế chỉ còn: `L = - (1/n) * Σ_i log(ŷ_{i, lớp_đúng})`  
  → chỉ quan tâm xác suất của **lớp đúng**.

Ví dụ một mẫu nhãn = `3`:

| Dự đoán `ŷ` của lớp 3 | Loss ≈ `-log(ŷ)` |
| ---: | ---: |
| 0.90 | nhỏ (~0.11) — đúng và khá chắc |
| 0.50 | vừa (~0.69) |
| 0.10 | lớn (~2.3) — sai / không chắc lớp đúng |
| 0.01 | rất lớn (~4.6) — sai mà còn “tự tin” sai |

→ Sai càng tự tin, phạt càng nặng.

### Vì sao dùng Cross-entropy (không dùng MSE)?

1. **Phù hợp output xác suất**  
   Softmax / logistic cho ra phân phối xác suất trên các lớp. Cross-entropy đo “khoảng cách” giữa phân phối thật (one-hot) và phân phối dự đoán.

2. **Phạt đúng kiểu lỗi phân loại**  
   Không chỉ “đúng/sai nhãn”, mà còn phạt khi model gán xác suất thấp cho lớp đúng. Accuracy không làm được việc đó trong lúc train (không trơn, khó tối ưu).

3. **Gradient tốt khi train**  
   Kết hợp softmax + cross-entropy cho gradient đơn giản dạng `ŷ - y`.  
   MSE trên xác suất thường học chậm hơn ở vùng bão hòa (sigmoid/softmax gần 0 hoặc 1).

4. **Chuẩn cho classification**  
   Logistic Regression và ANN phân loại nhiều lớp trong project đều tối ưu log-loss / cross-entropy.

### Trong project

- ANN tự viết: `SimpleANN.cross_entropy_loss(...)`.
- Logistic Regression (sklearn): tối ưu **log-loss / cross-entropy** (+ regularization theo `C`).

**Lưu ý:** Loss giảm ≠ luôn accuracy tăng; loss là tín hiệu học, accuracy là metric báo cáo.

---

## 5. Regularization là gì?

**Regularization** là kỹ thuật **giới hạn độ phức tạp của model** để giảm overfitting — không chỉ fit tốt train set mà còn dự đoán tốt trên dữ liệu mới.

Cách làm phổ biến: thêm **phạt** vào loss khi trọng số quá lớn:

```text
Loss_tổng = Loss_gốc + λ * (độ lớn trọng số)
```

- `λ` lớn → phạt mạnh → trọng số nhỏ hơn → model “đơn giản” hơn
- Trong sklearn Logistic Regression: `C ≈ 1/λ` → **C nhỏ = regularization mạnh**

Hai dạng thường gặp: **L2** (Ridge, phạt `w²`) và **L1** (Lasso, phạt `|w|`, có thể đưa một số trọng số về 0).

---

## 6. Tập nào thực sự train? Tập nào chỉ chạy để lấy kết quả?

| Tập | Có cập nhật trọng số? | Dùng để làm gì? |
| --- | --- | --- |
| **Fit (train)** | Có — đây là tập **thực sự train** | Học trọng số / bias |
| **Validation** | Không | Chọn `C`, theo dõi, early stopping |
| **Test** | Không | Chỉ chạy predict để **lấy kết quả cuối** |

Tóm lại:

- **Train thật** = **Fit set** (48 000, có thể kèm augment).
- **Val** và **Test** chỉ forward / score, **không học** từ chúng.
- Báo cáo độ chính xác cuối cùng → dùng **Test**.

---

## 7. Tập Val được dùng như thế nào?

**Validation** không cập nhật trọng số của model cuối. Nó dùng để **chọn cấu hình** và **theo dõi** khi train.

Trong project MNIST:

| Việc | Có dùng Val 12k? |
| --- | --- |
| Tune `C` (Logistic) | Có — chọn `C` theo validation accuracy |
| Báo validation accuracy sau train | Có |
| Fit `StandardScaler` | Không — scaler học từ Fit, rồi transform Val |
| `model.fit` trọng số cuối | Không — chỉ học trên Fit |
| Báo cáo điểm cuối | Không — dùng **Test** |

Một câu: **Val = thi thử** (chọn hyperparameter / theo dõi); **Test = thi thật** một lần cuối.

### Val có value không?

Có — value **khác** Fit/Test:

| Tập | Value mang lại |
| --- | --- |
| **Fit** | Học trọng số |
| **Val** | Chọn cấu hình đúng, phát hiện overfit, quyết định dừng |
| **Test** | Ước lượng độ tốt trên dữ liệu chưa đụng tới |

Không có Val → dễ tune trên Test (điểm lạc quan giả) hoặc không biết đang overfit.

### “Chọn cấu hình đúng” xảy ra khi nào?

**Trong lúc phát triển / trước khi báo cáo Test** — sau khi có model ứng viên trên Fit, trước khi chốt kết quả cuối.

| Thời điểm | Việc làm trên Val |
| --- | --- |
| Tune `C` (Logistic) | Train từng `C` trên Fit → chấm Val → chọn `C` tốt nhất → train lại trên Fit → mới chạy Test |
| Early stopping / theo dõi MLP | Mỗi epoch nhìn Val (hoặc val nội bộ) để dừng / giữ weights tốt |
| So sánh model | So logistic vs MLP… theo Val; Test chỉ để công bố |

**Không** chọn cấu hình khi đang `fit` trọng số trên Fit, và **không** thử nhiều lần trên Test rồi chọn điểm đẹp nhất.

**Một câu:** Chọn cấu hình = **tune / so sánh trên Val**, trước khi dùng Test để công bố điểm.

---

## 8. Early stopping đang dùng là gì?

**Early stopping** = dừng (hoặc giữ bộ trọng số tốt nhất) khi model **không còn cải thiện trên validation**, để tránh train quá lâu / overfit.

### MNIST MLP (`compact_sigmoid` / `tanh` / `relu`)

Config: `"early_stopping": true` → `sklearn.MLPClassifier`.

Cách sklearn làm:

- Tự tách **~10%** dữ liệu đang `fit` làm validation **nội bộ** (không phải Val 12k của project).
- Theo dõi điểm trên validation đó.
- Kết thúc: **giữ bộ trọng số tốt nhất** theo validation nội bộ.

Tham số trong code:

- `max_iter = 700`
- `tol = 0.0`
- `n_iter_no_change = 700` (bằng `max_iter` khi không set riêng)

→ Thực tế gần như **không cắt epoch sớm**; lợi ích chính là **restore best weights**.

**Logistic không dùng early stopping** (solver `lbfgs`).

### Demo Iris trên web (`server.py`)

Early stopping viết tay:

- Mỗi epoch đo **validation loss** trên Val của Iris.
- Nếu loss không giảm đủ (`min_delta = 0.001`) trong **`patience = 40`** epoch → dừng.

**Nhớ:** Early stopping nhìn Val (hoặc validation nội bộ) — **không dùng Test**.

---

## 9. Tối ưu bằng gì? MLP sklearn (Adam) dùng ở đâu?

Không phải mọi model đều dùng gradient descent thuần:

| Model | Tối ưu | Ghi chú |
| --- | --- | --- |
| **ANN tự viết (Iris / NumPy)** | Gradient descent + backprop | `θ ← θ − lr × gradient` |
| **MLP MNIST (sklearn)** | **Adam** (`solver='adam'`) | Biến thể SGD (adaptive) |
| **Logistic MNIST (sklearn)** | **L-BFGS** (`solver='lbfgs'`) | Quasi-Newton, không phải GD thuần |

### MLP sklearn + Adam dùng cho chỗ nào?

Dùng cho **nhận dạng chữ số MNIST** — các model ANN trong web/backend:

| Model ID | Vai trò |
| --- | --- |
| `compact_sigmoid` | MLP 784→196→98→10, hidden **sigmoid** |
| `compact_tanh` | MLP cùng kiến trúc, hidden **tanh** |
| `compact_relu` | MLP cùng kiến trúc, hidden **relu** (thường là model chính) |

Train qua `train_digit_model.py` → `build_mlp(...)` → `MLPClassifier(solver='adam', ...)`.

**Không dùng** Adam cho:
- Demo Iris ANN tự viết (**NumPy** + GD)
- Baseline logistic (**L-BFGS**)

### Epoch và batch size (MLP Adam)

Cấu hình: `batch_size = 256`, `max_iter = 700` (= tối đa 700 epoch; sklearn không có tên `epochs`).

```text
số batch / epoch ≈ số mẫu train / batch_size
```

1 epoch = đi hết 1 vòng dữ liệu train, mỗi bước Adam dùng 1 batch 256 mẫu.
