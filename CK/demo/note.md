# Note: Simplicial SMOTE

Nguồn: Kachan, Savchenko, Gusev (KDD '25), arXiv:2503.03418. Code chính thức: [av-savchenko/simplical-smote](https://github.com/av-savchenko/simplical-smote), clone tại `CK/simplical-smote/`.

## Ý tưởng

- SMOTE sinh điểm mới trên **đoạn thẳng** nối một điểm thiểu số với 1 hàng xóm (1-simplex).
- Simplicial SMOTE sinh điểm mới trên **simplex** (đoạn, tam giác, tứ diện...) được tạo bởi nhiều điểm thiểu số đôi một gần nhau.

## Thuật toán (Algorithm 1)

1. Dựng đồ thị k-NN `G_k(X+)` trên các điểm thiểu số. Cạnh `(x, x')` tồn tại nếu một điểm nằm trong k láng giềng gần nhất của điểm kia (đối xứng).
2. Lấy clique complex, tức các nhóm điểm mà mọi cặp đều có cạnh nối. Paper dùng `p`-skeleton. Code chính thức chỉ lấy clique tối đại (`nx.find_cliques`), rồi lấy các tập con kích thước `p`.
3. Cần sinh `m = n⁻ − n⁺` điểm. Paper: mỗi lần chọn đều một simplex tối đại, có hoàn lại. Code: không đều, xác suất tỉ lệ với số clique cùng kích thước và `C(kích thước, p)`.
4. Lấy `λ ~ Dir(1, ..., 1)` trên simplex, rồi tính `x̂ = Σ λᵢ xᵢ = λᵀX`.

Trước bước 1, có tiền xử lý không thuộc Algorithm 1: chuẩn hóa z-score ([trang 6](../papers/2503.03418v1.pdf#page=6)).

- **Fold:** một phần trong cross-validation. Với 5-fold, dữ liệu chia 5 phần, mỗi vòng một phần làm test và 4 phần còn lại làm train.
- **Tập train của fold:** 4 phần train trong vòng đó. `μ` và `σ` của chuẩn hóa chỉ tính trên tập này, rồi áp dụng cho tập test. Tính trên toàn bộ dữ liệu sẽ làm rò rỉ thông tin của test vào train.
- Paper dùng cross-validation lồng nhau: vòng ngoài 5 repeat × 5 fold để đánh giá, vòng trong 25 repeat × 4 fold để chọn siêu tham số ([trang 6](../papers/2503.03418v1.pdf#page=6)).
- Code chính thức dùng `RepeatedStratifiedKFold` ở cả hai vòng và đặt `StandardScaler` trong pipeline, nên scaler được fit lại trên từng tập train (`experiment.py`). Đây là chi tiết code, không phải của paper.

## Ký hiệu

- **`n⁺`** (n-plus): số điểm lớp **thiểu số** (positive). Đây là tập đầu vào của SMOTE / Simplicial SMOTE.
- **`n⁻`** (n-minus): số điểm lớp **đa số** (negative). Luôn có `n⁺ < n⁻` trong bài toán mất cân bằng này.
- **`m`**: số điểm tổng hợp cần sinh, `m = n⁻ − n⁺`, để sau oversample hai lớp bằng nhau về số lượng.
- Nhớ nhanh: **`+` = thiểu số cần “bơm thêm”**, **`-` = đa số làm mốc số lượng**.
- **`k`**: số láng giềng gần nhất dùng để dựng đồ thị k-NN. Hai điểm có cạnh nối nếu một điểm nằm trong `k` láng giềng gần nhất của điểm kia. `k` quyết định kích thước vùng lân cận. Paper grid search `k` với bước 2 trong một khoảng phụ thuộc dataset.
- **`p`**: kích thước tối đa của nhóm dùng để nội suy. Có hai cách hiểu:
  - **Theo paper (văn bản):** `p` là chiều tối đa, một `p`-simplex có `p + 1` đỉnh.
  - **Theo code chính thức:** `p` là số đỉnh, mỗi simplex lấy đúng `p` điểm (hoặc ít hơn nếu clique nhỏ hơn `p`). `experiment.py` grid search `p` trong `range(3, k+1)`, tức 3 đến `k` đỉnh.
  - Lưu ý: Algorithm 1 trong paper ghi `p ≥ k`, còn phần thực nghiệm ghi `p` từ 3 đến `k`. Code khớp với cách ghi thứ hai, nên cần đọc paper cẩn thận khi báo cáo.
- Trong ví dụ số, `m = 3` là giả định để minh họa, không được tính từ `n⁻ − n⁺`.

## Vì sao demo hay hiện 3 đỉnh, không phải 4–5–6 hay `n`?

- **Không bắt buộc phải là 3.** Số đỉnh là siêu tham số `p` (theo quy ước code). Phương pháp tổng quát cho mọi kích thước simplex:
  - 2 đỉnh → đoạn thẳng (= SMOTE cổ điển)
  - 3 đỉnh → tam giác (2-simplex)
  - 4 đỉnh → tứ diện (3-simplex)
  - `n` đỉnh → `(n−1)`-simplex
- Paper/code **grid search** `p` từ 3 đến `k`, không khóa một giá trị ([trang 6](../papers/2503.03418v1.pdf#page=6); `experiment.py`).
- `p` nhỏ quá ≈ gần SMOTE, phủ ít. `p` lớn quá cần clique lớn trong đồ thị k-NN → ít simplex đủ điều kiện, mẫu kém ổn định. Mindmap ghi “p nhỏ hoặc lớn quá đều kém” ([mindmap.md](mindmap.md), từ phần thảo luận paper).
- Trần thực tế: trong đồ thị k-NN, không có clique lớn hơn `k+1` điểm đôi một là hàng xóm, nên `p` (số đỉnh) không vượt quá khoảng đó.
- Demo canvas hay vẽ **tam giác** vì dữ liệu 2D: 3 đỉnh là simplex đầy đủ dễ nhìn trong mặt phẳng. Đây là lựa chọn minh họa của demo, không phải định lý “phải 3” của paper.

## Ví dụ số

Dữ liệu: A(0,0), B(3,0), C(1,4), D(4,4), E(8,1). Cấu hình `k = 2`, `p = 2` (theo quy ước paper: tam giác là `p = 2`), `m = 3`.

Láng giềng gần nhất (theo khoảng cách Euclid, so sánh bằng `d²`):

- A: B, C
- B: A, D
- C: D, A
- D: C, B
- E: D, B

Cạnh: AB, AC, BD, CD, DE, BE.

Simplex tối đại: {A,B}, {A,C}, {C,D}, {B,D,E}. Chỉ {B,D,E} là tam giác.

Mẫu minh họa (λ chọn tay, không từ RNG):

- Tam giác BDE, λ = (0.5, 0.3, 0.2): x̂ = (4.3, 1.4). Điểm nằm trong tam giác, SMOTE không sinh được.
- Đoạn AB, λ = (0.7, 0.3): x̂ = (0.9, 0).
- Đoạn CD, λ = (0.4, 0.6): x̂ = (2.8, 4).

## Độ đo khoảng cách

- **Euclid (L2).** Xác nhận trong code chính thức: `SimplicialSMOTE.fit_resample` và `SMOTE.fit_resample` trong `CK/simplical-smote/ssmote/smote.py` gọi `kneighbors_graph(X[y==1], n_neighbors=k)` không truyền `metric`, nên dùng mặc định Minkowski `p=2`, tức Euclid.
- Paper không ghi rõ công thức, chỉ nói chung là `d(x, y)`.
- Paper chuẩn hóa mỗi dataset về trung bình 0, phương sai 1 ([trang 6](../papers/2503.03418v1.pdf#page=6)). Code dùng `StandardScaler` trong `experiment.py`, đặt trong pipeline trước sampler.
- Công thức Euclid trong note là kiến thức nền, không trích từ paper hay tài liệu cụ thể.

## Dữ liệu nhiều chiều

- Mỗi điểm là vector `x ∈ ℝᵈ`, `d` là số feature ([trang 4](../papers/2503.03418v1.pdf#page=4)). Khoảng cách Euclid dùng cả `d` tọa độ.
- Độ phức tạp của bước tìm clique phụ thuộc mật độ đồ thị (`k`), không phụ thuộc số chiều ([trang 5](../papers/2503.03418v1.pdf#page=5)).
- Paper vẫn dùng dataset có `d` lớn, như `scene` (294 feature) và `optical_digits` (64), không giảm chiều ([bảng 4, trang 9](../papers/2503.03418v1.pdf#page=9)).
- Chưa có trong paper: cách xử lý điện tim (tín hiệu thô hay đặc trưng trích từ tín hiệu), căn chỉnh nhịp, PCA trước khi sinh điểm. Đây là suy luận hoặc ý kiến riêng, cần ghi rõ khi báo cáo.

## Nguồn và độ tin cậy

| Ý trong note | Nguồn | Trang |
|---|---|---|
| Định nghĩa `d` là hàm tổng quát trên `X × X` | Paper | [3](../papers/2503.03418v1.pdf#page=3) |
| `x_i ∈ ℝᵈ`, định nghĩa `n⁺`, `n⁻`, `m`, `k`, `p` | Paper | [4](../papers/2503.03418v1.pdf#page=4) |
| Grid search `k` với bước 2, `p` từ 3 đến `k` | Paper | [6](../papers/2503.03418v1.pdf#page=6) |
| Không cố định 3 đỉnh; `p` là siêu tham số, simplex tổng quát `n` đỉnh | Paper + code | [4](../papers/2503.03418v1.pdf#page=4), [6](../papers/2503.03418v1.pdf#page=6) |
| Demo hay vẽ tam giác vì dữ liệu 2D dễ nhìn | Demo / suy luận trình bày | Không thuộc paper |
| Chuẩn hóa về trung bình 0, phương sai 1 | Paper | [6](../papers/2503.03418v1.pdf#page=6) |
| Độ phức tạp không phụ thuộc số chiều | Paper | [5](../papers/2503.03418v1.pdf#page=5) |
| Dataset có `d` lớn (`scene`, `optical_digits`) | Paper, bảng 4 | [9](../papers/2503.03418v1.pdf#page=9) |
| Sampler chọn đều simplex tối đại | Paper, Algorithm 1 | [4](../papers/2503.03418v1.pdf#page=4) |
| Code chọn simplex theo trọng số `C(kích thước, p)` | Code: `ssmote/smote.py`, `_get_p_simplices` | Không thuộc paper |
| `p` là số đỉnh trong code | Code: `ssmote/smote.py`, `experiment.py` | Không thuộc paper |
| Euclid là mặc định của `kneighbors_graph` | Code: `ssmote/smote.py`, không truyền `metric` | Không thuộc paper |
| Pipeline đặt scaler trước sampler | Code: `experiment.py` | Không thuộc paper |
| Chuẩn hóa là tiền xử lý, không thuộc Algorithm 1 | Paper, phần thực nghiệm | [6](../papers/2503.03418v1.pdf#page=6) |
| Công thức Euclid | Kiến thức nền | Không có |
| MinMaxScaler, RobustScaler | Kiến thức nền | Không có |
| Curse of dimensionality | Kiến thức nền | Không có |
| Căn chỉnh nhịp điện tim, PCA trước khi sinh điểm | Suy luận hoặc ý kiến riêng | Không có |
| Ví dụ số 5 điểm A–E | Tự tính, không có trong paper | Không có |

## Demo hiện tại (`CK/demo/`)

- **Nguồn dataset demo:** cả 3 bộ đều **synthetic** trong `backend/server.py` (không lấy từ paper/bank):
  - Fraud: `Normal` quanh gốc (đa số) + 2 cụm `Normal` (thiểu số).
  - Moons: hai cung tròn, góc `Uniform[0,π]`, bán kính nhiễu `Normal`.
  - Ring: đĩa trong `r∝√U` (đa số) + vành ngoài góc `Uniform[0,2π]` (thiểu số).
  Landing hiện trường `howto` trên từng card.
- **Landing (`index.html`):** hiện kịch bản demo trước khi chọn dataset:
  1. Tạo mẫu với `k = 5`, `p = 3`, `seed = 2207`, `m = n⁻ − n⁺` (có chú thích `n⁺` = thiểu số, `n⁻` = đa số + số cụ thể theo dataset).
  2. Train `LogisticRegression` trên train đã cân bằng.
  3. Đo trên test: recall, precision, F1, FN, FP.
  4. So sánh **3 method**: Original, SMOTE, Simplicial.
- **Làm gì trên canvas:** tạo mẫu (`original`, `smote`, `simplicial`), train `LogisticRegression` trên dữ liệu đã tăng cường, rồi tính recall, precision, F1 và số FN, FP trên tập test (`backend/server.py`, hàm `run_sampler`, `train_model`, `metrics`). Random oversampling đã bỏ khỏi demo vì step animation không khớp (chỉ copy điểm).
- **Step “Nối neighborhood” trên canvas:** chỉ biểu diễn (vẽ cạnh + hiện `d`). Khoảng cách thật đã dùng khi dựng k-NN/pool trước đó; step này không tính thêm gì cho thuật toán.
- **Dữ liệu:** một dataset gian lận tổng hợp do `make_fraud_data` sinh ra, không phải benchmark của paper.
- **Không có:** 21 dataset, k-NN, Gradient Boosting, cross-validation, MCC, kiểm định thống kê, và các biến thể Borderline, Safe-level, ADASYN.
- **Animation λ:** demo tự lấy `eⱼ ~ Exp(1)`, `λⱼ = eⱼ/Σe`, rồi nội suy điểm trên timeline từ `λ` đó để hiện phép tính. Điểm synthetic dùng để train/F1 vẫn đến từ sampler chính thức — có thể khác điểm trên animation.
- **Lỗi còn lại:** với SMOTE, mapping parent có thể lệch (`parent_rows[index % len]`). Không dùng animation làm bằng chứng tuyệt đối cho từng sample của sampler.

## Lưu ý khi làm đồ án

- Chạy scale và sinh điểm chỉ trên tập train của từng fold, tránh rò rỉ dữ liệu.
- Kiểm tra lại: HDF5 dataset có đi kèm source không, phiên bản scikit-learn có tương thích không.
- Khi báo cáo, nêu rõ `p` theo quy ước nào (số chiều hay số đỉnh), vì paper và code không thống nhất.
