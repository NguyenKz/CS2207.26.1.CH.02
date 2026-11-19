# Các bước của Simplicial SMOTE

Nguồn: [Simplicial SMOTE (arXiv:2503.03418)](../papers/2503.03418v1.pdf). Số trang trong link là số trang PDF. Code chính thức: `CK/simplical-smote/ssmote/smote.py`.

## Đầu vào

- Tập điểm lớp thiểu số `X⁺` (`n⁺` điểm), lớp đa số có `n⁻` điểm, với `n⁺ < n⁻`.
- Tham số `k` (kích thước láng giềng) và `p` (kích thước nhóm tối đa, xem ghi chú bên dưới).
- Số điểm cần sinh: `m = n⁻ − n⁺`.

Ghi chú về `p`: paper dùng `p` là chiều (simplex có `p + 1` đỉnh). Code chính thức dùng `p` là số đỉnh (simplex có `p` điểm). Hai cách khác nhau một đơn vị.

Định nghĩa `n⁺`, `n⁻`, `m`, `k`, `p` xem [trang 4](../papers/2503.03418v1.pdf#page=4). Grid search cho `k`, `p` xem [trang 6](../papers/2503.03418v1.pdf#page=6).

## Bước 0: Chuẩn hóa dữ liệu (tiền xử lý, không thuộc Algorithm 1)

- Đưa mỗi feature về trung bình 0, phương sai 1 (z-score), với `μ` và `σ` tính trên tập train của fold.
- Paper nêu bước này trong phần thực nghiệm, không nằm trong Algorithm 1 ([trang 6](../papers/2503.03418v1.pdf#page=6)).
- Code chính thức đặt `StandardScaler` trước sampler trong pipeline, nên chỉ fit trên tập train của từng fold (`experiment.py`).

## Bước 1: Dựng đồ thị k-NN

- Với mỗi điểm trong `X⁺`, tìm `k` láng giềng gần nhất theo khoảng cách Euclid (mặc định của `kneighbors_graph`).
- Nối cạnh `(x, x')` nếu một điểm nằm trong `k` láng giềng gần nhất của điểm kia (đối xứng, `A + Aᵀ > 0`).
- Kết quả là đồ thị `G_k(X⁺)`.

Xem [trang 4](../papers/2503.03418v1.pdf#page=4) (định nghĩa `R_kNN`, Algorithm 1 dòng 1).

## Bước 2: Tìm clique

- Một tập điểm đôi một có cạnh nối tạo thành một simplex.
- Paper: lấy `p`-skeleton của clique complex, tức các simplex có chiều tối đa `p`, rồi lấy các simplex tối đại `Σ^MAX_p`.
- Code chính thức: `nx.find_cliques(G)` lấy clique tối đại. Sau đó với mỗi clique, lấy các tập con có đúng `p` đỉnh.

Xem [trang 4](../papers/2503.03418v1.pdf#page=4) (Algorithm 1 dòng 2, mục "Sampling from a simplicial complex").

## Bước 3: Chọn simplex

- Lặp `m` lần, mỗi lần chọn một simplex có hoàn lại.
- Paper: chọn đều ngẫu nhiên từ `Σ^MAX_p`.
- Code: không đều. Xác suất chọn một clique tỉ lệ với số clique cùng kích thước và `C(kích thước, p)`. Sau đó chọn đều một tập con `p` đỉnh từ clique đó.

Xem [trang 4](../papers/2503.03418v1.pdf#page=4) (Algorithm 1 dòng 3) cho paper. Phần code xem `SimplicalOversampler._get_p_simplices` trong `smote.py`.

## Bước 4: Sinh điểm bằng tọa độ barycentric

Với simplex có `d` đỉnh `x₀, …, x_{d−1}`:

1. Lấy `λ ~ Dir(α)` với `α = (1, …, 1)`. Điều này cho `λⱼ ≥ 0` và `Σλⱼ = 1`.
2. Tính điểm mới: `x̂ = Σⱼ λⱼ xⱼ = λᵀX`.

Xem [trang 4](../papers/2503.03418v1.pdf#page=4) (công thức (3), (4) và Algorithm 1 dòng 4 đến 7). Code chính thức dùng `np.random.default_rng(...).dirichlet` và `np.einsum`.

## Bước 5: Kết quả

- Gom `m` điểm `x̂` lại thành tập synthetic `X̂⁺`.
- Huấn luyện bộ phân loại trên tập đã cân bằng.

Xem [trang 4](../papers/2503.03418v1.pdf#page=4) (Algorithm 1 dòng 7).

## Ví dụ số

Dữ liệu: A(0,0), B(3,0), C(1,4), D(4,4), E(8,1). `k = 2`, `p = 2` theo quy ước paper (tam giác là `p = 2`), `m = 3`.

- Láng giềng: A→{B, C}, B→{A, D}, C→{D, A}, D→{C, B}, E→{D, B}.
- Cạnh: AB, AC, BD, CD, DE, BE.
- Simplex tối đại: {A,B}, {A,C}, {C,D}, {B,D,E}.
- Giả sử chọn: {B,D,E}, {A,B}, {C,D}.
- `λ = (0.5, 0.3, 0.2)` cho BDE → `(4.3, 1.4)`.
- `λ = (0.7, 0.3)` cho AB → `(0.9, 0)`.
- `λ = (0.4, 0.6)` cho CD → `(2.8, 4)`.

`λ` trong ví dụ là giá trị chọn tay để minh họa, không sinh bằng RNG. Ví dụ này dùng cách chọn đều của paper, không phải trọng số của code.

## Vị trí trong paper

| Nội dung | Trang |
|---|---|
| Ý tưởng, so sánh SMOTE và simplicial | [trang 2](../papers/2503.03418v1.pdf#page=2) |
| Hình minh họa khoảng cách, phần related work | [trang 3](../papers/2503.03418v1.pdf#page=3) |
| Định nghĩa đồ thị k-NN, `p`-ary relation | [trang 3](../papers/2503.03418v1.pdf#page=3), [trang 4](../papers/2503.03418v1.pdf#page=4) |
| Algorithm 1, lấy mẫu simplex, barycentric | [trang 4](../papers/2503.03418v1.pdf#page=4) |
| Biến thể Borderline, Safe-level, ADASYN | [trang 5](../papers/2503.03418v1.pdf#page=5) |
| Độ phức tạp | [trang 5](../papers/2503.03418v1.pdf#page=5) |
| Kết quả synthetic và grid search `k`, `p` | [trang 5](../papers/2503.03418v1.pdf#page=5), [trang 6](../papers/2503.03418v1.pdf#page=6) |
| Kết quả k-NN và Gradient Boosting | [trang 7](../papers/2503.03418v1.pdf#page=7), [trang 8](../papers/2503.03418v1.pdf#page=8) |
