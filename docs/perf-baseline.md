# numgo 性能基线（Phase 5 优化前）

> 本报告是 **Phase 5 优化前的基线（before-optimization baseline）**，用于与后续优化对比。
> 所有 numgo 数字来自 `*_bench_test.go`（`-benchtime 1s -count 3`，取中位数）；numpy 数字来自
> `scripts/bench_numpy.py`（best of 5）。比值 = numgo (ms) / numpy (ms)，>1 表示 numgo 更慢。

## 环境

| 项 | 值 |
|---|---|
| CPU | AMD Ryzen 9 7950X 16-Core（32 线程，bench GOMAXPROCS=32；单线程跑分） |
| Go | go1.27.0 linux/amd64，零依赖纯 Go 内核 |
| numpy | 2.5.1（OpenBLAS 0.3.33 后端，scipy-openblas wheel） |
| 方差 | `-count 3` 三轮中位数；除 MatmulBatch（±12%，大量分配触发 GC）与 Irfft65536（单次迭代，±3%）外，各基准轮间波动 < 5% |

## 对比表

| 操作 | 规模 | numgo ns/op | numgo allocs/op | numpy ms | 比值 (numgo/numpy) |
|---|---|---:|---:|---:|---:|
| Add 连续 | 1000×1000 | 2 258 914 | 10 (8.0 MB) | 0.404 | **5.6×** |
| Add 广播行 | 1000×1000 + 1×1000 | 9 214 333 | 10 (8.0 MB) | 0.262 | **35.2×** |
| Add 跨步 (A + Aᵀ) | 1000×1000 | 1 953 112 | 10 (8.0 MB) | 0.650 | **3.0×** |
| Sum 全归约 | 1000×1000 | 2 268 182 | 6 | 0.109 | **20.8×** |
| Sum axis=0 | 1000×1000 | 3 708 087 | 5 | 0.126 | **29.4×** || CumSum 末轴 | 1000×1000 | 3 222 162 | 6 (8.0 MB) | 2.262 | **1.4×** |
| Sort 末轴 | 1000×1000 | 28 624 696 | 12 008 (41 MB) | 0.296 | **96.6×** |
| Unique | 100k 1-d | 29 589 544 | 7 | 2.817 | **10.5×** |
| Matmul | 200×200×200 | 114 835 542 | 14 | 0.062 | **1852×** |
| Matmul 批量 | 16×64×64×64 | 265 899 909 | 16 908 334 (271 MB) | 0.103 | **2581×** |
| Einsum ij,jk->ik | 100×100×100 | 15 730 113 | 69 | 0.173 | **90.9×** |
| At/Set 逐元素 | 1e6 次 | 9 062 050 | 0 | 116.068 | 0.078×（快 12.8×） |
| Slice 视图 | 500×500 步长2 | 43.9 | 3 | ~0 | —（同数量级） |
| Reshape | →10000×100 | 40.5 | 3 | ~0 | —（同数量级） |
| AsType f64→f32 | 1e6 | 6 340 696 | 6 (8.0 MB) | 0.050 | **126.8×** |
| Solve | 100×100 | 302 747 | 25 | 2.724 | 0.11×（快 9×） |
| Inv | 100×100 | 2 050 412 | 15 | 5.057 | 0.41×（快 2.5×） |
| SVD | 100×100 | 35 841 746 | 27 | 52.987 | 0.68×（快 1.5×） |
| Eig | 50×50 | 10 554 972 | 8 243 (14 MB) | 0.592 | **17.8×** |
| Cholesky | 100×100 | 270 915 | 12 | 0.030 | **9.0×** |
| FFT | 2^16 | 53 685 379 | 6 159 (23.6 MB) | 0.756 | **71.0×** |
| FFT 混合基 | 2^13·5 | 27 508 997 | 6 159 (14.8 MB) | 0.440 | **62.5×** |
| FFTN | 64×64 | 4 125 262 | 668 | 0.030 | **137.5×** |
| IRFFT | 2^16 | 1 063 676 644 | 138 181 (374 MB) | 2.666 | **399.0×** |
| StandardNormal | 1e6 | 4 531 760 | 4 | 6.569 | 0.69× |
| Random | 1e6 | 2 572 312 | 4 | 1.560 | **1.65×** |
| Integers | 1e6 | 4 337 951 | 2 | 1.737 | **2.50×** |
| ma.Add 半掩码 | 1000×1000 | 77 106 962 | 3 500 013 (65 MB) | 0.564 | **136.7×** |

读取指引：linalg 大部分与 BLAS/LAPACK 打平或更快（numpy 侧 Solve/Inv/SVD 在小矩阵上被
OpenBLAS 线程池与调用开销拖累）；真正的差距集中在**逐元素/批量循环**与 **FFT**。

## Phase 5 优化后（2026-10-24，Matmul 内核 + nditer.Zip 广播行快路径）

> 本节是第一批 Phase 5 优化的 after 数字，环境与测量方式同上表（`-benchtime 1s -count 3`
> 取中位数）。只覆盖本批改到的操作；其余行以上表为准。
> 累加顺序变体：Matmul 内核采用**保持 k 升序累加**的 4×8 寄存器分块（每个输出元素的
> 求和顺序与朴素 i-j-k 循环逐位一致，bit-identical），未采用会改变顺序的分块重排。

| 操作 | 规模 | before ns/op | after ns/op | after allocs/op | 比值 (after/numpy) | 目标 |
|---|---|---:|---:|---:|---:|---|
| Matmul | 200×200×200 | 114 835 542 | **1 910 338** | 15 | **30.8×** | ≤50× 达标 |
| Matmul 批量 | 16×64×64×64 | 265 899 909 | **1 046 127** | 24（原 16 908 334） | **10.2×** | ≤100× 达标 |
| Add 广播行 | 1000×1000 + 1×1000 | 9 214 333 | **2 088 774** | 12 | **8.0×** | ≤8× 达标 |
| Add 连续 | 1000×1000 | 2 258 914 | 2 229 484（未动路径） | 10 | 5.5× | — |
| Add 跨步 (A + Aᵀ) | 1000×1000 | 1 953 112 | **2 089 870**（经新行路径） | 10 | 3.2× | — |
| Sum 全归约 | 1000×1000 | 2 268 182 | 2 283 740（未动路径） | 6 | 21.0× | — |
| Sum axis=0 | 1000×1000 | 3 708 087 | 3 746 316（未动路径） | 5 | 29.7× | — |

要点：

- **Matmul**：`dot.go` 重写为 dense 原生切片内核 `matmulKernel`（4×8 寄存器分块，i-k-j
  面板 FMA）。批量版分配从 1690 万/op 降到 23/op。非连续输入回退 `Copy`（O(n) 对
  O(n³) 可忽略）；1-D 提升路径经 dense 缓冲 + `CopyInto` 折叠回输出形状。
- **Add 广播行 / 跨步**：`nditer.Zip` 增加「dst 连续 + 各 src 末轴 stride 0 或单位
  步长」的行级快路径——每行一次回调（单位步长 src 零拷贝切片，stride-0 src 复用
  预填充缓冲），消掉每元素回调与 odometer。跨步用例（Aᵀ 的末轴 stride=N）也从
  回调内联循环受益。回调块长度契约不变（块长仍为 UNSPECIFIED）。
- **Sum axis=0**：走 `nditer.Reduce`（非 Zip），无末轴 stride-0 形态，本批未动
  （热点分析第 5 条的行分块建议留待后续批次）。

## 热点分析（Top 5）

### 1. Matmul 批量（2581×）/ Matmul 200（1852×）
**现象**：批量版 1690 万次分配（271 MB/op）；单次 200³ 也要 114 ms（BLAS 约 62 µs）。
**原因**：`dot.go` 的 Matmul 对**每个输出元素**调用 `At`/`Set`，每次都
`append([]int(nil), bc...)` 复制坐标切片再走 bounds-check 索引；16 个 batch × 64×64 输出
× 每 K 步两次 At = 千万级切片分配 + 装箱。加上无分块（blocked kernel）、无 SIMD，
三重乘积循环是 O(n³) 标量访问。这是纯 Go 与 BLAS 的数量级差距。
**优化建议**：批内把 batch 平面 `At` 一次取出为 `[]T` 行主序缓冲（每 batch 一次分配，
表复用），内层改为对 `aRow[k]·bCol[k]` 的扁平切片点积；进一步做 8×8 寄存器分块 +
4 路展开（float64 下 Go 编译器可自动向量化部分 FMA）。目标：消除 per-element 坐标路径。

### 2. IRFFT 65536（399×）与 FFT 族（62–137×）
**现象**：IRFFT 单次 1.06 s、3.7 亿字节分配（13.8 万次 alloc）；FFT 2^16 每次 6160 次分配。
**原因**：`fftimpl.go`/`fft.go` 每条变换线（每行）都新分配 twiddle 表、工作缓冲与中间
complex128 数组；2^16 = 1 条 64k 线尚可，但 IRFFT 内部走了 padding + 多轮中间数组，
且小因子 radix 路径递归分段复制。每调用 6 千次 alloc 说明缓冲完全没有复用。
**优化建议**：包级 twiddle 因子表缓存（按 n 复用，`sync`/包级 map）；`iterBatch` 逐线
循环提升到线外一次性分配共享工作区；IRFFT 把 half-spectrum → real 的搬运改为原地。
预期能把 allocs 压到个位数、时间降一个数量级（仍难追 pocketfft 的 SIMD，但 399×→~10×）。

### 3. ma.Add 半掩码（137×）
**现象**：1000×1000 加法 77 ms、350 万次分配。
**原因**：`ma/ops.go` 的 `zip2` 走 `walk`（坐标递归），对**每个元素**分配坐标切片并做
`valueAt`/`maskAtB` 全坐标索引（无连续快路径）；结果数组 + 掩码逐元素构造。
**优化建议**：掩码为纯切片时数据部分直接调根包 `Add`（连续 memcpy 级循环），掩码单独
`or`/`and` 逐字节合并；`walk` 仅用于非连续布局。

### 4. AsType f64→f32（127×）与 Sum/Sort/Unique（10–97×）
**现象**：100 万元素类型转换 6.3 ms（numpy 50 µs）；Sort 末轴 28.6 ms 且 41 MB 分配。
**原因**：`convert.go` 的 AsType 经 `parts`/`scalarFromParts` 逐元素泛型标量转换（复数
拆装开销落在 float→float 上）；reduce 走 nditer 回调每元素一次函数调用（无分块 SIMD
归约）；`sort.go` 对每行独立 `slices.Sort` 且 `Sort` 先整块 `Copy`（41 MB），numpy 用
SIMD 化的 introsort。
**优化建议**：AsType 对实数类型加专用快速循环（`dst[i] = T(src[i])`，可被编译器向量化）；
Reduce 已有连续快路径，但回调 fn 每元素调用——改为分块 `Reduce` 让 fn 收整块（nditer
接口已支持）；Sort 连续布局时直接对行切片排序、跳过 Copy。

### 5. Add 广播行（35×）/ SumAxis0（29×）等非连续引擎路径
**现象**：行广播加法 9.2 ms（连续版 2.3 ms 的 4 倍，numpy 0.26 ms）；axis=0 归约 3.7 ms。
**原因**：广播操作数未命中 nditer 连续快路径，退化为逐元素回调 + stride 步进；axis=0
归约按列跳跃访问（cache 不友好）且每元素一次闭包调用。
**优化建议**：广播行（stride=0 的尾维展开）特判为「内层连续 memcpy 循环」；axis=0 归约
先做行分块（一次读 8 行累加到行缓冲，减少跳跃）；中期给 nditer 增加分块缓冲（对齐
numpy nditer 的 buffering 能力）。

## 结论

- 已达标：linalg 直接法（Solve/Inv/Cholesky 甚至快于 numpy 的小矩阵路径）、随机数
  （PCG64 流水线相当）、视图/Reshape（纳秒级）、At/Set（Go 每元素调用仍比解释器快 12.8×）。
- 优化主战场（Phase 5 依次攻克）：**Matmul 分块内核** → **FFT 缓冲复用** → **ma 连续
  快路径** → **AsType/Sort 专用循环** → **广播/axis0 分块**。全部为纯 Go 改造，
  不引入 cgo/BLAS 依赖的前提下，预期把最大比值从 2581× 压到两位数以内。

## Phase 5 优化后（2026-10-24，FFT 缓存 / AsType / Sort / ma.Add 快路径）

> `-benchtime 1s -count 3` 取中位数，同机同 numpy 数字；变化列 = 优化前 → 优化后。
> 所有 13 组 numpy 黄金测试与随机位级对拍测试原样通过。

| 操作 | ns/op 变化 | allocs/op 变化 | 比值变化 |
|---|---|---|---|
| FFT 2^16 | 53.7 ms → **5.2 ms** (9.9×) | 6 159 → **16** | 71.0× → **6.9×** |
| FFT 混合基 2^13·5 | 27.5 ms → **2.8 ms** (9.8×) | 6 159 → **16** | 62.5× → **6.3×** |
| FFTN 64×64 | 4.13 ms → **0.32 ms** (12.7×) | 668 → **30** | 137.5× → **10.7×** |
| IRFFT 2^16 | 1 063.7 ms → **55.7 ms** (19.1×) | 138 181 → **193**（374 MB → 9.6 MB） | 399.0× → **20.9×** |
| AsType f64→f32 | 6.8 ms → **0.33 ms** (20.7×) | 6 → **4** | 126.8× → **6.6×** |
| Sort 末轴 | 31.7 ms → **2.9 ms** (11.0×) | 12 008 → **5**（41 MB → 8 MB） | 96.6× → **9.7×** |
| ma.Add 半掩码 | 77.1 ms → **3.1 ms** (24.9×) | 3 500 013 → **14**（65 MB → 9 MB） | 136.7× → **5.6×** |

实现要点（均为纯 Go，无新依赖）：

- **FFT**：twiddle 表按 (n, 方向) 进包级 `sync.Map`（同一 `cis(sign·k)` 表达式，位级
  不变）；Bluestein chirp 与卷积核谱缓存、fa/sc 工作区按尺寸进 `sync.Pool`；递归工作
  缓冲从每次顶层调用一个 2n 刮擦区切出（`fftScratch` 贯穿 `fullAxis`/`halfAxis*` 批处理，
  IRFFT 逐行只复用 coeff/full/res 四个缓冲）。IRFFT 1-D 走无 At/Set 直通路径；conjAll
  连续时一条循环共轭。
- **AsType**：`convertTyped` 一次性类型分发 + 16 组内建类型对的内联拷贝循环（f64→f32
  可被编译器向量化）；命名类型回退原 reflect 路径。Vdot 同样一次性分发（累加顺序不变）。
- **Sort**：C 连续 + 末轴时逐行就地 `slices.SortFunc`（f32/f64 用具体类型比较器，
  NaN 置于末尾与 `cmpNumber` 同序），不再整块 Copy + 逐通道收集。
- **ma.Add/Sub/Mul/Div**：两侧数据连续且形状等于广播结果时，数据部分平铺循环、掩码
  逐位 OR，跳过 walk 坐标递归；IEEE/NaN 行为与原路径一致（ma 黄金测试全绿）。

未达标项：AsType 6.6×（目标 ≤5×）——剩余时间为输出分配 + 4 MB 写入的内存带宽下限，
进一步需复用输出缓冲（超出本次 Pure 语义范围）。

## Phase 5 优化后（2026-10-25，Einsum matmul 折叠 / nditer axis-0 连续快路径 / FFT 缓存上限）

> `-benchtime 1s -count 3` 取中位数，同机同 numpy 数字。全部测试原样通过
> （`go test -count=1 ./...` 全绿），race 下 fft/nditer/根包全绿。

| 操作 | 规模 | before ns/op | after ns/op | after allocs/op | 比值 (after/numpy) | 目标 |
|---|---|---:|---:|---:|---:|---|
| Einsum ij,jk->ik | 100×100×100 | 15 730 113（基线）／7.3 ms（当前分支起点） | **272 258** | 93（8.4 MB → 0.33 MB） | **1.6×** | ≤40× 达标 |
| Sum axis=0 | 1000×1000 | 3 708 087（基线）／3.5 ms（起点） | **1 338 198** | 5 | **10.6×** | ≤8× 未达标 |
| FFT 2^16 | 2^16 | 5.2 ms | 5.1 ms（缓存上限未触发） | 16 | 6.7× | 不劣化 达标 |
| FFT 混合基 | 2^13·5 | 2.8 ms | 2.8 ms | 16 | 6.3× | 不劣化 达标 |
| FFTN 64×64 | 64×64 | 0.32 ms | 0.32 ms | 30 | 10.5× | 不劣化 达标 |
| IRFFT 2^16 | 2^16 | 55.7 ms | 55.6 ms | 193 | 20.9× | 不劣化 达标 |

实现要点（均为纯 Go，无新依赖）：

- **Einsum matmul 折叠**（`einsum_matmul.go`，`einsumMatmul`/`einMatmulPair`）：≥2 操作数时
  逐对左折叠为 dense matmul——共享且不出现在输出的标签按升序摊平成 K，左剩余输出标签为
  M、右剩余输出标签为 N，走 `dot.go` 的 `matmulKernel`。K 的摊平顺序（升序）与通用路径的
  求和顺序一致，结果逐位相同（随机对拍测试 `TestEinsumMatmulPathRandom` 断言 bit-identical）。
  每对操作数先按轴序 permute、广播摊平 batch（ellipsis）、`Copy` 成 C 连续块再进内核。
  模式不匹配即回退原 transpose-broadcast-multiply-sum 路径：共享且又在输出的标签
  （Hadamard 混合）、纯外积、输出含 ellipsis、隐式 ellipsis 输出、成对后需再求和的标签、
  自由标签 size-1 广播。>2 操作数为贪心左折叠（先 1·2，结果与 3 继续；numpy
  optimize=greedy 的简化版，文档注明）。
- **nditer axis-0 连续快路径**：`Reduce` 在输入 C 连续且仅折轴 0 时，先按行序把第 0 行
  折进 identity 预填的 out（与通用 C 序遍历逐位同序），随后每行一次线性扫描
  （`out[j] ⊕= data[base+j]`），消掉逐坐标 odometer。fn 契约不变。
- **FFT 缓存上限**：包级 `cacheEntries` 计数（`sync.Mutex` 保护）覆盖 twiddle 表、
  Bluestein 核、工作池三处 `sync.Map`；超过 `maxCacheEntries = 256` 条目时整体清空重建
  （简单全清逐出，非 LRU——少尺寸负载永不触发；多条目负载付出整窗重建代价）。
  白盒测试 `TestCacheBoundEvicts` 直填 map 验证逐出与恢复。

未达标项：SumAxis0 10.6×（目标 ≤8×）——剩余为 8 MB 读带宽下限 + 5 次分配；进一步需要
多行分块合并到一次扫描（收益递减，留待后续批次）。

## 整体对照：numgo vs numpy（2026-10-26，stdlib 数学切换后）

> 一键复现：`sh scripts/bench_compare.sh`（Go 侧 `go test -bench` `-count 3` 取中位数，
> numpy 侧 `scripts/bench_numpy.py` best-of-5；比值 = numgo ms / numpy ms，>1 = numgo 慢）。
> 本节全部数字为同日同机一次跑出的配套测量（Ryzen 9 7950X，numpy 2.5.1 / OpenBLAS，
> numpy matmul/linalg 走 BLAS 多线程、numgo 单线程纯 Go）。numpy 侧数字随机器负载波动
> 明显（如 SVD 上次记录 53 ms、本次 1.2 ms），linalg 行比值仅供定性参考；本节数值均为
> 当时实测快照。

| 操作 | 规模 | numgo | numpy | 比值 |
|---|---|---:|---:|---:|
| Add 连续 | 1000×1000 | 2.19 ms | 0.204 | **10.7×** |
| Add 广播行 | 1000×1000+1×1000 | 1.91 ms | 0.258 | **7.4×** |
| Add 跨步 (A+Aᵀ) | 1000×1000 | 1.84 ms | 0.692 | 2.7× |
| Sum 全归约 | 1000×1000 | 2.28 ms | 0.107 | **21.3×** |
| Sum axis=0 | 1000×1000 | 1.35 ms | 0.125 | **10.8×** |
| CumSum 末轴 | 1000×1000 | 2.82 ms | 2.302 | 1.2× |
| Sort 末轴 | 1000×1000 | 2.78 ms | 0.324 | **8.6×** |
| Unique | 100k 1-d | 29.3 ms | 2.882 | **10.2×** |
| Matmul | 200³ | 1.91 ms | 0.092 | **20.8×** |
| Matmul 批量 | 16×64×64×64 | 1.02 ms | 0.116 | **8.8×** |
| Einsum ij,jk→ik | 100³ | 0.29 ms | 0.168 | 1.7× |
| At/Set 逐元素 | 1e6 | 9.38 ms | 114.0 | 0.082×（快 12×） |
| Slice 视图 / Reshape | — | ~50 ns | ~0 | 同量级 |
| AsType f64→f32 | 1e6 | 0.56 ms | 0.094 | **6.0×** |
| Solve | 100×100 | 0.24 ms | 5.730 | 0.042×（快 24×） |
| Inv | 100×100 | 1.98 ms | 9.114 | 0.22×（快 4.6×） |
| SVD | 100×100 | 35.8 ms | 1.214 | **29.5×**（波动大，见上） |
| Eig | 50×50 | 10.9 ms | 0.583 | **18.6×** |
| Cholesky | 100×100 | 0.22 ms | 0.030 | 7.3× |
| FFT | 2^16 | 5.18 ms | 0.878 | **5.9×** |
| FFT 混合基 | 2^13·5 | 2.91 ms | 0.449 | **6.5×** |
| FFTN | 64×64 | 0.32 ms | 0.029 | **11.0×** |
| IRFFT | 2^16 | 56.7 ms | 3.741 | **15.2×** |
| StandardNormal | 1e6 | 4.53 ms | 6.563 | 0.69×（快） |
| Random | 1e6 | 2.72 ms | 1.545 | 1.8× |
| Integers | 1e6 | 4.17 ms | 1.649 | 2.5× |
| **StandardGamma** | 1e6 | 11.1 ms | 12.093 | **0.92×（打平）** |
| **Weibull** | 1e6 | 41.0 ms | 11.037 | **3.7×** |
| ma.Add 半掩码 | 1000×1000 | 2.59 ms | 0.888 | 2.9× |

要点：

- **libm 切换的实际影响面**：StandardNormal/Random/Integers（libm 无关）与切换前完全
  一致；StandardGamma 1.5× 退化后仍与 numpy 打平（0.92×）；Weibull（每抽一次 `math.Pow`）
  是唯一显著退化项（约 1.4× → 3.7×）。若后续成为热点，替换为 ARM optimized-routines
  （MIT）表驱动 pow/exp 内核即可回到切换前水平。
- **仍快于 numpy**：小矩阵 Solve/Inv（BLAS 线程池启动开销）、StandardNormal（ziggurat）、
  At/Set（纯 Go 索引远快于解释器）。
- **主要差距**（宽表逐元素/归约 10–21×、Sort 8.6×、FFT 族 6–15×）均为无 SIMD 的纯 Go
  循环 vs numpy 向量化内核，属结构性的，Phase 5 各批次已把其中最痛点从数量级差距压到
  个位数/十几倍。

## 2026-10-26 SIMD 探针（P0/P1，GOEXPERIMENT=simd）

Go 1.27 实验 `simd` 包（GOEXPERIMENT=simd 门控）接入两条热点，构建标签双实现：
`internal/simdvec`（`simd.go` / `nosimd.go`，API 同名），默认构建零改动。

落地内容：

- **Add 连续 float64 快路径**（P0）：`ufunc.go` `Add` 在三数组同形 C 连续时走
  `simdvec.AddF64`。默认构建借快路径重构 2.19ms → ~1.1ms；GOEXPERIMENT=simd
  AVX-512 全向量 Load/Add/Store + Part 尾处理，~1.0ms。
- **Matmul FMA 内核**（P1）：`dot.go` `matmulKernel` 的 float64 分支走
  `simdvec.MatmulKernelF64`。simd 构建：8 通道跨 N（每通道一个输出列），k 升序
  `MulAdd`（探测证实 ≡ 逐元素 `math.FMA` 位级）；einsum 折叠与批量 matmul 经共享
  内核自动继承。**语义注记**：GOEXPERIMENT=simd 构建为 FMA 累加语义，与默认构建
  （普通 mul+add，恢复 pre-P1 原始内核，位级与速度均不变）在一般 float64 输入上
  末位 ULP 可能不同——与 numpy 跨 BLAS 后端行为同类；库内契约（einsum 折叠路径与
  通用路径 bit-identical）在两种构建下各自成立。
- **AsType f64→f32**：Go 1.27 simd 包暂无 Float64s→Float32s 转换指令（仅
  Int32s.ConvertToFloat32 / 位重解释），两构建均保持标量循环，无数值变化。

基准（7950X，count=2，pre-P1 基线 → 默认 → GOEXPERIMENT=simd）：

| 基准 | 基线 | 默认 | GOEXPERIMENT=simd |
|---|---|---|---|
| UfuncAddContiguous 1000×1000 | 2.19ms | ~1.1ms | ~1.0ms |
| Matmul200 | 1.91ms | ~1.9ms（不变） | ~0.45ms（≈4.2×） |
| MatmulBatch 16×64×64 | 1.02ms | ~1.0ms（不变） | ~0.26ms（≈4×） |
| Einsum 100×100 | 0.29ms | ~0.29ms（不变） | ~0.12ms（≈2.4×） |
| Sum 全归约 1000×1000 | 2.28ms | ~2.24ms（不变） | **~0.072ms（≈31×，反超 numpy 0.107ms）** |
| Sum axis=0 1000×1000 | 1.35ms | ~1.35ms（不变） | **~0.088ms（≈15×，反超 numpy 0.125ms）** |
| AsType f64→f32 1e6 | 0.56ms | ~0.5ms | ~0.5ms |

P2（Sum 归约）：`SumF64` 车道累加 + 顺序 lane 合并（pairwise 类顺序，末位 ULP 语义注记
同 matmul FMA）；`SumAxis0F64` 车道跨列、每列行序累加——跨构建 **bit-identical**（测试
证明）。simd Sum 仅覆盖纯 Sum：Mean/Var/Std/Nan*/Prod/Max/Min 不走 simd 路径（Var 的
(Σx²−(Σx)²/n) 要求两个求和同序，避免灾难性消元；经 `reduced` 显式参数传递，非包级
哨兵——后者在 `t.Parallel()` 下有数据竞争，已移除）。SumAll/SumAxis0 的 2.28/1.35ms
基线是闭包回调遍历而非带宽极限，simd 后双双反超 numpy。

CI 新增 `test-goexperiment-simd` 任务（gofmt/vet/test 与主任务镜像，GOEXPERIMENT=simd）。
