# numgo roadmap：以 Go 风格实现 numpy

> 目标：实现 numpy（基准 2.6.0.dev0，见 [numpy-modules.md](./numpy-modules.md)）的全部数值能力，但按 Go 的习惯设计——不是逐函数翻译。
> 库尚无使用者，可随时做破坏性演进；每期以「与 numpy 对拍通过」为验收。

## 全局设计决策（先于一切分期）

这些决策一旦做出贯穿全库，必须在 Phase 0 落定：

1. **泛型即 dtype 分派**。numpy 用运行时 dtype + 代码生成展开运算矩阵；Go 用 `Array[T]` 编译期单态化。
   - 数值约束逐步放宽：`Float`（现状）→ `Number`（+复数 complex64/128，Go 内建运算符支持）→ 按需整数只读运算。
   - **不做**运行时动态 dtype 容器（那是 Python 的需求）；object/void(结构化)/StringDType 暂缓到有真实需求。
   - `longdouble/clongdouble` 降级为 float64/complex128；`float16` 软件实现，排后期。
2. **语义对拍 numpy**：广播规则、类型提升、NaN/Inf 传播、轴语义（axis 归约后该维消失/保留为 1 由 `keepdims` 显式控制）逐条对齐。错误消息沿用 numpy 措辞（`operands could not be broadcast together...`、`Singular matrix`）+ Go panic（非法输入=编程错误；IEEE 特例如除零走 Inf/NaN 不抛）。
3. **命名**：保留 numpy 名字首字母大写（`Arange/Tensordot/NanMean`——Go 导出规则），不用 Go 重命名发明新词汇；概念对齐降低迁移心智成本，这就是 Go 生态的惯例（如 gonum 对标 BLAS 命名）。
4. **重型后端用现成库**：LAPACK → `gonum.org/v1/gonum/lapack`（纯 Go）；FFT → `gonum.org/v1/gonum/dsp/fourier`；随机 → 自实现 BitGenerator（math/rand/v2 无法 bit-exact 对齐 numpy，PCG64/MT19937 算法公开且不长）。符合「不重复造轮子」。
5. **测试策略：golden 对拍**。每期用一个 Python 脚本从 numpy 生成 fixture（输入+期望输出 JSON/npy），Go 侧读入断言。numpy 源码在本地可随时生成。

## Phase 0 — 架构基线（1 期）

**范围**：决定 Phase 1 的地基，代码量小、决策量大。

- [ ] N 维核心类型定型：`Array[T]` 内部从 `r,c,data` 改为 `shape []int + strides []int + data []T`（视图/零拷贝切片的前提）；2D 便捷方法保留。
- [ ] 复数约束扩展：`type Number interface { Float | ~complex64 | ~complex128 }`。
- [ ] 错误语义基线文档化（panic 消息表、IEEE 特例表）。
- [ ] golden 对拍框架：`scripts/gen_fixtures.py`（调 numpy 产 fixture）+ Go 侧 fixture 加载断言 helper。
- [ ] 现有 2D API 盘点标记：哪些保留为便捷层（如 `New2DArray`），哪些在 N 维核心落地后改为薄包装。

**验收**：决策记录在本文件更新；对拍框架能跑通一个 hello-fixture。

## Phase 1 — `_core`：N 维数组核心（最大的一期，可再拆）

numpy 对应：`_core`。这是全路线图的地基，其他一切都是组合层。

- [ ] shape/strides/视图：切片返回视图（零拷贝）、`Reshape` 视图化、`Broadcast`（完整 numpy 广播规则，N 维对齐）、`Transpose/SwapAxes/ExpandDims/Squeeze/Ravel/Flatten`。
- [ ] 索引：整数/切片/步长/负索引/`Take/Put`；fancy 索引与布尔掩码索引。
- [ ] 迭代器：`Iter`（对齐 nditer 的多数组协同迭代：广播 + 缓冲 + 步进），所有运算的执行引擎。
- [ ] ufunc 层：一元/二元运算统一经 `zip`（已有雏形）+ `where`/`out` 参数；`Reduce/Accumulate/Outer`。
- [ ] reduction：`Sum/Max/Min/Mean/Std/Var/ArgMax/ArgMin/Any/All/CumSum`，支持 `axis int/[]int + keepdims`。
- [ ] 创建函数族：`Arange/Linspace/Logspace/Zeros/Ones/Empty/Full/Eye/Identity/FromFunction/Meshgrid`。
- [ ] 形状操作：`Concatenate/Stack/Split/Tile/Repeat/Resize/Flip/Roll`。
- [ ] 排序：`Sort/Argsort/Partition/ArgPartition/Unique`（先 `slices`/`sort` 后 SIMD 优化）。
- [ ] `Dot/Matmul/Einsum`（einsum 可后置到本期末尾）。
- [ ] type promotion 最小集：Go 静态类型下以「显式转换函数 + 运算前手动提升」为 Go 习惯，提供 `AsType[To, From]`；不模拟隐式提升。

**验收**：golden fixture 覆盖广播/索引/reduction/排序 ~200 例；现有 2D 测试全绿。

## Phase 2 — 三大数值后端：linalg / fft / random

numpy 对应：`linalg`、`fft`、`random`。相互独立，可三车道并行。

**linalg**（后端 gonum/lapack，纯 Go）：
- [ ] `Solve/Inv/Det/Slogdet/Cholesky/QR(Lstsq)/SVD/Eig/Eigh/Norm/Cond/MatrixRank/MultiDot/TensorSolve/TensorInv/MatrixPower`
- [ ] 批处理矩阵栈（numpy gufunc 语义：`(…,M,K)@(…,K,N)`）——依赖 Phase 1 广播。
- [ ] 具名结果类型（`SVDResult` 等）→ Go 习惯是具名 struct，直接对齐。

**fft**（后端 gonum/dsp/fourier 或自实现 pocketfft 移植）：
- [ ] `FFT/IFFT/RFFT/IRFFT/HFFT/IHFFT` + 2D/nD 族 + `FFTFreq/FFTShift`；`norm` 三模式。

**random**（自实现，对齐 bit-exact）：
- [ ] `BitGenerator` 接口 + PCG64/MT19937 先行（Philox/SFC64 后补）；`SeedSequence`（熵池 + spawn，并行流可复现是 numpy random 的灵魂，优先级最高）。
- [ ] `Generator`：`Random/Integers/Normal/Uniform/Choice/Shuffle/Permutation/Standard*` + 常用分布（Beta/Binomial/Exponential/Gamma/Poisson/Chisquare…），分布算法对照 numpy 的 C 源逐个移植。
- [ ] legacy RandomState **不做**（np.random 全局单例是 Python 遗产，Go 习惯是显式 Generator 注入）。

**验收**：linalg 对拍 LAPACK 例程 ~40 例；fft 精度对拍 1e-12；random 同种子序列逐位一致。

## Phase 3 — lib 工具层 + polynomial（纯组合层，快速推进）

numpy 对应：`lib`（第二梯队全部）、`polynomial`。核心稳定后机械推进，可多车道并行。

- [ ] IO：`Load/Save/SaveZ/SaveTxt/LoadTxt/FromRegex` + `.npy` 格式（header 协议自实现，格式公开）+ `OpenMemmap`（Go 用 mmap 包）。
- [ ] 形状/索引工具：`AsStrided/SlidingWindowView/BroadcastArrays/RavelMultiIndex/Mgrid/Ogrid/NdEnumerate`。
- [ ] 集合运算：`Unique(全变体)/IsIn/Intersect1D/Setdiff1D/Union1D/Ediff1D`。
- [ ] NaN 族：`NanSum/NanMean/NanMedian/NanStd/NanVar/NanArgMax/...`。
- [ ] 统计：`Percentile/Quantile/Cov/CorrCoef/Histogram/Histogram2D/HistogramDD/Gradient/Interp/Piecewise`。
- [ ] `Pad`（全 mode）/`TwodimBase`（Diag/Tri*）/`TypeCheck`（NanToNum 等）/`Emath`（复数安全数学，依赖 Phase 0 复数）。
- [ ] polynomial：6 个类共享同一泛型基类接口（numpy 72 方法 API 面的 Go 版）；`PolyFit/PolyVal/PolyDer/PolyInt/Roots/FromRoots/Vander/Companion/Convert`。

**验收**：对拍 fixture 覆盖每子模块代表性函数；`.npy` 文件与 numpy 双向读写一致。

## Phase 4 — 数组变体（按需启动，真需求驱动）

numpy 对应：`ma`、`strings`/`char`、`rec`。

- [ ] `strings`：向量化字符串操作（upper/split/replace/count…46 个），Go 的 `strings` 包逐函数映射到数组上，实现简单。
- [ ] `ma`：MaskedArray = `data Array[T] + mask Array[bool]` + 掩码传播运算包装。API 大（232 导出），仅在 Phase 1-3 完成且有需求后启动。
- [ ] `rec`/结构化 dtype：Go 习惯是 `[]struct`，倾向**不做** numpy 式 void dtype，提供 `Array[T] ↔ []struct` 转换即可（在 roadmap review 时再决策）。
- [ ] `matrixlib`、`char`（弃用 shim）、`emath` 已并入 scimath：不做/已覆盖。

## Phase 5 — 工程配套

- [ ] `testing`：`AssertArrayEqual/AssertAllClose/AssertArrayAlmostEq` 等 17 断言的 Go 版（`*testing.T` 风格）。
- [ ] `typing`：Go 泛型约束即文档；导出 `NDArray`/`ArrayLike` 别名与 constraint 集，写 API 规范文档。
- [ ] 互操作：`__array_interface__`/DLPack 协议的 Go 侧等价物——先做 `UnsafeData() unsafe.Pointer` + cgo 互操作示例（ctypeslib 对应），协议化互操作后置。
- [ ] 性能基线：SIMD（Go assembly 或 lattice 生成）对标 numpy 的 x86-simd-sort/loops——**最后做**，先正确后快速。

## 明确不做（及理由）

| numpy 模块 | 理由 |
|---|---|
| `f2py` | Fortran 工具链，Python 专属 |
| `doc` | 纯文档 |
| `matrixlib` | numpy 自己都标注不推荐；Go 有固定 2D Array 即可 |
| `char`（shim） | 已被 strings 取代 |
| array_api 兼容层 | Python 生态协议 |
| RandomState/np.random 全局单例 | Python 遗产，违背 Go 显式依赖习惯 |
| 隐式 dtype 提升 | 静态类型语言不做运行时类型提升；显式 `AsType` |
| longdouble/clongdouble | Go 无原生 80/128-bit 浮点，降级并文档标注 |

## 里程碑依赖图

```
Phase 0 ──→ Phase 1 (_core) ──┬─→ Phase 2 (linalg ∥ fft ∥ random)
                              ├─→ Phase 3 (lib ∥ polynomial)
                              └─→ Phase 4 (strings → ma)
                                        ↘ Phase 5 (贯穿，随做随补)
```

Phase 2 与 Phase 3 可并行；Phase 1 未稳定前不动 Phase 2/3。
