# numpy 模块清单

> 基准：本地 numpy 源码 `/home/pana/go/src/github.com/numpy/numpy/`（2.6.0.dev0，commit ef046f8）。
> 本文是事实盘点，作为 [roadmap.md](./roadmap.md) 的输入。

## 总体规模

| 项 | 数值 |
|---|---|
| 顶层公开 API（`numpy.__all__`） | **498 个名字** |
| C/C++ 核心代码 | ~154k 行 + 33.7k 行 `*.c.src` 代码生成模板 |
| Cython | ~13.9k 行（**仅 random** 模块） |
| Python 层（非测试） | ~125k 行（大量是 docstring 与参数校验壳） |
| dtype 注册 | 24 个传统 dtype + StringDType + 用户扩展位 |

结构性结论：numpy 的 Python 层本质是「API 协议 + 校验壳」，真正算法在 C；只有 random 用 Cython。Go 移植只需复刻**语义**，无需复刻构建体系。numpy 用代码生成展开 dtype × 运算矩阵（132 个 ufunc × 238 个签名组）；Go 用**泛型**在编译期达到同样效果——这正是 Go 风格的正解。

## 模块清单

### 第一梯队：数值内核

| 模块 | 功能面 | 规模 | 实现依赖 |
|---|---|---|---|
| `_core` | ndarray（N 维、shape/strides/视图）、广播、索引（basic/slice/fancy/bool）、dtype 系统与类型提升、ufunc 机制、迭代器 nditer、排序（含 SIMD）、einsum/matmul、标量层级 | 238 def / 32 class；ndarray 方法 72；132 个 ufunc | 纯 C + 代码生成 |
| `linalg` | solve/inv/eig/eigh/eigvals/svd/qr/cholesky/det/slogdet/pinv/lstsq/norm/cond/matrix_rank/multi_dot/tensordot/matmul + 具名结果类型（EigResult 等），批处理矩阵栈 | 33 def / 6 class | BLAS/LAPACK（自带 lapack_lite 兜底，51 个 routine） |
| `fft` | fft/ifft/rfft/hfft 及 2D/nD 族 + fftfreq/fftshift | 14 def | vendored pocketfft (C++) |
| `random` | Generator（45 方法：36 分布 + integers/choice/shuffle/spawn）+ 5 个 BitGenerator（MT19937/PCG64/PCG64DXSM/Philox/SFC64）+ SeedSequence（可复现熵派生、并行流）+ legacy RandomState（49 方法） | 5 BitGenerator + 94 采样方法 | Cython + C 分布库 |
| `polynomial` | 6 个多项式类（Polynomial/Chebyshev/Legendre/Hermite/HermiteE/Laguerre）共享同一 72 方法 API 面：算术/der/int/roots/fromroots/fit/vander/companion/convert | 6 class / 164 def | 纯 Python（依赖核心运算） |

### 第二梯队：算法工具层（纯 Python，依赖核心）

| 模块（lib 子模块） | 功能 | 公开量 |
|---|---|---|
| npyio / format | load/save/savez/savetxt/loadtxt/genfromtxt、`.npy` 二进制格式、open_memmap、NpzFile | 22 |
| shape_base | stack/split/hsplit/vsplit/apply_along_axis/expand_dims/tile/kron | 14 |
| stride_tricks | as_strided、sliding_window_view、broadcast_arrays | 5 |
| twodim_base | diag/eye/tri/triu/tril/vander/histogram2d | 15 |
| function_base | percentile/quantile/median/interp/gradient/cov/corrcoef/vectorize/meshgrid/窗函数 | 35 |
| arraysetops | unique/isin/intersect1d/setdiff1d/union1d/ediff1d | 11 |
| arraypad | pad（constant/edge/reflect/wrap 等 mode） | 1 |
| nanfunctions | nansum/nanmean/nanmedian/nanstd/nanarg*/nanpercentile | 14 |
| histograms | histogram/histogramdd/histogram_bin_edges | 3 |
| index_tricks | c_/r_/mgrid/ogrid/ix_/ndenumerate/ravel_multi_index | 13 |
| type_check | iscomplexobj/nan_to_num/real/imag/common_type | 11 |
| _polynomial | legacy poly1d + poly/roots/polyfit | 11 |
| scimath/emath | 复数域安全 sqrt/log/…（自动升级复数） | 9 |
| mixins | NDArrayOperatorsMixin（运算符协议） | 1 class |
| recfunctions | 结构化数组字段操作（append_fields/join_by…） | 21 |
| arrayterator / datasource / introspect | 分块迭代、文件抽象、SIMD 查询 | 少量 |

### 第三梯队：数组变体与互操作

| 模块 | 功能面 | 公开量 |
|---|---|---|
| `ma` | MaskedArray（data+mask、掩码传播运算、domain 检查）、extras、mrecords | 232 |
| `char` / `strings` | 向量化字符串操作（upper/split/find/replace/…）；char 是弃用 shim，strings 是 2.x 现行 API（StringDType） | 53 / 46 |
| `rec` | recarray（字段属性访问）、format_parser | 9 |
| `matrixlib` | 遗留 matrix（`*`=矩阵乘）、bmat（文档标注不推荐） | 3 |
| `ctypeslib` | ndpointer/as_ctypes/as_array（零拷贝互操作） | 6 |
| `dtypes` / `exceptions` | 41 个 DType 类命名空间；7 个异常类 | 42 / 7 |
| `emath` / `matlib` | 复数安全数学；MATLAB 风格构造器 | 9 / 8 |

### 第四梯队：工程配套

| 模块 | 功能面 | 公开量 |
|---|---|---|
| `testing` | 17 个 assert* 数组断言 + 工具 | 45 def |
| `typing` | NDArray[Shape, DType]、ArrayLike 等（58k 行 stub） | 9 |
| `f2py` | Fortran→Python 接口生成器（**Python 专属工具链**） | 259 def |
| `doc` | 纯文档，无代码 | 0 |
| array_api | 2.0 已移除，改为 `__array_namespace_info__`（外部 array-api-compat 承接） | — |

### dtype 全集（`_core`）

`bool, int8/16/32/64, longlong, uint8/16/32/64, ulonglong, float16, float32, float64, longdouble, complex64, complex128, clongdouble, object, bytes, str, void(结构化), datetime64, timedelta64` + 2.x `StringDType`（UTF-8 可变长）。

Go 侧硬约束：`longdouble/clongdouble`（80/128-bit）无原生对应——按 float64/complex128 降级并文档标注；`float16` 需软件实现（无硬件转换指令时）。

## 对 Go 移植最重要的 5 个发现

1. **代码生成 → 泛型**：numpy 用 33.7k 行 `.c.src` 模板展开 dtype×运算；Go 泛型单态化天然对齐，无需代码生成器。
2. **执行引擎是难点**：`nditer` + strided loops（任意 strides/buffering/casting 迭代）是一切 ufunc 与 reduction 的地基，成本最高。
3. **dtype 提升规则**是语义核心：`can_cast_table`、类型提升、NaN/整数溢出语义必须与 numpy 一致，否则对拍全挂。
4. **纯 Python 层移植快**：lib/polynomial/ma/char/rec 全是组合层，核心稳定后机械推进。
5. **可直接裁剪**：f2py/doc/matrixlib/char(弃用 shim)/array_api 与数值内核无关；heavy 后端（LAPACK/FFT）Go 生态有现成库（gonum lapack/fourier），符合「不重复造轮子」的 Go 习惯。
