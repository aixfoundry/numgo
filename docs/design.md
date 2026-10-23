# numgo 设计：N 维泛型数组内核与项目布局

> 状态：待评审。批准后按 §8 步骤执行（删除现有 2D 实现、按新 layout 重建）。
> 依据：[roadmap.md](./roadmap.md)、[numpy-modules.md](./numpy-modules.md)、现有实现的 bug 史（见 §7）。

## 1. 目标与原则

- 语义对齐 numpy（广播、提升、NaN/Inf、轴、错误消息措辞），接口符合 Go 习惯（显式、可读、零魔法）。
- 泛型即 dtype 分派：`Array[T]` 编译期单态化，不做运行时 dtype 容器。
- **执行引擎唯一**：所有跨步索引代数只存在于 `internal/nditer` 一处（§7 规则 P0-1）。
- 根包 = numpy 顶层命名空间；重型子域独立子包。

## 2. 项目布局

```
numgo/
├── array.go            # Array[T] 类型、视图/形状方法（Reshape/Transpose/Slice…）
├── index.go            # At/Set/Idx/切片类型（Range）
├── create.go           # Zeros/Ones/Full/Arange/Linspace/Eye/FromFunction…
├── ufunc.go            # Add/Sub/Mul/Div/Pow/比较/数学函数族（经 nditer）
├── reduce.go           # Sum/Max/Min/Mean/Std/Var/ArgMax/Any/All/CumSum（axis+keepdims）
├── sort.go             # Sort/Argsort/Partition/Unique
├── manip.go            # Concatenate/Stack/Split/Tile/Repeat/Flip/Roll
├── dot.go              # Dot/Matmul/Einsum(后期)
├── errors.go           # panic 消息集中表（对拍 numpy 措辞）
├── convert.go          # AsType/Copy/AsArray/FromSlice…
├── internal/
│   ├── nditer/         # 执行引擎：多数组广播协同迭代、strides 步进、连续快路径
│   └── fixtures/       # golden fixture 加载/断言 helper（对拍 numpy）
├── linalg/             # Phase 2：gonum/lapack 后端
├── fft/                # Phase 2：gonum/dsp/fourier 后端
├── random/             # Phase 2：BitGenerator + Generator（bit-exact 对齐）
├── polynomial/         # Phase 3：6 个多项式类
├── assert/             # Phase 5：AssertAllClose 等（公开包，供下游测试用）
├── scripts/gen_fixtures.py   # 从本地 numpy 源生成 fixture
├── docs/
└── testdata/fixtures/        # golden 对拍数据
```

### numpy → 包映射

| numpy | numgo | 说明 |
|---|---|---|
| 顶层 498 名字 | 根包（按主题分文件） | numpy 2.x 已将 lib 提升到顶层，同构 |
| `numpy.linalg` | `numgo/linalg` | 子包，依赖根包 Array |
| `numpy.fft` | `numgo/fft` | 同上 |
| `numpy.random` | `numgo/random` | 同上 |
| `numpy.polynomial` | `numgo/polynomial` | 同上 |
| `numpy.ma` / `strings` | `numgo/ma` / `numgo/strings` | Phase 4 |
| `numpy.testing` | `numgo/assert` | 下游可复用的断言包 |
| nditer / strided loops | `internal/nditer` | 不导出，全库唯一索引代数 |

依赖方向严格单向：`linalg/fft/random/… → 根包 → internal`，无循环。

## 3. 核心类型

```go
type Number interface {
    ~float32 | ~float64 | ~complex64 | ~complex128
}

type Array[T Number] struct {
    shape   []int
    strides []int // 元素单位（非字节）
    offset  int
    data    []T
}
```

- **视图是一等公民**：`Slice/Transpose/Broadcast/Reshape`（可视图时）共享 `data`，用 `offset+strides` 表达；`Copy()` 才产生新底层数组。连续性按 strides 推导（`IsContiguous()`），不设 flag 字段。
- **索引公式只有一处**：`data[offset + Σ(iₖ·stridesₖ)]`，写在 nditer；其余代码禁止出现（§7 P0-1）。
- 语义二分（每个 API 必归一类，写入文档第一行）：
  1. **视图**（共享数据）：Slice/Transpose/Broadcast/Reshape
  2. **纯计算**（新数据）：所有 ufunc/reduction——默认形态
  3. **就地**：显式 `Into` 后缀（`AddInto(dst, b)`），不存在隐式变异
- 指针接收者引用语义，与 numpy 一致；`shape/strides` 对外只读（提供 `Shape() []int` 返回副本）。

## 4. API 风格规则

1. **方法优先**：操作单个数组 → 方法（`a.T()`, `a.Sum(axis)`）。构造器与多数组组合 → 包级函数（`Zeros`, `Concatenate(as ...*Array[T])`）。不提供方法+包级双份。
2. **轴**：`axis int`（-1 表全部，沿用现有约定）；N 维后支持 `axes []int` 变体；`keepdims bool` 显式参数。
3. **索引**：`At(coords ...int) T` / `Set(coords []int, v T)`；切片用 `type Range struct{ Start, Stop, Step int }` + `Slice(rs ...Range) *Array[T]`（`Range()` 构造器 + 负数/省略约定同 numpy）。
4. **错误**：非法形状/轴/索引 → panic，消息集中在 `errors.go`，措辞对拍 numpy（`operands could not be broadcast together with shapes (2,3) (1,2)`、`Singular matrix`、`index (3,1) is out of bounds for axis 0 with size 2`）；IEEE 特例（除零、空归约）不抛。
5. **类型转换显式**：`AsType[F, T](a *Array[F]) *Array[T]`，不做隐式提升（Go 静态类型边界，已在 roadmap 定案）。
6. **零值不可用**：`var a *Array[T]` 为 nil；构造器是唯一入口；nil 接收者 → panic（不再静默）。

## 5. 执行引擎 `internal/nditer`（接口草案）

对齐 numpy nditer 的职责：多数组广播对齐 + strides 步进 + 连续快路径 + 缓冲。

```go
package nditer

type Layout struct { Shape, Strides []int; Offset int } // Array[T] 的 dtype 无关投影

// Zip: dst 与 srcs 广播对齐后协同迭代。fn 收到连续块（引擎负责
// 转置/跨步展开，连续时零拷贝直通）。这是所有 ufunc/reduction 的入口。
func Zip[T any](dst *Layout, dstData []T, srcs []*Layout, srcData [][]T,
    fn func(dst []T, src [][]T))

// Reduce: 沿 axes 归约，keepdims 语义在根包层处理。
func Reduce[T any](l *Layout, data []T, axes []int,
    fn func(acc *T, v T))
```

Phase 1 先实现正确版（逐元素回调），连续快路径与分块后置——**先对拍正确，后谈性能**。

## 6. 泛型约束层级

```
Float     = ~float32 | ~float64                      // 现有
Number    = Float | ~complex64 | ~complex128         // Phase 0 目标
（整数不进 Number：Mean/Div 整除截断，roadmap 已定案）
```

`sort.go` 等需要序的运算用 `cmp.Ordered` 独立约束，不与 Number 绑定。

## 7. 硬规则（从 bug 史提炼，review 时逐条对照 PR）

| # | 规则 | 沉淀自 |
|---|---|---|
| P0-1 | 索引代数只在 nditer；运算代码禁止手写 `i*c+j` 类公式 | Swap/Delete/RangeDelete 5 处 stride 用错维度 |
| P0-2 | 构造器拷贝输入；视图必须文档标注「共享数据」 | New1DArray 别名 vs New2DArray 拷贝不一致；Delete 别名变异 |
| P0-3 | 变异必须显式命名（`Into` 后缀），禁止隐式原地 | Median(-1) 静默排序输入；Histogram2d Reshape 改写入参 |
| P0-4 | panic 消息只从 errors.go 取，禁止散落字面量 | 消息对拍可维护 |
| P0-5 | 每个运算至少一个非方阵用例 | 全部历史 bug 的共性：测试只有 3×3 |

## 8. 落地步骤（设计批准后）

1. 删除根包现有 10 个实现文件与其测试（git 历史保留全部；README 的函数表转为 API checklist 存 docs/）。
2. 建目录骨架 + `errors.go` + `internal/nditer` 最小实现（含 P0-5 非方阵测试）。
3. `Array[T]` 新类型 + create/convert/index 三组先行（可编译、可测）。
4. golden 对拍框架跑通（scripts/gen_fixtures.py + internal/fixtures）。
5. ufunc/reduce/manip/sort/dot 依序填充，每函数先 fixture 后实现。
6. 每 PR 对照 §7 硬规则自查。
