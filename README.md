# numgo

Go 泛型 N 维数组库，语义对齐 numpy，接口符合 Go 习惯。

> 重建中：旧 2D 实现已移除（git 历史 `e146d43` 及之前可查），N 维内核按 [docs/design.md](docs/design.md) 重建。

`Go 1.27+ · 零第三方依赖 · 语义对拍 numpy 2.6`

| 文档 | 内容 |
|---|---|
| [docs/numpy-modules.md](docs/numpy-modules.md) | numpy 2.6 模块清单（移植范围的事实基准） |
| [docs/roadmap.md](docs/roadmap.md) | 分期路线图（Phase 0-5、非目标） |
| [docs/design.md](docs/design.md) | 核心设计：布局、Array[T]、语义三分类、硬规则 |

## 包

| 包 | 内容 |
|---|---|
| `numgo`（根） | Array[T] 视图/构造、ufunc、归约、排序、操纵、点积、einsum、IO（.npy/.npz/txt）、互操作 |
| `numgo/linalg` | numpy.linalg：解方程、最小二乘、SVD/QR/Cholesky/特征分解、范数（纯 Go LAPACK 内核） |
| `numgo/fft` | numpy.fft：mixed-radix FFT/逆变换/rfft 族 + fftshift，norm 三模式 |
| `numgo/random` | numpy.random：SeedSequence + MT19937/PCG64 + Generator 分布层（bit-exact 对齐） |
| `numgo/polynomial` | numpy.polynomial：Polynomial/Chebyshev/Legendre/Laguerre/Hermite/HermiteE 六类 |
| `numgo/strings` | numpy.strings：StrArray 向量化字符串操作（upper/split/replace…） |
| `numgo/ma` | numpy.ma：MaskedArray = data + mask，掩码传播运算 |
| `numgo/emath` | numpy.emath：scimath 函数，出定义域自动升复数 |
| `numgo/assert` | numpy.testing：AssertAllClose/AssertArrayEqual 等断言，供下游测试用 |

## 快速上手

```go
package main

import (
    "os"

    "github.com/kaisawind/numgo"
)

func main() {
    a := numgo.Arange(0, 6, 1).Reshape(2, 3)          // [[0 1 2], [3 4 5]]
    b := numgo.Ones[float64](1, 3)                     // 广播相加 (2,3)+(1,3)
    c := a.Add(b)                                      // 纯计算：新数据

    m := numgo.Ones[float64](3, 2)
    p := c.Matmul(m)                                   // (2,3) @ (3,2) → (2,2)

    numgo.SaveFile("a.npy", a)                         // .npy 写/读
    if v, err := numgo.LoadFile("a.npy"); err == nil {
        arr := v.AsFloat64()
        _ = arr.Sum()                                  // 15
    }
    _ = p
    os.Remove("a.npy")
}
```

## 布局

根包 = numpy 顶层命名空间；`linalg`/`fft`/`random`/`polynomial` 等镜像 numpy 子模块。
