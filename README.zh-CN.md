# numgo

Go 泛型 N 维数组库，语义对齐 numpy，接口符合 Go 习惯。

[![CI](https://github.com/aixfoundry/numgo/actions/workflows/ci.yml/badge.svg)](https://github.com/aixfoundry/numgo/actions/workflows/ci.yml)
![Go](https://img.shields.io/badge/Go-1.27%2B-00ADD8)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)

[English](README.md)

## 为什么是 numgo

- **numpy 语义 + Go 习惯。** 广播规则、类型提升、NaN/Inf 传播、轴语义乃至错误消息措辞都与 numpy 对齐；接口则遵循 Go 惯例——泛型做 dtype 分派、显式类型转换、视图是一等公民。
- **零依赖。** `go.mod` 只含 Go 标准库。LAPACK、FFT、随机数与数学内核全部在库内实现，无 cgo、无 BLAS。
- **可复现随机流。** `SeedSequence` + MT19937/PCG64 为逐行移植：同种子下，不依赖 libm 的分布与 numpy 逐位一致；libm 敏感分布经 Go 标准库数学，ULP 级接近。
- **双构建模式。** 默认纯 Go 构建；可选 `GOEXPERIMENT=simd` 构建加速热点路径（末位 ULP 差异已文档化）。两类构建各自维持库内结果契约。

## 安装

```
go get github.com/aixfoundry/numgo
```

需要 Go 1.27 或更高版本。

## 快速上手

```go
package main

import (
	"os"

	"github.com/aixfoundry/numgo"
)

func main() {
	a := numgo.Arange(0.0, 6.0, 1.0).Reshape(2, 3)    // [[0 1 2], [3 4 5]]
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

## 包

| 包 | 内容 |
|---|---|
| `numgo`（根） | Array[T] 视图/构造、ufunc、归约、排序、操纵、点积、einsum、IO（.npy/.npz/txt）、互操作 |
| `numgo/linalg` | numpy.linalg：解方程、最小二乘、SVD/QR/Cholesky/特征分解、范数（纯 Go LAPACK 内核） |
| `numgo/fft` | numpy.fft：mixed-radix FFT/逆变换/rfft 族 + fftshift，norm 三模式 |
| `numgo/random` | numpy.random：SeedSequence + MT19937/PCG64 + Generator 分布层（流与 numpy 一致：libm 无关路径逐位一致，libm 敏感分布使用 Go 标准库数学，ULP 级接近） |
| `numgo/polynomial` | numpy.polynomial：Polynomial/Chebyshev/Legendre/Laguerre/Hermite/HermiteE 六类 |
| `numgo/strings` | numpy.strings：StrArray 向量化字符串操作（upper/split/replace…） |
| `numgo/ma` | numpy.ma：MaskedArray = data + mask，掩码传播运算 |
| `numgo/emath` | numpy.emath：scimath 函数，出定义域自动升复数 |
| `numgo/assert` | numpy.testing：AssertAllClose/AssertArrayEqual 等断言，供下游测试用 |

## 性能

- 小矩阵 `Solve`（快 24×）、`Inv`（快 4.6×）、`StandardNormal`、`StandardGamma`、逐元素 `At`/`Set`（快 12×）快于 numpy。
- einsum、cumsum、掩码数组运算与 numpy 接近。
- 单线程逐元素与归约运算目前落后 numpy 向量化内核 3–21×。
- 可选的 `GOEXPERIMENT=simd` 构建：Matmul 提速 4.2×，`SumAll`/`SumAxis0` **反超 numpy**。

完整方法学与数据：[docs/performance.md](docs/performance.md)。

## numpy 对齐

numgo 逐模块对齐 numpy 2.x 语义，用 numpy 生成的 golden fixture 验证。部分接口有意跳过（rec 数组、`np.matrix`、遗留 `RandomState`），部分行为是刻意偏离（显式 `AsType` 取代隐式提升、panic 取代异常）。

逐模块清单、精度声明与偏离说明见 [docs/numpy-parity.md](docs/numpy-parity.md)，设计文档见 [docs/architecture.md](docs/architecture.md)，路线图见 [docs/roadmap.md](docs/roadmap.md)。

## 参与贡献

开发环境搭建、测试命令与 PR 要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

Apache-2.0，Copyright 2024-2026 kaisawind。零第三方依赖，数学计算基于纯 Go 标准库。本库包含自 NumPy（BSD-3-Clause）翻译的代码，详见 [NOTICE](NOTICE) 与 [LICENSES/](LICENSES/)。
