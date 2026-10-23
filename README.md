# numgo

Go 泛型 N 维数组库，语义对齐 numpy，接口符合 Go 习惯。

> 重建中：旧 2D 实现已移除（git 历史 `e146d43` 及之前可查），N 维内核按 [docs/design.md](docs/design.md) 重建。

| 文档 | 内容 |
|---|---|
| [docs/numpy-modules.md](docs/numpy-modules.md) | numpy 2.6 模块清单（移植范围的事实基准） |
| [docs/roadmap.md](docs/roadmap.md) | 分期路线图（Phase 0-5、非目标） |
| [docs/design.md](docs/design.md) | 核心设计：布局、Array[T]、语义三分类、硬规则 |

## 布局

根包 = numpy 顶层命名空间；`linalg`/`fft`/`random`/`polynomial` 等镜像 numpy 子模块。
