//go:build !goexperiment.simd

// Package simdvec holds vectorized inner loops for per-element, order-free
// operations (a[i]+b[i], dtype conversion, matmul inner kernel, sum
// reduction). Two implementations share one
// exported API, selected at build time:
//
//	simd.go   //go:build goexperiment.simd  → Go 1.27 experimental simd package
//	nosimd.go //go:build !goexperiment.simd → plain scalar loops
//
// Build tags are mandatory, not an optimization choice: the stdlib simd package
// only exists when the toolchain runs with GOEXPERIMENT=simd, so the default
// build could not compile simd.go at all. Under GOEXPERIMENT=simd the exported
// functions must stay bit-identical to the scalar versions, which restricts
// this package to operations with no cross-element dependence — reductions and
// dot products accumulate in a different order and must not be added here.
//
// Matmul is the one deliberate exception to bit-equality between builds:
// GOEXPERIMENT=simd 构建 matmul 为 FMA 累加语义，与默认构建（普通 mul+add）在
// 一般 float64 输入上末位 ULP 可能不同——与 numpy 跨 BLAS 后端行为同类；库内契约
// （einsum 折叠路径与通用路径 bit-identical）在两种构建下各自成立。默认构建的
// 代码路径保持 pre-P1 位级与速度不变，这是双构建设计的核心契约。
//
// Float32 matmul joins the exception list too (simd 构建的 MatmulKernelF32
// 为 FMA 累加，与默认构建 generic 内核在一般 float32 输入上末位 ULP 可能
// 不同)：默认构建中它是只返回 false 的 shim，float32 永远走 generic 内核，
// 默认路径位级与速度保持不变。
//
// SumF64 joins the exception list: simd 通道并行累加后合并，合并顺序近似两两
// （pairwise-ish），与默认构建的顺序累加在一般 float64 输入上末位 ULP 可能不同
// （见 SumF64 注释）。SumAxis0F64 不是例外：通道只跨列，每列仍是升序行累加，
// 两构建 bit-identical。
package simdvec

// AddF64 writes a[i]+b[i] into dst for i in [0, len(a)). dst, a and b must have
// equal length. IEEE float64 addition is exact and order-free, so the vector
// result is bit-identical to the scalar loop.
func AddF64(dst, a, b []float64) {
	for i, x := range a {
		dst[i] = x + b[i]
	}
}

// MatmulKernelF64 computes c = a@b for dense C-order float64 blocks: a is
// M×K, b is K×N, c is M×N and must be pre-zeroed. Verbatim lift of dot.go's
// pre-P1 kernel (git show 683270a:dot.go, 4×8 register-blocked i-k-j, plain
// mul+add): "default unchanged" is the core contract of the dual-build design,
// so default-mode matmul keeps pre-P1 bits AND speed. The GOEXPERIMENT=simd
// twin accumulates fused (MulAdd) and may differ in the last ulp on generic
// float64 inputs — documented in the package doc and docs/perf-baseline.md.
func MatmulKernelF64(a, b, c []float64, M, K, N int) {
	M4, N8 := M-M%4, N-N%8
	for i0 := 0; i0 < M4; i0 += 4 {
		c0 := c[(i0+0)*N:]
		c1 := c[(i0+1)*N:]
		c2 := c[(i0+2)*N:]
		c3 := c[(i0+3)*N:]
		for j0 := 0; j0 < N8; j0 += 8 {
			var r [4][8]float64
			for k := range K {
				bk := b[k*N+j0 : k*N+j0+8 : k*N+j0+8]
				a0, a1 := a[(i0+0)*K+k], a[(i0+1)*K+k]
				a2, a3 := a[(i0+2)*K+k], a[(i0+3)*K+k]
				r0 := &r[0]
				r0[0] += a0 * bk[0]
				r0[1] += a0 * bk[1]
				r0[2] += a0 * bk[2]
				r0[3] += a0 * bk[3]
				r0[4] += a0 * bk[4]
				r0[5] += a0 * bk[5]
				r0[6] += a0 * bk[6]
				r0[7] += a0 * bk[7]
				r1 := &r[1]
				r1[0] += a1 * bk[0]
				r1[1] += a1 * bk[1]
				r1[2] += a1 * bk[2]
				r1[3] += a1 * bk[3]
				r1[4] += a1 * bk[4]
				r1[5] += a1 * bk[5]
				r1[6] += a1 * bk[6]
				r1[7] += a1 * bk[7]
				r2 := &r[2]
				r2[0] += a2 * bk[0]
				r2[1] += a2 * bk[1]
				r2[2] += a2 * bk[2]
				r2[3] += a2 * bk[3]
				r2[4] += a2 * bk[4]
				r2[5] += a2 * bk[5]
				r2[6] += a2 * bk[6]
				r2[7] += a2 * bk[7]
				r3 := &r[3]
				r3[0] += a3 * bk[0]
				r3[1] += a3 * bk[1]
				r3[2] += a3 * bk[2]
				r3[3] += a3 * bk[3]
				r3[4] += a3 * bk[4]
				r3[5] += a3 * bk[5]
				r3[6] += a3 * bk[6]
				r3[7] += a3 * bk[7]
			}
			b0 := c0[j0 : j0+8 : j0+8]
			b1 := c1[j0 : j0+8 : j0+8]
			b2 := c2[j0 : j0+8 : j0+8]
			b3 := c3[j0 : j0+8 : j0+8]
			for jj := range 8 {
				b0[jj] = r[0][jj]
				b1[jj] = r[1][jj]
				b2[jj] = r[2][jj]
				b3[jj] = r[3][jj]
			}
		}
	}
	// Edge columns/rows outside the 4×8 panels: plain k-ascending dot per
	// element (same accumulation order, bit-identical).
	for i := range M {
		for j := N8; j < N; j++ {
			var sum float64
			for k := range K {
				sum += a[i*K+k] * b[k*N+j]
			}
			c[i*N+j] = sum
		}
	}
	for i := M4; i < M; i++ {
		for j := range N8 {
			var sum float64
			for k := range K {
				sum += a[i*K+k] * b[k*N+j]
			}
			c[i*N+j] = sum
		}
	}
}

// MatmulKernelF32 is the default-build shim: it only reports false so the
// caller falls through to its generic kernel — default-build float32 matmul
// keeps today's exact code path, bits and speed (the core contract). The
// GOEXPERIMENT=simd twin computes with fused lanes and reports true; its
// fused accumulation joins the documented last-ulp exception list above.
func MatmulKernelF32(a, b, c []float32, M, K, N int) bool {
	return false
}

// SumF64 folds src into one float64 sum: the existing sequential
// left-to-right loop, verbatim semantics (default build keeps pre-P2 bits).
func SumF64(src []float64) float64 {
	sum := 0.0
	for _, v := range src {
		sum += v
	}
	return sum
}

// SumAxis0F64 folds nrows consecutive rows of width rowstride from src into
// dst: dst[j] += src[row*rowstride+j] for rows 1..n-1 (row 0 already folded
// into dst by the caller). Sequential row order per column — bit-identical
// to the simd twin by construction (lanes span columns only).
func SumAxis0F64(dst, src []float64, nrows, rowstride int) {
	for j := range rowstride {
		sum := dst[j]
		for row := 1; row < nrows; row++ {
			sum += src[row*rowstride+j]
		}
		dst[j] = sum
	}
}

// ConvertF64ToF32 writes float32(src[i]) into dst for i in [0, len(src)).
// dst must have at least len(src) elements.
func ConvertF64ToF32(dst []float32, src []float64) {
	for i, v := range src {
		dst[i] = float32(v)
	}
}
