//go:build goexperiment.simd

// Package simdvec holds vectorized inner loops for per-element, order-free
// operations (a[i]+b[i], dtype conversion). Two implementations share one
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
package simdvec

import "simd"

// AddF64 writes a[i]+b[i] into dst for i in [0, len(a)). dst, a and b must have
// equal length. Full-vector loads cover the body; the Part variants handle the
// tail without a scalar remainder loop. simd.Float64s{}.Len() is dynamic (not a
// const), so the loop must step by it.
func AddF64(dst, a, b []float64) {
	l := simd.Float64s{}.Len()
	n := len(a)
	i := 0
	for ; i+l <= n; i += l {
		simd.LoadFloat64s(a[i : i+l]).Add(simd.LoadFloat64s(b[i : i+l])).Store(dst[i : i+l])
	}
	if i < n {
		va, _ := simd.LoadFloat64sPart(a[i:])
		vb, _ := simd.LoadFloat64sPart(b[i:])
		va.Add(vb).StorePart(dst[i:])
	}
}

// SumF64 folds src into one float64 sum. Vector body keeps l parallel lane
// accumulators (lane i sees elements i, i+l, i+2l, …), tail folded scalar
// ascending after the last full vector; lanes merge left-to-right at the end.
// This is NOT the naive left-to-right sum: lane-merge order is pairwise-ish
// and may differ in the last ulp on generic inputs — the same documented
// semantics class as matmul's FMA note (see package doc in nosimd.go).
func SumF64(src []float64) float64 {
	l := simd.Float64s{}.Len()
	var acc simd.Float64s
	i := 0
	n := len(src)
	for ; i+l <= n; i += l {
		acc = acc.Add(simd.LoadFloat64s(src[i : i+l]))
	}
	tmp := make([]float64, l)
	acc.Store(tmp)
	sum := 0.0
	for _, v := range tmp {
		sum += v
	}
	for ; i < n; i++ {
		sum += src[i]
	}
	return sum
}

// SumAxis0F64 folds nrows consecutive rows of width rowstride from src into
// dst: dst[j] = src[j] + src[rowstride+j] + … (row order per column). Lanes
// span columns, so every output element keeps its own ascending row-order
// accumulator — bit-identical to the sequential row loop in both builds.
func SumAxis0F64(dst, src []float64, nrows, rowstride int) {
	l := simd.Float64s{}.Len()
	j := 0
	for ; j+l <= rowstride; j += l {
		acc := simd.LoadFloat64s(dst[j : j+l])
		for row := 1; row < nrows; row++ {
			acc = acc.Add(simd.LoadFloat64s(src[row*rowstride+j : row*rowstride+j+l]))
		}
		acc.Store(dst[j : j+l])
	}
	for ; j < rowstride; j++ {
		sum := dst[j]
		for row := 1; row < nrows; row++ {
			sum += src[row*rowstride+j]
		}
		dst[j] = sum
	}
}

// MatmulKernelF64 computes c = a@b for dense C-order float64 blocks: a is
// M×K, b is K×N, c is M×N and must be pre-zeroed. Lanes span N (each lane owns
// one output column), accumulating k ascending with fused MulAdd — probed:
// the fused chain matches a per-element math.FMA accumulation bit-for-bit.
// The scalar twin in nosimd.go therefore accumulates with math.FMA too, so
// both builds agree with each other AND with per-element FMA semantics. Note
// this differs from a plain mul-then-add sum in the last bits; the einsum
// xcheck (root package) compares against an independent FMA-form oracle.
func MatmulKernelF64(a, b, c []float64, M, K, N int) {
	l := simd.Float64s{}.Len()
	M4, N8 := M-M%4, N-N%l
	for i0 := 0; i0 < M4; i0 += 4 {
		c0 := c[(i0+0)*N:]
		c1 := c[(i0+1)*N:]
		c2 := c[(i0+2)*N:]
		c3 := c[(i0+3)*N:]
		for j0 := 0; j0 < N8; j0 += l {
			var r0, r1, r2, r3 simd.Float64s
			for k := range K {
				bk := simd.LoadFloat64s(b[k*N+j0 : k*N+j0+l : k*N+j0+l])
				a0 := simd.BroadcastFloat64s(a[(i0+0)*K+k])
				r0 = bk.MulAdd(a0, r0)
				a1 := simd.BroadcastFloat64s(a[(i0+1)*K+k])
				r1 = bk.MulAdd(a1, r1)
				a2 := simd.BroadcastFloat64s(a[(i0+2)*K+k])
				r2 = bk.MulAdd(a2, r2)
				a3 := simd.BroadcastFloat64s(a[(i0+3)*K+k])
				r3 = bk.MulAdd(a3, r3)
			}
			r0.Store(c0[j0 : j0+l : j0+l])
			r1.Store(c1[j0 : j0+l : j0+l])
			r2.Store(c2[j0 : j0+l : j0+l])
			r3.Store(c3[j0 : j0+l : j0+l])
		}
	}
	// Edge columns/rows outside the 4×l panels: plain k-ascending dot per
	// element, the same loop shape and order as the scalar kernel. Note the
	// fused panel lanes and these non-fused tail columns can differ in the
	// last ulp on generic inputs; the test oracle is per-region (FMA inside
	// panels, naive in the tails) to pin each region to its true arithmetic.
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

// ConvertF64ToF32 writes float32(src[i]) into dst. Go 1.27's simd package has
// no Float64s→Float32s conversion method (only Int32s.ConvertToFloat32 and the
// bit-reinterpret BitsToFloat32), so this stays the scalar loop — shortcut:
// no vector speedup here yet, upgrade when the stdlib grows the conversion op.
func ConvertF64ToF32(dst []float32, src []float64) {
	for i, v := range src {
		dst[i] = float32(v)
	}
}
