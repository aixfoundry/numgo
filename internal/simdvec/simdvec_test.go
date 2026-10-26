package simdvec

import (
	"math"
	"testing"
)

func TestMatmulKernelF64Bits(t *testing.T) {
	// Mode-correct bit oracle. Default build: plain mul+add accumulation —
	// the kernel must match the scalar default-build matmul bits exactly.
	// GOEXPERIMENT=simd: fused MulAdd lanes — assert per-element math.FMA
	// bits. The root package's TestEinsumMatmulPathRandom independently pins
	// end-to-end bits in each mode.
	shapes := [][3]int{{5, 7, 9}, {4, 8, 8}, {3, 5, 17}, {1, 1, 1}, {7, 13, 6}, {2, 3, 0}}
	for _, sh := range shapes {
		M, K, N := sh[0], sh[1], sh[2]
		a := make([]float64, M*K)
		b := make([]float64, K*N)
		for i := range a {
			a[i] = float64(i%17-8) / 3
		}
		for i := range b {
			b[i] = float64(i%11-5) / 7
		}
		want := make([]float64, M*N)
		// Panel boundaries derived from the SAME runtime values the kernel
		// uses (simd.Float64s{}.Len() under the tag, 8 in the nosimd twin):
		// region→oracle classification must match kernel behavior at any
		// lane width (AVX-512 l=8 locally, AVX-2 l=4 on CI).
		panelEdge := N - N%f64Lanes()
		for i := range M {
			for j := range N {
				var acc float64
				// Oracle per region: panels are fused in the simd build
				// (MulAdd lanes); tail columns/rows (j >= panelEdge or
				// i >= M4) are plain mul+add in BOTH builds. Mirror that.
				panelCol := j < panelEdge && fmaMode
				panelRow := i < M-M%4 && fmaMode
				if panelCol && panelRow {
					for k := range K {
						acc = math.FMA(a[i*K+k], b[k*N+j], acc)
					}
				} else {
					for k := range K {
						acc += a[i*K+k] * b[k*N+j]
					}
				}
				want[i*N+j] = acc
			}
		}
		got := make([]float64, M*N)
		MatmulKernelF64(a, b, got, M, K, N)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("M=%d K=%d N=%d [%d]: got %v want %v", M, K, N, i, got[i], want[i])
			}
		}
	}
}

func TestMatmulKernelF32Mode(t *testing.T) {
	// Mode-correct oracle mirroring the F64 test. Default build: the shim
	// reports false and must leave c untouched (caller falls through to its
	// generic kernel). GOEXPERIMENT=simd: per-region bits — fused FMA inside
	// 4×l panels, plain mul+add in the edge columns/rows, exactly like the
	// scalar tails of the kernel itself.
	shapes := [][3]int{{5, 7, 9}, {4, 8, 8}, {3, 5, 17}, {1, 1, 1}, {7, 13, 6}, {2, 3, 0}}
	for _, sh := range shapes {
		M, K, N := sh[0], sh[1], sh[2]
		a := make([]float32, M*K)
		b := make([]float32, K*N)
		for i := range a {
			a[i] = float32(i%17-8) / 3
		}
		for i := range b {
			b[i] = float32(i%11-5) / 7
		}
		want := make([]float32, M*N)
		if fmaMode {
			// Panel width from the runtime lane count the kernel uses
			// (simd.Float32s{}.Len(): 16 on AVX-512, 8 on AVX2).
			panelEdge := N - N%f32Lanes()
			for i := range M {
				for j := range N {
					var acc float32
					panelCol := j < panelEdge
					panelRow := i < M-M%4
					if panelCol && panelRow {
						// float32 FMA oracle: widen to float64 (products and
						// sums of float32 are exact there) so math.FMA +
						// round-to-float32 == per-element fmaf.
						for k := range K {
							acc = float32(math.FMA(float64(a[i*K+k]), float64(b[k*N+j]), float64(acc)))
						}
					} else {
						for k := range K {
							acc += a[i*K+k] * b[k*N+j]
						}
					}
					want[i*N+j] = acc
				}
			}
		}
		got := make([]float32, M*N)
		for i := range got {
			got[i] = -1 // sentinel: shim must not touch c
		}
		handled := MatmulKernelF32(a, b, got, M, K, N)
		if handled != fmaMode {
			t.Fatalf("M=%d K=%d N=%d: handled=%v want %v", M, K, N, handled, fmaMode)
		}
		if !fmaMode {
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("M=%d K=%d N=%d [%d]: got %v want %v", M, K, N, i, got[i], want[i])
			}
		}
	}
}

func TestSumF64(t *testing.T) {
	// Mode-branched oracle: default = sequential left-to-right; simd =
	// lane accumulators merged left-to-right (documented pairwise-ish).
	// Small-int inputs make both agree; fraction inputs expose the order
	// difference only in the simd lane merge, so oracle must match mode.
	for n := 0; n <= 2*sumLanes()+1; n++ {
		src := make([]float64, n)
		for i := range src {
			src[i] = float64(i%17-8) / 4 // quarters: dyadic → order-free bits
		}
		got := SumF64(src)
		want := 0.0
		if sumLaneMerge {
			// simd semantics: l lane sums, merged left-to-right, then tail.
			// l is the runtime lane count the kernel uses.
			l := sumLanes()
			lanes := make([]float64, l)
			i := 0
			for ; i+l <= n; i += l {
				for jj := range l {
					lanes[jj] += src[i+jj]
				}
			}
			for _, v := range lanes {
				want += v
			}
			for ; i < n; i++ {
				want += src[i]
			}
		} else {
			for _, v := range src {
				want += v
			}
		}
		if got != want {
			t.Fatalf("n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestSumAxis0F64(t *testing.T) {
	// Contract: bit-identical to sequential row-order accumulation in BOTH
	// builds (lanes span columns only). If this ever fails, the lane argument
	// is wrong — stop and report, do not ship.
	for _, sh := range [][2]int{{5, 9}, {4, 8}, {3, 17}, {1, 1}, {7, 6}} {
		nrows, w := sh[0], sh[1]
		src := make([]float64, nrows*w)
		for i := range src {
			src[i] = float64(i%13-6) / 4 // dyadic quarters: order-free bits
		}
		want := make([]float64, w)
		for j := range w {
			sum := src[j]
			for row := 1; row < nrows; row++ {
				sum += src[row*w+j]
			}
			want[j] = sum
		}
		dst := make([]float64, w)
		copy(dst, src[:w])
		SumAxis0F64(dst, src, nrows, w)
		for j := range want {
			if dst[j] != want[j] {
				t.Fatalf("nrows=%d w=%d [%d]: got %v want %v (order broken!)", nrows, w, j, dst[j], want[j])
			}
		}
	}
}

func TestAddF64MatchesScalar(t *testing.T) {
	// Lengths 0..2*l+1 cover empty, exact-vector, and every tail shape.
	l := 8
	if testing.Short() {
		l = 1
	}
	for n := 0; n <= 2*l+1; n++ {
		a := make([]float64, n)
		b := make([]float64, n)
		for i := range a {
			a[i] = float64(i)*1e-9 - 3.5
			b[i] = float64(n-i) * 1e-9
		}
		want := make([]float64, n)
		for i := range want {
			want[i] = a[i] + b[i]
		}
		got := make([]float64, n)
		AddF64(got, a, b)
		for i := range want {
			if got[i] != want[i] { // bit-exact on purpose: IEEE add is order-free
				t.Fatalf("n=%d i=%d: got %v want %v", n, i, got[i], want[i])
			}
		}
	}
}

func TestConvertF64ToF32MatchesScalar(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 16, 17} {
		src := make([]float64, n)
		for i := range src {
			src[i] = float64(i) + 0.25
		}
		want := make([]float32, n)
		for i, v := range src {
			want[i] = float32(v)
		}
		got := make([]float32, n)
		ConvertF64ToF32(got, src)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d i=%d: got %v want %v", n, i, got[i], want[i])
			}
		}
	}
}
