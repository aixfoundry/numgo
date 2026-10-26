package numgo

import (
	"math"
	"testing"
)

func TestMatmul2x3Times3x2(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{7, 8, 9, 10, 11, 12}, 3, 2)
	got := a.Matmul(b)
	if !shapesEqual(got.Shape(), []int{2, 2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{58, 64, 139, 154} // numpy: A@B
	for i, w := range want {
		if v := got.At(i/2, i%2); math.Abs(v-w) > tol {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
	if a.At(0, 0) != 1 || b.At(0, 0) != 7 {
		t.Fatal("Matmul mutated an operand")
	}
}

func TestMatmulBatched(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3)
	b := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 3, 2)
	got := a.Matmul(b)
	if !shapesEqual(got.Shape(), []int{2, 2, 2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{22, 28, 49, 64, 220, 244, 301, 334} // numpy batched @
	for i, w := range want {
		if v := got.At(i/4, (i/2)%2, i%2); math.Abs(v-w) > tol {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestMatmulBroadcastLead(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	b := FromSlice([]float64{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, 2, 3, 2)
	got := a.Matmul(b)
	if !shapesEqual(got.Shape(), []int{2, 2, 2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	if got.At(0, 0, 1) == got.At(1, 0, 1) {
		t.Error("lead dim was not broadcast (rows unexpectedly equal)")
	}
	if math.Abs(got.At(0, 0, 1)-64) > tol || math.Abs(got.At(1, 0, 1)-100) > tol {
		t.Errorf("broadcast values wrong: %v, %v", got.At(0, 0, 1), got.At(1, 0, 1))
	}
}

func TestMatmul1DOperands(t *testing.T) {
	v := FromSlice([]float64{1, 2, 3, 4}, 4)
	m := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 4, 3)
	got := v.Matmul(m) // (4,)@(4,3) -> (3,)
	if !shapesEqual(got.Shape(), []int{3}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{70, 80, 90}
	for i, w := range want {
		if v := got.At(i); math.Abs(v-w) > tol {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
	mt := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got2 := mt.Matmul(FromSlice([]float64{7, 8, 9}, 3)) // (2,3)@(3,) -> (2,)
	if !shapesEqual(got2.Shape(), []int{2}) {
		t.Fatalf("shape = %v", got2.Shape())
	}
	want2 := []float64{50, 122}
	for i, w := range want2 {
		if v := got2.At(i); math.Abs(v-w) > tol {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
	// Batched matrix @ vector: (2,2,3)@(3,) -> (2,2).
	big := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3)
	got2b := big.Matmul(FromSlice([]float64{7, 8, 9}, 3))
	if !shapesEqual(got2b.Shape(), []int{2, 2}) || got2b.At(1, 0) != 194 {
		t.Fatalf("(2,2,3)@(3,) = %v", got2b)
	}
	// numpy rejects (3,)@(2,2,3): the promoted row (1,3) can't align with
	// leading dim 2, so the inner-dim check panics.
	mustPanic(t, "numgo: matmul: shape mismatch (1,3) x (2,2,3)", func() {
		FromSlice([]float64{7, 8, 9}, 3).Matmul(big)
	})
	mustPanic(t, "numgo: matmul: cannot multiply two 1-d arrays; use Vdot", func() {
		v.Matmul(v)
	})
}

func TestMatmulShapeMismatch(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	mustPanic(t, "numgo: matmul: shape mismatch (2,3) x (2,2)", func() {
		a.Matmul(b)
	})
	mustPanic(t, "numgo: operands could not be broadcast together with shapes (2,1,3) (4,3,2)", func() {
		a.Reshape(2, 1, 3).Matmul(FromFunction([]int{4, 3, 2}, func(coords ...int) float64 { return 1 }))
	})
}

func TestVdot(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if v := a.Vdot(a); math.Abs(v-91) > tol {
		t.Errorf("Vdot = %v, want 91", v)
	}
	// numpy: vdot([1+1j, 2-1j], [1+1j, 2-1j]) = (7+0j)
	c := FromSlice([]complex128{1 + 1i, 2 - 1i, 0, 0, 0, 0}, 2, 3)
	if v := c.Vdot(c); v != complex(7, 0) {
		t.Errorf("complex Vdot = %v, want (7+0i)", v)
	}
	// Non-conjugated would be 1+1i + 5+(-1i)·(-1i)... check conj explicitly:
	// conj(2-1i)*(2-1i) = 5, conj(1+1i)*(1+1i) = 2.
	// numpy vdot requires same-size 1-D; shape differs -> mismatch panic.
	mustPanic(t, "numgo: vdot: shape mismatch (3) x (2)", func() {
		FromSlice([]float64{1, 2, 3}, 3).Vdot(FromSlice([]float64{1, 2}, 2))
	})
}

func TestDotDispatch(t *testing.T) {
	// 1-D · 1-D -> [70]
	v := FromSlice([]float64{1, 2, 3, 4}, 4)
	w := FromSlice([]float64{5, 6, 7, 8}, 4)
	got := v.Dot(w)
	if !shapesEqual(got.Shape(), []int{1}) || math.Abs(got.At(0)-70) > tol {
		t.Fatalf("1d·1d = %v, want [70]", got.Shape())
	}
	// 2-D · 1-D -> (2,)
	m := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	got2 := m.Dot(FromSlice([]float64{1, 2, 3}, 3))
	if !shapesEqual(got2.Shape(), []int{2}) || got2.At(0) != 14 || got2.At(1) != 32 {
		t.Fatalf("2d·1d = %v", got2)
	}
	// 1-D · 2-D -> (3,)
	got3 := FromSlice([]float64{1, 2, 3, 4}, 4).Dot(FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 4, 3))
	if !shapesEqual(got3.Shape(), []int{3}) || got3.At(0) != 70 {
		t.Fatalf("1d·2d = %v", got3)
	}
	// 2-D · 2-D matrix product.
	got4 := m.Dot(FromSlice([]float64{1, 2, 3, 4, 5, 6}, 3, 2))
	if !shapesEqual(got4.Shape(), []int{2, 2}) || got4.At(1, 0) != 49 {
		t.Fatalf("2d·2d = %v", got4)
	}
	// N-D · N-D, unequal leads: a's full shape[:-1] then b's leads then N —
	// (2,2,3)@(2,3,2) -> (2,2,2,2) with out[ia,i,jb,j] (numpy-verified).
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3)
	b := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 3, 2)
	got5 := a.Dot(b)
	if !shapesEqual(got5.Shape(), []int{2, 2, 2, 2}) {
		t.Fatalf("nd·nd shape = %v", got5.Shape())
	}
	coords5 := [][4]int{} // (ia, i, jb, j) in C order — a's dims first
	for _, ia := range []int{0, 1} {
		for _, i := range []int{0, 1} {
			for _, jb := range []int{0, 1} {
				for _, j := range []int{0, 1} {
					coords5 = append(coords5, [4]int{ia, i, jb, j})
				}
			}
		}
	}
	// numpy ravel of np.dot(a,b) in (ia,i,jb,j) order.
	want5 := []float64{22, 28, 58, 64, 49, 64, 139, 154, 76, 100, 220, 244, 103, 136, 301, 334}
	for i, c := range coords5 {
		if v := got5.At(c[0], c[1], c[2], c[3]); math.Abs(v-want5[i]) > tol {
			t.Errorf("nd·nd%v = %v, want %v", c, v, want5[i])
		}
	}
	// (2,2,3)@(3,4): 2-D b shared across the batch.
	got6 := a.Dot(FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3, 4))
	if !shapesEqual(got6.Shape(), []int{2, 2, 4}) {
		t.Fatalf("(2,2,3)@(3,4) shape = %v", got6.Shape())
	}
	want6 := []float64{38, 44, 50, 56, 83, 98, 113, 128, 128, 152, 176, 200, 173, 206, 239, 272}
	for i, w := range want6 {
		if v := got6.At(i/8, (i/4)%2, i%4); math.Abs(v-w) > tol {
			t.Errorf("shared-b[%d] = %v, want %v", i, v, w)
		}
	}
	// (2,3)@(2,3,4): a's batch is OUTER, b's lead lands in the middle.
	c := FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	d := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, 2, 3, 4)
	got7 := c.Dot(d)
	if !shapesEqual(got7.Shape(), []int{2, 2, 4}) {
		t.Fatalf("(2,3)@(2,3,4) shape = %v", got7.Shape())
	}
	want7 := []float64{38, 44, 50, 56, 110, 116, 122, 128, 83, 98, 113, 128, 263, 278, 293, 308}
	for i, w := range want7 {
		if v := got7.At(i/8, (i/4)%2, i%4); math.Abs(v-w) > tol {
			t.Errorf("outer-batch[%d] = %v, want %v", i, v, w)
		}
	}
	// (3,)@(2,3,4): row vector, b's lead kept -> (2,4).
	got8 := FromSlice([]float64{1, 2, 3}, 3).Dot(d)
	if !shapesEqual(got8.Shape(), []int{2, 4}) || got8.At(0, 0) != 38 || got8.At(1, 0) != 110 {
		t.Fatalf("(3,)@(2,3,4) = %v", got8)
	}
	// (2,2,3)@(3,): column vector shared -> (2,2).
	got9 := a.Dot(FromSlice([]float64{1, 2, 3}, 3))
	if !shapesEqual(got9.Shape(), []int{2, 2}) || got9.At(0, 0) != 14 || got9.At(1, 1) != 68 {
		t.Fatalf("(2,2,3)@(3,) = %v", got9)
	}
	// Inner mismatch.
	mustPanic(t, "numgo: dot: shape mismatch (2,3) x (2,2)", func() {
		m.Dot(FromSlice([]float64{1, 2, 3, 4}, 2, 2))
	})
}

func TestDotComplexConj(t *testing.T) {
	// numpy dot of complex 1-D arrays conjugates the first argument:
	// dot([1+1j, 2-1j], [1+1j, 2-1j]) = 7.
	c := FromSlice([]complex128{1 + 1i, 2 - 1i}, 2)
	got := c.Dot(c)
	if got.At(0) != complex(7, 0) {
		t.Errorf("complex dot = %v, want (7+0i)", got.At(0))
	}
}

func TestMatmulComplex(t *testing.T) {
	a := FromSlice([]complex128{1 + 1i, 2, 3, 4}, 2, 2)
	b := FromSlice([]complex128{1, 1i, 1i, 1}, 2, 2)
	got := a.Matmul(b)
	want := []complex128{1 + 3i, 1 + 1i, 3 + 4i, 4 + 3i}
	for i, w := range want {
		if v := got.At(i/2, i%2); cmplxAbs(v-w) > 1e-12 {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

// cmplxAbs is a tiny helper so this file does not import math/cmplx twice.
func cmplxAbs(c complex128) float64 {
	return real(c)*real(c) + imag(c)*imag(c)
}

func TestMatmulFloat32(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSlice([]float32{7, 8, 9, 10, 11, 12}, 3, 2)
	got := a.Matmul(b)
	if got.At(0, 0) != 58 || got.At(1, 1) != 154 {
		t.Fatalf("float32 matmul = %v", got)
	}
}

func TestTensorDot(t *testing.T) {
	// axes=() outer product: numpy tensordot(x, y, 0) shape (2,3,2,2).
	x := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	y := FromSlice([]float64{0, 1, 2, 3}, 2, 2)
	outer := TensorDot(x, y, nil, nil)
	if !shapesEqual(outer.Shape(), []int{2, 3, 2, 2}) {
		t.Fatalf("outer shape = %v", outer.Shape())
	}
	assertAllClose(t, outer, Einsum("ij,kl->ijkl", x, y))
	// One-axis contraction: ij,jk->ik.
	b := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 4)
	mv := TensorDot(x, b, []int{1}, []int{0})
	if !shapesEqual(mv.Shape(), []int{2, 4}) {
		t.Fatalf("mv shape = %v", mv.Shape())
	}
	assertAllClose(t, mv, Einsum("ij,jk->ik", x, b))
	// Negative axes: contract a's last with b's second-to-last.
	// x (2,3) with b2 (3,4): tensordot(x, b2, (-1), (-2)) -> (2,4),
	// equal to x @ b2 (einsum 'ij,jk->ik').
	b2 := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 4)
	neg := TensorDot(x, b2, []int{-1}, []int{-2})
	if !shapesEqual(neg.Shape(), []int{2, 4}) {
		t.Fatalf("neg shape = %v", neg.Shape())
	}
	assertAllClose(t, neg, Einsum("ij,jk->ik", x, b2))
	// Batch contraction: tensordot(c,d,([2,1],[0,1])) with c (2,3,4), d (4,3,2);
	// numpy: [[880, 946], [2464, 2674]] == einsum('ijk,kjl->il').
	c := FromSlice(range64(24), 2, 3, 4)
	d := FromSlice(range64(24), 4, 3, 2)
	batch := TensorDot(c, d, []int{2, 1}, []int{0, 1})
	if !shapesEqual(batch.Shape(), []int{2, 2}) {
		t.Fatalf("batch shape = %v", batch.Shape())
	}
	assertAllClose(t, batch, Einsum("ijk,kjl->il", c, d))
	want := []float64{880, 946, 2464, 2674}
	for i, w := range want {
		if v := batch.At(i/2, i%2); v != w {
			t.Errorf("batch[%d] = %v, want %v", i, v, w)
		}
	}
	// Mismatched contraction dims panic (numpy: shape-mismatch for sum).
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected shape-mismatch panic")
			}
		}()
		TensorDot(x, b2, []int{0}, []int{0}) // 2 vs 3
	}()
	// Axis count mismatch panics.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected axes-count panic")
			}
		}()
		TensorDot(x, b2, []int{0}, []int{0, 1})
	}()
}

func range64(n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = float64(i)
	}
	return out
}

func assertAllClose(t *testing.T, got, want *Array[float64]) {
	t.Helper()
	if !shapesEqual(got.Shape(), want.Shape()) {
		t.Fatalf("shape %v, want %v", got.Shape(), want.Shape())
	}
	gd, wd := got.Ravel().Data()[:got.Size()], want.Ravel().Data()[:want.Size()]
	for i := range gd {
		if math.Abs(gd[i]-wd[i]) > 1e-9 {
			t.Errorf("[%d] = %v, want %v", i, gd[i], wd[i])
		}
	}
}
