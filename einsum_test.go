package numgo

import (
	"math"
	"testing"
)

func TestEinsumMatmulMatchesMatmul(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	b := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 3, 2)
	got := Einsum("ij,jk->ik", a, b)
	want := a.Matmul(b) // numpy: [[10,13],[28,40]]
	if !shapesEqual(got.Shape(), []int{2, 2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	for i := range 2 {
		for j := range 2 {
			if got.At(i, j) != want.At(i, j) {
				t.Errorf("[%d,%d] = %v, want %v", i, j, got.At(i, j), want.At(i, j))
			}
		}
	}
}

func TestEinsumBatchMatmulEllipsis(t *testing.T) {
	// numpy: [[[10,13],[28,40]], [[172,193],[244,274]]]
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 2, 2, 3)
	b := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 2, 3, 2)
	got := Einsum("...ij,...jk->...ik", a, b)
	if !shapesEqual(got.Shape(), []int{2, 2, 2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{10, 13, 28, 40, 172, 193, 244, 274}
	for k := range 8 {
		if v := got.At(k/4, (k/2)%2, k%2); math.Abs(v-want[k]) > tol {
			t.Errorf("[%d] = %v, want %v", k, v, want[k])
		}
	}
}

func TestEinsumDotSumScalar(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2}, 3)
	b := FromSlice([]float64{10, 20, 30}, 3)
	got := Einsum("i,i->", a, b) // numpy: 80
	if !shapesEqual(got.Shape(), []int{1}) || got.At(0) != 80 {
		t.Fatalf("got shape %v value %v, want [1] = 80", got.Shape(), got.At(0))
	}
}

func TestEinsumDiagonalSquare(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}, 3, 3)
	got := Einsum("ii->i", a) // numpy: [0,4,8]
	want := []float64{0, 4, 8}
	if !shapesEqual(got.Shape(), []int{3}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	for i, w := range want {
		if v := got.At(i); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumDiagonalNonSquareMinDim(t *testing.T) {
	// min(2,4) diagonal: numpy np.diagonal agrees ([[0,5],[10,15]] rows).
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6, 7}, 2, 4)
	got := Einsum("ii->i", a)
	want := []float64{0, 5}
	if !shapesEqual(got.Shape(), []int{2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	for i, w := range want {
		if v := got.At(i); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumImplicitTrace(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}, 3, 3)
	got := Einsum("ii", a) // numpy: 12 as a scalar
	if !shapesEqual(got.Shape(), []int{1}) || got.At(0) != 12 {
		t.Fatalf("got shape %v value %v, want [1] = 12", got.Shape(), got.At(0))
	}
}

func TestEinsumBatchTrace(t *testing.T) {
	// numpy "...ii->...i": batches [[0,4,8],[9,13,17]]
	a := FromSlice([]float64{
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16, 17,
	}, 2, 3, 3)
	got := Einsum("...ii->...i", a)
	if !shapesEqual(got.Shape(), []int{2, 3}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{0, 4, 8, 9, 13, 17}
	for i, w := range want {
		if v := got.At(i/3, i%3); math.Abs(v-w) > tol {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumTranspose(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	got := Einsum("ij->ji", a)
	if !shapesEqual(got.Shape(), []int{3, 2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{0, 3, 1, 4, 2, 5}
	for i, w := range want {
		if v := got.At(i/2, i%2); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumImplicitMatmul(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	b := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 3, 2)
	got := Einsum("ij,jk", a, b) // implicit output "ik"
	if !shapesEqual(got.Shape(), []int{2, 2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{10, 13, 28, 40}
	for i, w := range want {
		if v := got.At(i/2, i%2); math.Abs(v-w) > tol {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumSumOverEllipsisTail(t *testing.T) {
	// "i...->i": the explicit output omits "...", so the broadcast tail is
	// summed. numpy errors here; we follow the doc comment's extension.
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	got := Einsum("i...->i", a) // row sums [3,12]
	want := []float64{3, 12}
	if !shapesEqual(got.Shape(), []int{2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	for i, w := range want {
		if v := got.At(i); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumMatrixVector(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	v := FromSlice([]float64{1, 1, 1}, 3)
	got := Einsum("ij,j->i", a, v) // numpy: [3,12]
	want := []float64{3, 12}
	if !shapesEqual(got.Shape(), []int{2}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	for i, w := range want {
		if v := got.At(i); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumChainThreeOperands(t *testing.T) {
	// numpy: [[52,75,98,121],[160,228,296,364]]
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	b := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 3, 2)
	c := FromSlice([]float64{0, 1, 2, 3, 4, 5, 6, 7}, 2, 4)
	got := Einsum("ij,jk,kl->il", a, b, c)
	if !shapesEqual(got.Shape(), []int{2, 4}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{52, 75, 98, 121, 160, 228, 296, 364}
	for i, w := range want {
		if v := got.At(i/4, i%4); math.Abs(v-w) > tol {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumOuter(t *testing.T) {
	a := FromSlice([]float64{0, 1}, 2)
	b := FromSlice([]float64{0, 1, 2}, 3)
	got := Einsum("i,j->ij", a, b)
	if !shapesEqual(got.Shape(), []int{2, 3}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{0, 0, 0, 0, 1, 2}
	for i, w := range want {
		if v := got.At(i/3, i%3); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumBroadcastEllipsisBetweenOperands(t *testing.T) {
	// (1,3) broadcast against (3,): numpy ...i,...i->... -> [6]
	a := Ones[float64](1, 3)
	b := FromSlice([]float64{1, 2, 3}, 3)
	got := Einsum("...i,...i->...", a, b)
	if !shapesEqual(got.Shape(), []int{1}) || got.At(0) != 6 {
		t.Fatalf("got shape %v value %v, want [1] = 6", got.Shape(), got.At(0))
	}
}

func TestEinsumImplicitEllipsisFirst(t *testing.T) {
	// numpy implicit output puts broadcast dims before free labels:
	// 'i...j' on (2,3,4) -> (3,2,4), out[b,i,j] = a[i,b,j].
	a := FromSlice([]float64{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	}, 2, 3, 4)
	got := Einsum("i...j", a)
	if !shapesEqual(got.Shape(), []int{3, 2, 4}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	if got.At(0, 0, 0) != 0 || got.At(0, 1, 0) != 12 || got.At(2, 0, 1) != 9 {
		t.Errorf("values wrong: %v %v %v", got.At(0, 0, 0), got.At(0, 1, 0), got.At(2, 0, 1))
	}
}

func TestEinsumEllipsisDimsBroadcastAcrossOperands(t *testing.T) {
	// numpy: 'i...,i' on ones(3,4) x [1,2,3] -> [6,6,6,6]: i is summed (it
	// appears twice), the ellipsis tail stays as the output.
	a := Ones[float64](3, 4)
	b := FromSlice([]float64{1, 2, 3}, 3)
	got := Einsum("i...,i", a, b)
	if !shapesEqual(got.Shape(), []int{4}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	for i := range 4 {
		if got.At(i) != 6 {
			t.Errorf("[%d] = %v, want 6", i, got.At(i))
		}
	}
}

func TestEinsumComplex(t *testing.T) {
	// numpy: (1+2j)·(1-1j) + (3+4j)·2 = (9+9j)
	a := New1DArray([]complex128{1 + 2i, 3 + 4i})
	b := New1DArray([]complex128{1 - 1i, 2})
	got := Einsum("i,i->", a, b)
	want := complex(9, 9)
	if got.At(0) != want {
		t.Fatalf("got %v, want %v", got.At(0), want)
	}
}

func TestEinsumFloat32(t *testing.T) {
	a := FromSlice([]float32{0, 1, 2, 3, 4, 5}, 2, 3)
	b := FromSlice([]float32{0, 1, 2, 3, 4, 5}, 3, 2)
	got := Einsum("ij,jk->ik", a, b)
	want := []float32{10, 13, 28, 40}
	for i, w := range want {
		if v := got.At(i/2, i%2); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestEinsumOperandsUntouched(t *testing.T) {
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	b := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 3, 2)
	Einsum("ij,jk->ik", a, b)
	if a.At(0, 0) != 0 || b.At(0, 0) != 0 {
		t.Fatal("Einsum mutated an operand")
	}
}

func TestEinsumLabelCountPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != "numgo: einsum: subscript label count 3 does not match array dimension 2" {
			t.Fatalf("recover = %v", r)
		}
	}()
	Einsum("ijk", FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3))
}

func TestEinsumTooFewOperandsPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != "numgo: einsum: 2 subscripts for 1 operands" {
			t.Fatalf("recover = %v", r)
		}
	}()
	Einsum("ij,jk", FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3))
}

func TestEinsumTooManyOperandsPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != "numgo: einsum: 2 subscripts for 3 operands" {
			t.Fatalf("recover = %v", r)
		}
	}()
	a := FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	Einsum("ij,jk", a, a, a)
}

func TestEinsumOutputLabelPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != "numgo: einsum: output label z does not appear in any input" {
			t.Fatalf("recover = %v", r)
		}
	}()
	Einsum("ij->iz", FromSlice([]float64{0, 1, 2, 3, 4, 5}, 2, 3))
}

func TestEinsumOutputEllipsisPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != "numgo: einsum: output has an ellipsis but no operand provides one" {
			t.Fatalf("recover = %v", r)
		}
	}()
	Einsum("i->...", FromSlice([]float64{0, 1, 2}, 3))
}

func TestEinsumBadCharPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != `numgo: einsum: invalid subscript '$' in operand 0, subscripts must be letters` {
			t.Fatalf("recover = %v", r)
		}
	}()
	Einsum("i$", FromSlice([]float64{0, 1, 2}, 3))
}

func TestEinsumStrayDotPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != `numgo: einsum: invalid subscript '.' in operand 0, subscripts must be letters` {
			t.Fatalf("recover = %v", r)
		}
	}()
	Einsum("i..", FromSlice([]float64{0, 1, 2}, 3))
}
