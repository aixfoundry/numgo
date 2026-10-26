package random

import (
	"math"
	"testing"

	"github.com/aixfoundry/numgo"
)

// oraclePermute replays the numpy permuted algorithm manually: for each
// all-but-axis slice in C order, Fisher-Yates with random_interval, exactly
// what g.Shuffle does on a 1-D row.
func TestPermuted(t *testing.T) {
	g := DefaultRng(42)
	x := numgo.Arange(0.0, 24, 1).Reshape(2, 3, 4)
	before := x.Copy().Data()[:24]

	y := g.Permuted(x, 2)

	// Original untouched.
	for i := range 24 {
		if x.Data()[i] != float64(i) {
			t.Fatalf("Permuted mutated x at %d", i)
		}
	}
	_ = before

	// Shape and permutation property: each axis-2 slice is a rearrangement.
	if !shapes3Equal(y.Shape(), []int{2, 3, 4}) {
		t.Fatalf("shape = %v", y.Shape())
	}

	// Stream oracle: a fresh generator replaying per-slice Shuffle must
	// produce the identical array (C-order slice iteration, same Fisher-Yates
	// walk — numpy _generator.pyx permuted semantics). Each row slice is
	// copied out, shuffled, and written back (a non-contiguous view can't be
	// shuffled in place through Data()).
	g2 := DefaultRng(42)
	manual := x.Copy()
	for i := range 2 {
		for j := range 3 {
			view := manual.Slice(numgo.R(i, i+1, 1), numgo.R(j, j+1, 1), numgo.R(0, 4, 1))
			row := view.Copy()
			g2.Shuffle(row)
			numgo.CopyInto(view, row)
		}
	}
	for idx := range 24 {
		if y.Data()[idx] != manual.Data()[idx] {
			t.Fatalf("Permuted[%d] = %v, manual shuffle %v", idx, y.Data()[idx], manual.Data()[idx])
		}
	}

	// axis=0: each length-2 column shuffled independently; slices in C order
	// over the remaining dims (3,4). Columns are non-contiguous, so copy,
	// shuffle, write back.
	g3 := DefaultRng(7)
	y0 := g3.Permuted(x, 0)
	g4 := DefaultRng(7)
	manual0 := x.Copy()
	for j := range 3 {
		for k := range 4 {
			view := manual0.Slice(numgo.R(0, 2, 1), numgo.R(j, j+1, 1), numgo.R(k, k+1, 1))
			col := view.Copy()
			g4.Shuffle(col)
			numgo.CopyInto(view, col)
		}
	}
	for idx := range 24 {
		if y0.Data()[idx] != manual0.Data()[idx] {
			t.Fatalf("Permuted axis=0 [%d] = %v, manual %v", idx, y0.Data()[idx], manual0.Data()[idx])
		}
	}

	// Negative axis: axis=-1 == axis=2 with a same-seed generator.
	yNeg := DefaultRng(42).Permuted(x, -1)
	yPos := DefaultRng(42).Permuted(x, 2)
	for idx := range 24 {
		if yNeg.Data()[idx] != yPos.Data()[idx] {
			t.Fatalf("axis=-1 diverges from axis=2 at %d", idx)
		}
	}

	// 1-D: equals PermutationArray (plain Shuffle of a copy).
	y1 := DefaultRng(5).Permuted(numgo.Arange(0.0, 8, 1), -1)
	p1 := DefaultRng(5).PermutationArray(numgo.Arange(0.0, 8, 1))
	for idx := range 8 {
		if y1.Data()[idx] != p1.Data()[idx] {
			t.Fatalf("1-D Permuted diverges from Shuffle at %d", idx)
		}
	}

	// Out-of-range axis panics.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected axis panic")
			}
		}()
		DefaultRng(1).Permuted(x, 3)
	}()
}

func shapes3Equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestBytes: numpy Generator.bytes — ceil(n/4) uint32 draws via
// integers(0, 2^32), little-endian, truncated to n.
func TestBytes(t *testing.T) {
	g := DefaultRng(123)
	b := g.Bytes(10)
	if len(b) != 10 {
		t.Fatalf("len = %d, want 10", len(b))
	}
	// Reproducible under the same seed.
	b2 := DefaultRng(123).Bytes(10)
	for i := range b {
		if b[i] != b2[i] {
			t.Fatalf("Bytes not reproducible at %d", i)
		}
	}
	// Distinct seeds diverge.
	b3 := DefaultRng(124).Bytes(10)
	same := true
	for i := range b {
		if b[i] != b3[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("distinct seeds produced identical bytes")
	}
	// Truncation: bytes(1) is the first byte of bytes(4)'s stream (one
	// uint32 draw either way) — numpy consumes the same number of draws.
	bOne := DefaultRng(9).Bytes(1)
	bFour := DefaultRng(9).Bytes(4)
	if bOne[0] != bFour[0] {
		t.Fatalf("bytes(1)=%v inconsistent with bytes(4)=%v", bOne, bFour)
	}
	// n=0 is empty and consumes no draws.
	if len(DefaultRng(1).Bytes(0)) != 0 {
		t.Fatal("bytes(0) not empty")
	}
	// Random byte stream sanity: not all zero.
	nonzero := false
	for _, v := range DefaultRng(2).Bytes(64) {
		if v != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("byte stream all zero")
	}
	_ = math.NaN
}
