package numgo

import (
	"math"
	"testing"
)

func TestConvolveModes(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	v := FromSlice([]float64{0, 1, 0.5})
	// numpy: np.convolve([1,2,3],[0,1,0.5],mode).
	assertClose(t, Convolve(a, v, "full"), []float64{0, 1, 2.5, 4, 1.5})
	assertClose(t, Convolve(a, v, "same"), []float64{1, 2.5, 4})
	assertClose(t, Convolve(a, v, "valid"), []float64{2.5})
	// Swapped lengths: numpy convolves with the longer as the kernel.
	assertClose(t, Convolve(v, a, "full"), []float64{0, 1, 2.5, 4, 1.5})
	// Flattening: a 2-D input ravels to the same 1-D sequence.
	a2 := FromSlice([]float64{1, 2, 3}, 3, 1)
	assertClose(t, Convolve(a2, v, "full"), Convolve(a, v, "full").Data()[:5])
	// Complex: numpy np.convolve([1+2i,3],[0,1j],'full') = [0, -2+1j, 3i].
	ca := FromSlice([]complex128{1 + 2i, 3})
	cv := FromSlice([]complex128{0, 1i})
	assertCclose(t, Convolve(ca, cv, "full"), []complex128{0, -2 + 1i, 3i})
	// Empty input panics like numpy's ValueError.
	mustPanic(t, "numgo: a cannot be empty", func() { Convolve(New1DArray([]float64{}), v, "full") })
	// Unknown mode panics.
	mustPanic(t, "numgo: mode must be one of 'full', 'same', 'valid'", func() { Convolve(a, v, "wrap") })
}

func TestCorrelate(t *testing.T) {
	a := FromSlice([]float64{1, 2, 3})
	v := FromSlice([]float64{0, 1, 0.5})
	// numpy: np.correlate([1,2,3],[0,1,0.5],mode).
	assertClose(t, Correlate(a, v, "full"), []float64{0.5, 2, 3.5, 3, 0})
	assertClose(t, Correlate(a, v, "same"), []float64{2, 3.5, 3})
	assertClose(t, Correlate(a, v, "valid"), []float64{3.5})
	// Cross-check against Convolve over a reversed kernel.
	assertClose(t, Correlate(a, v, "full"), Convolve(a, Flip(v, 0), "full").Data()[:5])
	// correlate(a, v) != correlate(v, a) (numpy: not commutative).
	assertNotEqual(t, Correlate(a, v, "full"), Correlate(v, a, "full"))
	// Complex: numpy np.correlate([1+2j,3],[0,1j],'full') conjugates the
	// kernel (correlate2): [2-1j, -3j, 0]. The core keeps the raw product;
	// the conj difference shows up in the kernel reversal sign.
	ca := FromSlice([]complex128{1 + 2i, 3})
	cv := FromSlice([]complex128{0, 1i})
	// numpy conjugates v: c_k = Σ a[m+k]·conj(v[m]).
	assertCclose(t, Correlate(ca, cv, "full"), []complex128{2 - 1i, -3i, 0})
}

func assertClose(t *testing.T, got *Array[float64], want []float64) {
	t.Helper()
	if got.Size() != len(want) {
		t.Fatalf("size %d, want %d", got.Size(), len(want))
	}
	for i, w := range want {
		if v := got.At(i); math.Abs(v-w) > tol {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func assertCclose(t *testing.T, got *Array[complex128], want []complex128) {
	t.Helper()
	if got.Size() != len(want) {
		t.Fatalf("size %d, want %d", got.Size(), len(want))
	}
	for i, w := range want {
		if v := got.At(i); math.Abs(real(v)-real(w)) > tol || math.Abs(imag(v)-imag(w)) > tol {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func assertNotEqual[T Number](t *testing.T, got, want *Array[T]) {
	t.Helper()
	if shapesEqual(got.Shape(), want.Shape()) {
		same := true
		for i := range got.Ravel().Data()[:got.Size()] {
			if got.Ravel().Data()[i] != want.Ravel().Data()[i] {
				same = false
				break
			}
		}
		if same {
			t.Fatal("expected different arrays")
		}
	}
}
