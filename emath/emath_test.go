package emath

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/aixfoundry/numgo"
)

// tol matches the root package fixture tolerance.
const tol = 1e-12

// checkC compares a complex128 array's shape and values against numpy emath
// goldens with NaN == NaN.
func checkC(t *testing.T, got *numgo.Array[complex128], shape []int, want []complex128) {
	t.Helper()
	if !equal(got.Shape(), shape) {
		t.Fatalf("shape = %v, want %v", got.Shape(), shape)
	}
	d := got.Data()
	if len(d) != len(want) {
		t.Fatalf("len = %d, want %d", len(d), len(want))
	}
	for i, w := range want {
		if !near(d[i], w) {
			t.Fatalf("data[%d] = %v, want %v", i, d[i], w)
		}
	}
}

func equal(a, b []int) bool {
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

func near(g, w complex128) bool {
	return nearF(real(g), real(w)) && nearF(imag(g), imag(w))
}

func nearF(g, w float64) bool {
	if math.IsNaN(w) {
		return math.IsNaN(g)
	}
	if math.IsInf(w, 0) {
		return g == w
	}
	return math.Abs(g-w) <= tol+tol*math.Abs(w)
}

func f64(vals ...float64) *numgo.Array[float64] {
	return numgo.New1DArray(vals)
}

// numpy: np.emath.sqrt([-4, 4]) -> [0.+2.j, 2.+0.j]
func TestSqrt(t *testing.T) {
	checkC(t, Sqrt(f64(-4, 4)), []int{2}, []complex128{2i, 2})
	checkC(t, Sqrt(numgo.New1DArray([]complex128{complex(-4, 0)})), []int{1}, []complex128{2i})
	// float32 input still yields complex128.
	checkC(t, Sqrt(numgo.New1DArray([]float32{-4, 4})), []int{2}, []complex128{2i, 2})
}

// numpy: np.emath.log([-1, 1]) -> [0.+3.14159265j, 0.+0.j]
func TestLog(t *testing.T) {
	checkC(t, Log(f64(-1, 1)), []int{2},
		[]complex128{complex(0, math.Pi), 0})
}

// numpy: np.emath.log2([-4, -8, 8]) -> [2.+4.53236014j, 3.+4.53236014j, 3.+0.j]
func TestLog2Log10(t *testing.T) {
	checkC(t, Log2(f64(-4, -8, 8)), []int{3}, []complex128{
		complex(2, 4.532360141827194), complex(3, 4.532360141827194), 3})
	// numpy: np.emath.log10([-10, -100, 100]) -> [1.+1.36437635j, 2.+1.36437635j, 2.+0.j]
	checkC(t, Log10(f64(-10, -100, 100)), []int{3}, []complex128{
		complex(1, 1.3643763538418412), complex(2, 1.3643763538418412), 2})
}

// numpy: np.emath.logn(2, [-4, 8]) -> [2.+4.53236014j, 3.+0.j]
func TestLogn(t *testing.T) {
	checkC(t, Logn(2, f64(-4, 8)), []int{2}, []complex128{
		complex(2, 4.532360141827194), 3})
}

// numpy: np.emath.power([-1, 4], [0.5, 2]) -> [~0+1j, 16+0j]
func TestPower(t *testing.T) {
	x := f64(-1, 4)
	p := f64(0.5, 2)
	checkC(t, Power(x, p), []int{2}, []complex128{
		cmplx.Pow(-1, 0.5), 16})
	// scalar p broadcasts (numpy emath.power([2,4], 2) stays real; ours is 4+0i).
	checkC(t, Power(f64(2, 4), f64(2)), []int{2}, []complex128{4, 16})
}

// numpy goldens: emath.arccos([2, 0.5]), emath.arcsin([-2, 0.5]),
// emath.arctanh([2, 0.5, -1]).
func TestArcTrig(t *testing.T) {
	checkC(t, Arccos(f64(2, 0.5)), []int{2}, []complex128{
		complex(0, -1.3169578969248164), complex(1.0471975511965976, 0)})
	checkC(t, Arcsin(f64(-2, 0.5)), []int{2}, []complex128{
		complex(-1.5707963267948966, 1.3169578969248166), complex(0.523598775598298, 0)})
	checkC(t, Arctanh(f64(2, 0.5, -1)), []int{3}, []complex128{
		complex(0.5493061443340549, 1.5707963267948966), complex(0.5493061443340548, 0),
		complex(math.Inf(-1), 0)})
}

// Shapes are preserved, including 2-D input.
func TestShapes(t *testing.T) {
	a := numgo.New2DArray([][]float64{{-4, 9}, {16, -1}})
	got := Sqrt(a)
	checkC(t, got, []int{2, 2}, []complex128{2i, 3, 4, 1i})
}
