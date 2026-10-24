package assert

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/aixfoundry/numgo"
)

// mockT records Errorf calls so failing assertions can be tested. The
// embedded nil testing.TB satisfies the interface; the assert package only
// ever calls Errorf and Helper, so the nil methods are never invoked.
type mockT struct {
	testing.TB
	msgs []string
}

func (m *mockT) Errorf(format string, args ...any) {
	m.msgs = append(m.msgs, fmt.Sprintf(format, args...))
}
func (m *mockT) Helper() {}

func arr[T numgo.Number](data []T, shape ...int) *numgo.Array[T] {
	return numgo.FromSlice(data, shape...)
}

func TestEqual(t *testing.T) {
	Equal(t, 1, 1)
	Equal(t, math.NaN(), math.NaN())
	Equal(t, []int{1, 2}, []int{1, 2})
	Equal(t, 1.0, 1.0)

	m := &mockT{}
	Equal(m, 1, 2)
	Equal(m, 1, 1.0) // different types
	Equal(m, math.NaN(), 1.0)
	if len(m.msgs) != 3 {
		t.Fatalf("got %d failures, want 3: %v", len(m.msgs), m.msgs)
	}
}

func TestNotEqual(t *testing.T) {
	NotEqual(t, 1, 2)
	NotEqual(t, math.NaN(), 1.0)
	m := &mockT{}
	NotEqual(m, math.NaN(), math.NaN())
	NotEqual(m, 1, 1)
	if len(m.msgs) != 2 {
		t.Fatalf("got %d failures, want 2", len(m.msgs))
	}
}

func TestArrayEqual(t *testing.T) {
	// pass: identical and NaN positions
	ArrayEqual(t, arr([]float64{1, math.NaN(), 3}), arr([]float64{1, math.NaN(), 3}))
	// 2D shape
	ArrayEqual(t, arr([]float64{1, 2, 3, 4}, 2, 2), arr([]float64{1, 2, 3, 4}, 2, 2))

	// fail: element mismatch — check numpy-style message
	m := &mockT{}
	ArrayEqual(m, arr([]float64{1, 2, 3, 4, 5, 6}, 2, 3), arr([]float64{1, 2, 0, 4, 5, 0}, 2, 3))
	if len(m.msgs) != 1 {
		t.Fatalf("got %d failures, want 1", len(m.msgs))
	}
	msg := m.msgs[0]
	for _, want := range []string{
		"Arrays are not equal",
		"Mismatched elements: 2 / 6 (33.3%)",
		"Max absolute difference: 6.",
		" x: array([1., 2., 3., 4., 5., 6.])",
		" y: array([1., 2., 0., 4., 5., 0.])",
		"x[2]=3. y[2]=0.",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("message %q missing %q", msg, want)
		}
	}

	// fail: shape mismatch
	m2 := &mockT{}
	ArrayEqual(m2, arr([]float64{1, 2, 3, 4, 5, 6}, 2, 3), arr([]float64{1, 2, 3, 4, 5, 6}, 3, 2))
	if len(m2.msgs) != 1 || !strings.Contains(m2.msgs[0], "Shapes: (2,3) and (3,2)") {
		t.Errorf("shape-mismatch message wrong: %v", m2.msgs)
	}
}

func TestArrayNotEqual(t *testing.T) {
	ArrayNotEqual(t, arr([]float64{1, 2}), arr([]float64{1, 3}))
	ArrayNotEqual(t, arr([]float64{1, 2}), arr([]float64{1, 2}, 2, 1)) // shapes differ
	m := &mockT{}
	ArrayNotEqual(m, arr([]float64{1, math.NaN()}), arr([]float64{1, math.NaN()}))
	if len(m.msgs) != 1 {
		t.Errorf("NaN-matching arrays should fail ArrayNotEqual, got %v", m.msgs)
	}
}

func TestArrayAlmostEqual(t *testing.T) {
	// 0.5*10^-4 <= 1.000001? no: 1.000001 vs 1.0 diff 1e-6 < 5e-5 → pass at decimal=4
	ArrayAlmostEqual(t, arr([]float64{1.000001, 2.0}), arr([]float64{1.0, 2.0}), 4)
	// boundary: diff == 0.5*10^-decimal is NOT almost equal (strict <)
	m := &mockT{}
	ArrayAlmostEqual(m, arr([]float64{1.05}), arr([]float64{1.0}), 1) // diff 0.05, lim 0.05
	if len(m.msgs) != 1 || !strings.Contains(m.msgs[0], "Arrays are not almost equal to 1 decimals") {
		t.Errorf("decimal boundary should fail: %v", m.msgs)
	}
	// decimal=10: 1e-10 diff fails, 1e-11 passes
	ArrayAlmostEqual(t, arr([]float64{1 + 1e-11}), arr([]float64{1}), 10)
	m2 := &mockT{}
	ArrayAlmostEqual(m2, arr([]float64{1 + 1e-9}), arr([]float64{1}), 10)
	if len(m2.msgs) != 1 {
		t.Errorf("1e-9 diff should fail at decimal=10: %v", m2.msgs)
	}
	// NaN mismatches (equal_nan=False default)
	m3 := &mockT{}
	ArrayAlmostEqual(m3, arr([]float64{math.NaN()}), arr([]float64{math.NaN()}), 6)
	if len(m3.msgs) != 1 {
		t.Errorf("NaN vs NaN should fail ArrayAlmostEqual: %v", m3.msgs)
	}
}

func TestAllClose(t *testing.T) {
	// numpy spot-checks (python):
	//   allclose([1.0, 2.0], [1.001, 2.001], rtol=1e-2, atol=0) → True
	AllClose(t, arr([]float64{1.0, 2.0}), arr([]float64{1.001, 2.001}), 1e-2, 0)
	//   allclose([1.0, 2.0], [1.001, 2.001], rtol=1e-4, atol=0) → False
	m := &mockT{}
	AllClose(m, arr([]float64{1.0, 2.0}), arr([]float64{1.001, 2.001}), 1e-4, 0)
	if len(m.msgs) != 1 || !strings.Contains(m.msgs[0], "Not equal to tolerance rtol=0.0001, atol=0") {
		t.Errorf("expected failure with tolerance header, got %v", m.msgs)
	}
	//   allclose([1e100, 1e-7], [1.00001e100, 1e-8], rtol=0, atol=0) → False (python doc)
	//   allclose([1e100, 1e-8], [1.00001e100, 1e-9], rtol=1e-5, atol=5e-324) → True (python doc)
	// spot-check 3: atol dominates for values near zero
	//   allclose([0.0], [1e-9], rtol=0, atol=1e-8) → True
	AllClose(t, arr([]float64{0.0}), arr([]float64{1e-9}), 0, 1e-8)
	//   allclose([0.0], [1e-7], rtol=0, atol=1e-8) → False
	m2 := &mockT{}
	AllClose(m2, arr([]float64{0.0}), arr([]float64{1e-7}), 0, 1e-8)
	if len(m2.msgs) != 1 {
		t.Errorf("atol violation should fail: %v", m2.msgs)
	}
	// NaN fails by default
	m3 := &mockT{}
	AllClose(m3, arr([]float64{math.NaN()}), arr([]float64{math.NaN()}), 1e-8, 1e-8)
	if len(m3.msgs) != 1 {
		t.Errorf("NaN should fail AllClose: %v", m3.msgs)
	}
}

func TestAllCloseNan(t *testing.T) {
	AllCloseNan(t, arr([]float64{math.NaN(), 1.0}), arr([]float64{math.NaN(), 1.0}), 1e-8, 1e-8, true)
	m := &mockT{}
	AllCloseNan(m, arr([]float64{math.NaN()}), arr([]float64{math.NaN()}), 1e-8, 1e-8, false)
	if len(m.msgs) != 1 {
		t.Errorf("equalNaN=false should fail on NaN: %v", m.msgs)
	}
}

func TestArrayLess(t *testing.T) {
	ArrayLess(t, arr([]float64{1, 2, 3}), arr([]float64{2, 3, 4}))
	m := &mockT{}
	ArrayLess(m, arr([]float64{1, 5}), arr([]float64{2, 3}))
	if len(m.msgs) != 1 || !strings.Contains(m.msgs[0], "Max absolute difference: 2.") {
		t.Errorf("ArrayLess failure message: %v", m.msgs)
	}
	// NaN fails
	m2 := &mockT{}
	ArrayLess(m2, arr([]float64{math.NaN()}), arr([]float64{1}))
	if len(m2.msgs) != 1 {
		t.Errorf("NaN should fail ArrayLess: %v", m2.msgs)
	}
}

func TestArrayLessEqual(t *testing.T) {
	ArrayLessEqual(t, arr([]float64{1, 3}), arr([]float64{2, 3}))
	m := &mockT{}
	ArrayLessEqual(m, arr([]float64{3}), arr([]float64{2}))
	if len(m.msgs) != 1 {
		t.Errorf("ArrayLessEqual failure: %v", m.msgs)
	}
}

func TestRaises(t *testing.T) {
	Raises(t, "shape", func() { numgo.FromSlice([]float64{1, 2, 3}, 2, 2) })
	m := &mockT{}
	Raises(m, "shape", func() {}) // no panic
	if len(m.msgs) != 1 || !strings.Contains(m.msgs[0], "did not panic") {
		t.Errorf("no-panic case: %v", m.msgs)
	}
	m2 := &mockT{}
	Raises(m2, "xyzzy", func() { panic("reshape: incompatible shapes") })
	if len(m2.msgs) != 1 || !strings.Contains(m2.msgs[0], "does not contain") {
		t.Errorf("wrong-panic case: %v", m2.msgs)
	}
}

func TestShape(t *testing.T) {
	Shape(t, arr([]float64{1, 2, 3, 4, 5, 6}, 2, 3), 2, 3)
	m := &mockT{}
	Shape(m, arr([]float64{1, 2, 3}, 3), 2, 3)
	if len(m.msgs) != 1 || !strings.Contains(m.msgs[0], "shape = [3], want [2 3]") {
		t.Errorf("Shape failure message: %v", m.msgs)
	}
}

func TestFloat32(t *testing.T) {
	ArrayEqual(t, arr([]float32{1, 2}), arr([]float32{1, 2}))
	ArrayAlmostEqual(t, arr([]float32{1.0001}), arr([]float32{1.0}), 3)
	AllClose(t, arr([]float32{1.0}), arr([]float32{1.001}), 1e-2, 0)
	ArrayNotEqual(t, arr([]float32{1}), arr([]float32{2}))
	ArrayLess(t, arr([]float32{1}), arr([]float32{2}))
	ArrayLessEqual(t, arr([]float32{1}), arr([]float32{2}))
	Shape(t, arr([]float32{1, 2}, 2), 2)

	m := &mockT{}
	ArrayEqual(m, arr([]float32{1, 2}), arr([]float32{1, 3}))
	if len(m.msgs) != 1 {
		t.Errorf("float32 failure not reported: %v", m.msgs)
	}
}
