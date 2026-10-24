// Package assert provides numpy.testing-style assertions for numgo arrays,
// for use in downstream tests.
//
// numpy.testing mapping:
//
//	assert_equal              Equal
//	assert_not_equal          NotEqual
//	assert_array_equal        ArrayEqual            (NaN == NaN)
//	assert_array_not_equal    ArrayNotEqual
//	assert_array_almost_equal ArrayAlmostEqual      (|a-b| < 0.5·10^-decimal)
//	assert_allclose           AllClose / AllCloseNan
//	assert_array_less         ArrayLess / ArrayLessEqual
//	assert_raises             Raises
//	(inline shape checks)     Shape
//	assert_array_max_ulp      not implemented
//
// All assertion functions accept testing.TB, so a *testing.T (or *testing.B)
// is passed directly; the TB interface also enables failure-message tests
// with a mock recorder. Failure paths report via t.Errorf (formatted numpy
// style) and return, never Fatal.
package assert

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/aixfoundry/numgo"
)

// Equal asserts deep equality of scalars or slices via reflect.DeepEqual,
// with the special case NaN == NaN (IEEE says false; numpy's assert_equal
// also treats NaNs as equal). Different types are never equal.
func Equal(t testing.TB, a, b any) {
	t.Helper()
	if !eqValue(reflect.ValueOf(a), reflect.ValueOf(b)) {
		t.Errorf("not equal:\na: %v\nb: %v", a, b)
	}
}

// NotEqual is the negation of Equal (same NaN == NaN semantics).
func NotEqual(t testing.TB, a, b any) {
	t.Helper()
	if eqValue(reflect.ValueOf(a), reflect.ValueOf(b)) {
		t.Errorf("equal:\na: %v\nb: %v", a, b)
	}
}

// ArrayEqual asserts got and want have the same shape and equal elements
// elementwise, with NaN == NaN (numpy assert_array_equal / equal_nan=True).
// Failure reports shape, mismatch count, max abs/rel differences and the
// first mismatching values.
func ArrayEqual[T numgo.Number](t testing.TB, got, want *numgo.Array[T]) {
	t.Helper()
	failDiffs(t, "Arrays are not equal", got, want, eqNaN[T])
}

// ArrayNotEqual asserts the arrays differ (numpy assert_array_not_equal):
// pass when shapes differ or any element differs (NaN != NaN except under
// ArrayEqual's special case, i.e. NaN vs NaN counts as equal here).
func ArrayNotEqual[T numgo.Number](t testing.TB, got, want *numgo.Array[T]) {
	t.Helper()
	if !slices.Equal(got.Shape(), want.Shape()) {
		return // differing shape counts as not equal
	}
	if scan(got, want, eqNaN[T]).mismatched == 0 {
		t.Errorf("arrays are equal:\n x: %s\n y: %s", repr(got), repr(want))
	}
}

// ArrayAlmostEqual asserts every |got-want| < 0.5·10^-decimal (numpy
// assert_array_almost_equal semantics). NaN anywhere mismatches (unlike
// ArrayEqual; pass NaN positions only via ArrayEqual or AllCloseNan).
func ArrayAlmostEqual[T numgo.Number](t testing.TB, got, want *numgo.Array[T], decimal int) {
	t.Helper()
	lim := 0.5 * math.Pow10(-decimal)
	failDiffs(t, fmt.Sprintf("Arrays are not almost equal to %d decimals", decimal), got, want,
		func(g, w T) bool { return !isNan(g) && !isNan(w) && absDiff(g, w) < lim })
}

// AllClose asserts |got-want| <= atol + rtol*|want| for every element
// (numpy assert_allclose). Note rtol scales with want — the DESIRED
// argument, matching numpy's |b|. Same-sign infinities are equal; any NaN
// mismatches (use AllCloseNan for equal_nan=true).
func AllClose[T numgo.Number](t testing.TB, got, want *numgo.Array[T], rtol, atol float64) {
	t.Helper()
	AllCloseNan(t, got, want, rtol, atol, false)
}

// AllCloseNan is AllClose with explicit NaN handling: when equalNaN is true,
// NaN == NaN at the same position (numpy allclose equal_nan=True).
func AllCloseNan[T numgo.Number](t testing.TB, got, want *numgo.Array[T], rtol, atol float64, equalNaN bool) {
	t.Helper()
	failDiffs(t, fmt.Sprintf("Not equal to tolerance rtol=%g, atol=%g", rtol, atol), got, want,
		func(g, w T) bool {
			if g == w {
				return true
			}
			if equalNaN && isNan(g) && isNan(w) {
				return true
			}
			return absDiff(g, w) <= atol+rtol*math.Abs(absT(w))
		})
}

// ArrayLess asserts got < want strictly at every position; NaN mismatches
// (numpy assert_array_less). Complex element types are not ordered and panic.
func ArrayLess[T numgo.Number](t testing.TB, got, want *numgo.Array[T]) {
	t.Helper()
	failDiffs(t, "Arrays are not less-ordered", got, want, lessE[T])
}

// ArrayLessEqual asserts got <= want at every position (no direct numpy
// analog; assert_array_less with strict=False). NaN mismatches.
func ArrayLessEqual[T numgo.Number](t testing.TB, got, want *numgo.Array[T]) {
	t.Helper()
	failDiffs(t, "Arrays are not less-or-equal", got, want, lessEqE[T])
}

// Raises asserts fn panics with a message containing substr (numpy
// assert_raises analog for this package's panic-based errors).
func Raises(t testing.TB, substr string, fn func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("did not panic; want panic containing %q", substr)
			return
		}
		if msg := fmt.Sprint(r); !strings.Contains(msg, substr) {
			t.Errorf("panic message %q does not contain %q", msg, substr)
		}
	}()
	fn()
}

// Shape asserts the array's shape equals want.
func Shape[T numgo.Number](t testing.TB, a *numgo.Array[T], want ...int) {
	t.Helper()
	if got := a.Shape(); !slices.Equal(got, want) {
		t.Errorf("shape = %v, want %v", got, want)
	}
}

// ---- comparison helpers ----

func eqNaN[T numgo.Number](g, w T) bool { return g == w || (isNan(g) && isNan(w)) }

func lessE[T numgo.Number](g, w T) bool {
	switch a := any(g).(type) {
	case float64:
		return a < any(w).(float64)
	case float32:
		return a < any(w).(float32)
	}
	panic("ArrayLess: complex element types are not ordered")
}

func lessEqE[T numgo.Number](g, w T) bool {
	switch a := any(g).(type) {
	case float64:
		return a <= any(w).(float64)
	case float32:
		return a <= any(w).(float32)
	}
	panic("ArrayLessEqual: complex element types are not ordered")
}

// absDiff returns |g-w| as float64 with NaN semantics: equal values (incl.
// same-sign inf) → 0, both NaN → 0, one NaN → +Inf.
func absDiff[T numgo.Number](g, w T) float64 {
	if g == w {
		return 0
	}
	if isNan(g) && isNan(w) {
		return 0
	}
	if isNan(g) || isNan(w) {
		return math.Inf(1)
	}
	switch a := any(g).(type) {
	case float64:
		return math.Abs(a - any(w).(float64))
	case float32:
		return math.Abs(float64(a - any(w).(float32)))
	case complex64:
		return cabs(complex128(a) - complex128(any(w).(complex64)))
	case complex128:
		return cabs(a - any(w).(complex128))
	}
	return math.Inf(1)
}

func absT[T numgo.Number](v T) float64 {
	switch a := any(v).(type) {
	case float64:
		return math.Abs(a)
	case float32:
		return math.Abs(float64(a))
	case complex64:
		return cabs(complex128(a))
	case complex128:
		return cabs(a)
	}
	return math.Inf(1)
}

func relDiff(d, denom float64) float64 {
	if denom == 0 {
		if d == 0 {
			return 0
		}
		return math.Inf(1)
	}
	return d / denom
}

func isNan[T numgo.Number](v T) bool {
	switch a := any(v).(type) {
	case float64:
		return math.IsNaN(a)
	case float32:
		return math.IsNaN(float64(a))
	case complex64:
		return isNanC(complex128(a))
	case complex128:
		return isNanC(a)
	}
	return false
}

func isNanC(c complex128) bool { return math.IsNaN(real(c)) || math.IsNaN(imag(c)) }

func cabs(c complex128) float64 { return math.Hypot(real(c), imag(c)) }

// eqValue is Equal's walker: NaN-aware for float/complex, recursive for
// slices/arrays/pointers/interfaces, reflect.DeepEqual otherwise (covers
// maps and structs).
func eqValue(a, b reflect.Value) bool {
	if !a.IsValid() || !b.IsValid() {
		return a.IsValid() == b.IsValid()
	}
	if a.Type() != b.Type() {
		return false
	}
	switch a.Kind() {
	case reflect.Float32, reflect.Float64:
		x, y := a.Float(), b.Float()
		return x == y || (math.IsNaN(x) && math.IsNaN(y))
	case reflect.Complex64, reflect.Complex128:
		x, y := a.Complex(), b.Complex()
		return x == y || (isNanC(x) && isNanC(y))
	case reflect.Slice, reflect.Array:
		if a.Len() != b.Len() {
			return false
		}
		for i := 0; i < a.Len(); i++ {
			if !eqValue(a.Index(i), b.Index(i)) {
				return false
			}
		}
		return true
	case reflect.Pointer:
		if a.IsNil() || b.IsNil() {
			return a.IsNil() == b.IsNil()
		}
		return eqValue(a.Elem(), b.Elem())
	case reflect.Interface:
		return eqValue(a.Elem(), b.Elem())
	default:
		return reflect.DeepEqual(a.Interface(), b.Interface())
	}
}

// ---- failure message ----

type scanResult struct {
	mismatched, total int
	maxAbs, maxRel    float64
	samples           []string
}

// scan walks C-order elements, applying ok and accumulating the numpy-style
// mismatch statistics.
func scan[T numgo.Number](got, want *numgo.Array[T], ok func(g, w T) bool) *scanResult {
	gd, wd := got.Ravel().Data(), want.Ravel().Data()
	r := &scanResult{total: len(gd)}
	for i := range gd {
		g, w := gd[i], wd[i]
		d := absDiff(g, w)
		if d > r.maxAbs {
			r.maxAbs = d
		}
		if rel := relDiff(d, absT(w)); rel > r.maxRel {
			r.maxRel = rel
		}
		if !ok(g, w) {
			r.mismatched++
			if len(r.samples) < 5 {
				r.samples = append(r.samples,
					fmt.Sprintf("x[%d]=%s y[%d]=%s", i, fnumT(g), i, fnumT(w)))
			}
		}
	}
	return r
}

// failDiffs reports one formatted numpy-style t.Errorf when got/want differ
// under ok; shape mismatches short-circuit to a shapes + arrays report.
func failDiffs[T numgo.Number](t testing.TB, header string, got, want *numgo.Array[T], ok func(g, w T) bool) {
	t.Helper()
	if !slices.Equal(got.Shape(), want.Shape()) {
		t.Errorf("%s\n\nShapes: %s and %s\n x: %s\n y: %s",
			header, fshape(got.Shape()), fshape(want.Shape()), repr(got), repr(want))
		return
	}
	r := scan(got, want, ok)
	if r.mismatched == 0 {
		return
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s\n\nMismatched elements: %d / %d (%.1f%%)\nMax absolute difference: %s\nMax relative difference: %s",
		header, r.mismatched, r.total, 100*float64(r.mismatched)/float64(r.total), fnum(r.maxAbs), fnum(r.maxRel))
	if len(r.samples) > 0 {
		b.WriteString("\nMismatched values (first 5):")
		for _, s := range r.samples {
			b.WriteString("\n  " + s)
		}
	}
	fmt.Fprintf(&b, "\n x: %s\n y: %s", repr(got), repr(want))
	t.Errorf("%s", b.String())
}

// repr formats an array C-order flattened, numpy array(...) style, capped
// at 100 elements.
func repr[T numgo.Number](a *numgo.Array[T]) string {
	d := a.Ravel().Data()
	n := min(len(d), 100)
	parts := make([]string, n)
	for i := range n {
		parts[i] = fnumT(d[i])
	}
	if n < len(d) {
		parts = append(parts, "...")
	}
	return "array([" + strings.Join(parts, ", ") + "])"
}

func fshape(s []int) string {
	p := make([]string, len(s))
	for i, d := range s {
		p[i] = strconv.Itoa(d)
	}
	return "(" + strings.Join(p, ",") + ")"
}

func fnumT[T numgo.Number](v T) string {
	switch a := any(v).(type) {
	case float64:
		return fnum(a)
	case float32:
		return fnum(float64(a))
	case complex64:
		return cnum(complex128(a))
	case complex128:
		return cnum(a)
	}
	return "?"
}

// fnum renders a float numpy-repr style: always a decimal point ("3.", not
// "3") unless exponential or non-finite.
func fnum(x float64) string {
	s := strconv.FormatFloat(x, 'g', -1, 64)
	if !strings.ContainsAny(s, ".enN") {
		s += "."
	}
	return s
}

func cnum(c complex128) string {
	im := imag(c)
	sign := "+"
	if im < 0 || (im == 0 && math.Signbit(im)) {
		sign = "-"
		im = -im
	}
	return "(" + fnum(real(c)) + sign + fnum(im) + "j)"
}
