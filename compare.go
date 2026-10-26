package numgo

import (
	"math"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// This file holds the element-wise comparison family and the float
// predicates. The library has no boolean Array type (Number constrains
// elements to float/complex), so every operation here returns a flattened
// C-order []bool of the broadcast size, matching the []bool convention of
// SetMask, MaskSelect and IsIn. Complex comparisons follow numpy's
// lexicographic order (real part first, imaginary part as tie-break, the
// same order cmpNumber uses for sorting); equality compares both
// components. Any NaN operand makes an ordering comparison false by IEEE
// semantics (NotEqual stays true).

// comparePair broadcasts a and b and returns the flattened C-order mask of
// positions where cmp(a[i], b[i]) holds.
func comparePair[T Number](a, b *Array[T], cmp func(x, y T) bool) []bool {
	la, lb, shape := broadcastPair(a, b)
	out := make([]bool, nditer.Size(shape))
	if len(out) == 0 {
		return out
	}
	coords := make([]int, len(shape))
	for k := range out {
		out[k] = cmp(a.data[nditer.Index(la, coords)], b.data[nditer.Index(lb, coords)])
		advance(coords, shape)
	}
	return out
}

// gtLex reports x > y lexicographically (real, then imaginary), which is
// numpy's complex ordering and reduces to the plain comparison for reals.
func gtLex[T Number](x, y T) bool {
	xr, xi := parts(x)
	yr, yi := parts(y)
	return xr > yr || (xr == yr && xi > yi)
}

// elemBools walks a in C order and returns f's real/imaginary verdict for
// every element as a flattened []bool.
func elemBools[T Number](a *Array[T], f func(re, im float64) bool) []bool {
	out := make([]bool, a.Size())
	if len(out) == 0 {
		return out
	}
	coords := make([]int, len(a.shape))
	for k := range out {
		re, im := parts(a.data[nditer.Index(a.layout(), coords)])
		out[k] = f(re, im)
		advance(coords, a.shape)
	}
	return out
}

// Greater returns a flattened C-order mask where a exceeds b (numpy a > b;
// complex operands ordered lexicographically, real then imaginary).
func (a *Array[T]) Greater(b *Array[T]) []bool {
	return comparePair(a, b, gtLex[T])
}

// GreaterValue is Greater against a scalar (numpy a > v).
func (a *Array[T]) GreaterValue(v T) []bool {
	return elemBools(a, func(re, im float64) bool {
		vr, vi := parts(v)
		return re > vr || (re == vr && im > vi)
	})
}

// GreaterEqual returns a flattened C-order mask where a is at least b
// (numpy a >= b; lexicographic for complex).
func (a *Array[T]) GreaterEqual(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y T) bool { return gtLex(x, y) || eqParts(x, y) })
}

// GreaterEqualValue is GreaterEqual against a scalar (numpy a >= v).
func (a *Array[T]) GreaterEqualValue(v T) []bool {
	return elemBools(a, func(re, im float64) bool {
		vr, vi := parts(v)
		return re > vr || (re == vr && im >= vi)
	})
}

// Less returns a flattened C-order mask where a is below b (numpy a < b;
// lexicographic for complex).
func (a *Array[T]) Less(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y T) bool { return gtLex(y, x) })
}

// LessValue is Less against a scalar (numpy a < v).
func (a *Array[T]) LessValue(v T) []bool {
	return elemBools(a, func(re, im float64) bool {
		vr, vi := parts(v)
		return re < vr || (re == vr && im < vi)
	})
}

// LessEqual returns a flattened C-order mask where a is at most b (numpy
// a <= b; lexicographic for complex).
func (a *Array[T]) LessEqual(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y T) bool { return gtLex(y, x) || eqParts(x, y) })
}

// LessEqualValue is LessEqual against a scalar (numpy a <= v).
func (a *Array[T]) LessEqualValue(v T) []bool {
	return elemBools(a, func(re, im float64) bool {
		vr, vi := parts(v)
		return re < vr || (re == vr && im <= vi)
	})
}

// eqParts reports whether x and y agree in both components (numpy's
// elementwise complex equality; NaN never equals anything).
func eqParts[T Number](x, y T) bool {
	xr, xi := parts(x)
	yr, yi := parts(y)
	return xr == yr && xi == yi
}

// Equal returns a flattened C-order mask where the elements agree in both
// components (numpy a == b; NaN never equals anything).
func (a *Array[T]) Equal(b *Array[T]) []bool {
	return comparePair(a, b, eqParts[T])
}

// EqualValue is Equal against a scalar (numpy a == v).
func (a *Array[T]) EqualValue(v T) []bool {
	vr, vi := parts(v)
	return elemBools(a, func(re, im float64) bool { return re == vr && im == vi })
}

// NotEqual returns a flattened C-order mask where the elements differ in
// either component (numpy a != b; NaN differs from everything).
func (a *Array[T]) NotEqual(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y T) bool { return !eqParts(x, y) })
}

// NotEqualValue is NotEqual against a scalar (numpy a != v).
func (a *Array[T]) NotEqualValue(v T) []bool {
	vr, vi := parts(v)
	return elemBools(a, func(re, im float64) bool { return !(re == vr && im == vi) })
}

// IsNaN returns a flattened C-order mask of NaN elements (numpy np.isnan).
// A complex element is NaN when either component is.
func (a *Array[T]) IsNaN() []bool {
	return elemBools(a, func(re, im float64) bool {
		return math.IsNaN(re) || math.IsNaN(im)
	})
}

// IsInf returns a flattened C-order mask of infinite elements, either sign
// (numpy np.isinf). A complex element is infinite when either component is.
func (a *Array[T]) IsInf() []bool {
	return elemBools(a, func(re, im float64) bool {
		return math.IsInf(re, 0) || math.IsInf(im, 0)
	})
}

// IsFinite returns a flattened C-order mask of finite elements: neither NaN
// nor infinite in either component (numpy np.isfinite).
func (a *Array[T]) IsFinite() []bool {
	return elemBools(a, func(re, im float64) bool {
		return !math.IsNaN(re) && !math.IsInf(re, 0) && !math.IsNaN(im) && !math.IsInf(im, 0)
	})
}

// IsPosInf returns a flattened C-order mask of +Inf elements (real part or,
// for complex, either component).
func (a *Array[T]) IsPosInf() []bool {
	return elemBools(a, func(re, im float64) bool {
		return math.IsInf(re, 1) || math.IsInf(im, 1)
	})
}

// IsNegInf returns a flattened C-order mask of -Inf elements (real part or,
// for complex, either component).
func (a *Array[T]) IsNegInf() []bool {
	return elemBools(a, func(re, im float64) bool {
		return math.IsInf(re, -1) || math.IsInf(im, -1)
	})
}
