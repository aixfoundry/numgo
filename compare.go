package numgo

import (
	"math"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// This file holds the element-wise comparison family and the float
// predicates. The library has no boolean Array type (Number constrains
// elements to float/complex), so every operation here returns a flattened
// C-order []bool of the broadcast size, matching the []bool convention of
// SetMask, MaskSelect and IsIn. Complex elements compare by real part (the
// package-wide comparison convention); any NaN operand makes a comparison
// false by IEEE semantics.

// comparePair broadcasts a and b and returns the flattened C-order mask of
// positions where cmp(real(a[i]), real(b[i])) holds.
func comparePair[T Number](a, b *Array[T], cmp func(ra, rb float64) bool) []bool {
	la, lb, shape := broadcastPair(a, b)
	out := make([]bool, nditer.Size(shape))
	if len(out) == 0 {
		return out
	}
	coords := make([]int, len(shape))
	for k := range out {
		x := a.data[nditer.Index(la, coords)]
		y := b.data[nditer.Index(lb, coords)]
		out[k] = cmp(mustFloat(x), mustFloat(y))
		advance(coords, shape)
	}
	return out
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

// Greater returns a flattened C-order mask where a's real part exceeds b's
// (numpy a > b, complex operands compared by real part).
func (a *Array[T]) Greater(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y float64) bool { return x > y })
}

// GreaterValue is Greater against a scalar (numpy a > v).
func (a *Array[T]) GreaterValue(v T) []bool {
	rv, _ := parts(v)
	return elemBools(a, func(re, _ float64) bool { return re > rv })
}

// GreaterEqual returns a flattened C-order mask where a's real part is at
// least b's (numpy a >= b).
func (a *Array[T]) GreaterEqual(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y float64) bool { return x >= y })
}

// GreaterEqualValue is GreaterEqual against a scalar (numpy a >= v).
func (a *Array[T]) GreaterEqualValue(v T) []bool {
	rv, _ := parts(v)
	return elemBools(a, func(re, _ float64) bool { return re >= rv })
}

// Less returns a flattened C-order mask where a's real part is below b's
// (numpy a < b).
func (a *Array[T]) Less(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y float64) bool { return x < y })
}

// LessValue is Less against a scalar (numpy a < v).
func (a *Array[T]) LessValue(v T) []bool {
	rv, _ := parts(v)
	return elemBools(a, func(re, _ float64) bool { return re < rv })
}

// LessEqual returns a flattened C-order mask where a's real part is at most
// b's (numpy a <= b).
func (a *Array[T]) LessEqual(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y float64) bool { return x <= y })
}

// LessEqualValue is LessEqual against a scalar (numpy a <= v).
func (a *Array[T]) LessEqualValue(v T) []bool {
	rv, _ := parts(v)
	return elemBools(a, func(re, _ float64) bool { return re <= rv })
}

// Equal returns a flattened C-order mask where the real parts agree
// (numpy a == b; NaN never equals anything).
func (a *Array[T]) Equal(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y float64) bool { return x == y })
}

// EqualValue is Equal against a scalar (numpy a == v).
func (a *Array[T]) EqualValue(v T) []bool {
	rv, _ := parts(v)
	return elemBools(a, func(re, _ float64) bool { return re == rv })
}

// NotEqual returns a flattened C-order mask where the real parts differ
// (numpy a != b; NaN differs from everything).
func (a *Array[T]) NotEqual(b *Array[T]) []bool {
	return comparePair(a, b, func(x, y float64) bool { return x != y })
}

// NotEqualValue is NotEqual against a scalar (numpy a != v).
func (a *Array[T]) NotEqualValue(v T) []bool {
	rv, _ := parts(v)
	return elemBools(a, func(re, _ float64) bool { return re != rv })
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
