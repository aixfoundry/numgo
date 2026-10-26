package numgo

import (
	"math"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// IsClose reports, elementwise over the broadcast of a and b, whether
// |x - y| <= atol + rtol*|y| (b is the reference, as in numpy). Complex
// elements compare by magnitude; NaN is not close to anything, including
// itself (numpy's default equal_nan=false). The result is a flat C-order
// []bool of length equal to the broadcast size.
func IsClose[T Number](a, b *Array[T], rtol, atol float64) []bool {
	la, lb, shape := broadcastPair(a, b)
	out := make([]bool, nditer.Size(shape))
	coords := make([]int, len(shape))
	for i := range out {
		out[i] = isCloseScalar(a.data[nditer.Index(la, coords)], b.data[nditer.Index(lb, coords)], rtol, atol)
		advance(coords, shape)
	}
	return out
}

func isCloseScalar[T Number](x, y T, rtol, atol float64) bool {
	xr, xi := parts(x)
	yr, yi := parts(y)
	if math.IsNaN(xr) || math.IsNaN(xi) || math.IsNaN(yr) || math.IsNaN(yi) {
		return false
	}
	diff := math.Hypot(xr-yr, xi-yi)
	infX := math.IsInf(xr, 0) || math.IsInf(xi, 0)
	infY := math.IsInf(yr, 0) || math.IsInf(yi, 0)
	if infX || infY {
		// Inf is close only to itself (same sign in every infinite component).
		return infX == infY && xr == yr && xi == yi
	}
	return diff <= atol+rtol*math.Hypot(yr, yi)
}
