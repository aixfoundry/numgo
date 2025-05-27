package numgo

import "math"

// PolyFit performs polynomial fitting.
//
// numpy: [numpy.polyfit]
//
// Inputs:
//
//	xy: an (n, 2) matrix where each row is a (x, y) pair.
//	m: the degree of the polynomial, m < N-1.
//
// Output:
//
//	sol: the coefficients of the polynomial that fits the data.
//
// Example:
//
//	xy := NewArray(2, 2, []float64{1, 2, 3, 2, 3, 5})
//	sol := PolyFit(xy, 1)
//	fmt.Println(sol) // Output will be the coefficients of the best fit line
//
// [numpy.polyfit]: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
func PolyFit(xy *Array, m int) (sol *Array) {
	n := xy.r
	a := ZeroArray(n, m+1)
	for i := range n {
		for j := range m + 1 {
			a.Set(i, j, math.Pow(xy.At(i, 0), float64(j)))
		}
	}
	return Lstsq(a, xy.Of(1, 0))
}

// Poly1D evaluates a polynomial at a given point.
//
// numpy: [numpy.poly1d]
//
// Inputs:
//
//	b: a 1D array of coefficients.
//	x: the value at which to evaluate the polynomial.
//
// Output:
//
//	y: the result of evaluating the polynomial at x.
//
// Example:
//
//	b := NewArray(1, 1, []float64{1, 2}) // Represents the polynomial 1 + 2x
//	y := Poly1D(b, 3)
//	fmt.Println(y) // Output will be 7
//
// [numpy.poly1d]: https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html
func Poly1D(b *Array, x float64) (y float64) {
	if b.r != 1 {
		return
	}
	for ic := range b.c {
		y += b.At(0, ic) * math.Pow(x, float64(ic))
	}
	return
}

// PolyVal evaluates a polynomial at multiple points.
//
// numpy: [numpy.polyval]
//
// Inputs:
//
//	b: a 1D array of coefficients.
//	x: an array of points at which to evaluate the polynomial.
//
// Output:
//
//	y: an array of results from evaluating the polynomial at each point in x.
//
// Example:
//
//	b := NewArray(1, 2, []float64{1, 2}) // Represents the polynomial 1 + 2x
//	x := NewArray(2, 1, []float64{1, 2})
//	y := PolyVal(b, x)
//	fmt.Println(y) // Output will be an array with values [3, 5]
//
// [numpy.polyval]: https://numpy.org/doc/stable/reference/generated/numpy.polyval.html
func PolyVal(b *Array, x *Array) (y *Array) {
	if b.r != 1 {
		return
	}
	y = ZeroArray(x.r, x.c)
	for k, v := range x.All() {
		y.Set(k[0], k[1], Poly1D(b, v))
	}
	return
}
