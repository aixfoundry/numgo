package numgo

import "math"

// PolyFit 多项式拟合
//
// numpy: [numpy.polyfit]
//
// 输入:
//
//	[ xy ] (n, 2)维矩阵
//	m: 多项式次数，m < N-1
//
// 输出: sol 解
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
	return Lstsq(a, xy.Of(1, 1))
}

// Poly1D 多项式的值
//
// numpy: [numpy.poly1d]
//
// 输入:
//
//	b: 常数列向量(一维)
//	x: x值
//
// 输出: y 解
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

// PolyVal 多项式的值
//
// numpy: [numpy.polyval]
//
// 输入:
//
//	b: 常数列向量(一维)
//	x: x值
//
// 输出: y 解
//
// [numpy.polyval]: https://numpy.org/doc/stable/reference/generated/numpy.polyval.html
func PolyVal(b *Array, x *Array) (y *Array) {
	if b.r != 1 {
		return
	}
	y = ZeroArray(x.r, x.c)
	for ir := range x.r {
		for ic := range x.c {
			y.Set(ir, ic, Poly1D(b, x.At(ir, ic)))
		}
	}
	return
}
