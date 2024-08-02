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
