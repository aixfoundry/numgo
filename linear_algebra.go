package numgo

import (
	"math"
)

// EMCP 列主元消去(高斯列主元消去法)
//
// 输入: [ A | b ] 增广矩阵, A是方形矩阵
//
//	A : 线性代数方程组(A x = b)的系数矩阵
//	b : 线性代数方程组(A x = b)的常数列向量
//
// 输出: sol 解
func EMCP(x *Array) (sol *Array) {
	if x.c-1 != x.r {
		return
	}
	n := x.r // n为方程数
	// 求上三角
	for i := range n { // 遍历所有行
		// 求第i列的主元素并调整顺序
		cols := []float64{}
		for ic := i; ic < n; ic++ { // 遍历未调整顺序的行
			cols = append(cols, x.At(ic, i)) // 取[ic:n][i]数据
		}
		acol := NewArray(1, len(cols), cols)
		_, _, c := acol.MaxAbs()
		if c+i != i {
			x.Swap(c+i, i, 0)
		}
		for j := i + 1; j < n; j++ {
			mul := x.At(j, i) / x.At(i, i)
			for k := i; k < x.c; k++ {
				x.Set(j, k, x.At(j, k)-x.At(i, k)*mul)
			}
		}
	}
	buf := make([]float64, n)
	buf[n-1] = x.At(n-1, x.c-1) / x.At(n-1, n-1)
	for i := n - 2; i >= 0; i-- {
		t := 0.0
		for j := i + 1; j < n; j++ {
			t += x.At(i, j) * buf[j]
		}
		buf[i] = (x.At(i, x.c-1) - t) / x.At(i, i)
	}
	return &Array{
		r:    1,
		c:    n,
		data: buf,
	}
}

// CELSQ 矛盾方程组的最小二乘法(contradictory equations)
//
// 输入:
//
//	A : 线性代数方程组(A x = b)的系数矩阵
//	b : 线性代数方程组(A x = b)的常数列向量
//
// 输出: sol 解
func CELSQ(a *Array, b *Array) (sol *Array) {
	if a.r != b.r {
		return
	}
	at := T(a)
	aa := at.Dot(a)
	ab := at.Dot(b)
	return EMCP(Cat(aa, ab, 1))
}

// LSQFitting 线性最小二乘拟合
//
// 输入: [ xy ] (n, 2)维矩阵
//
// 输出: sol 解
func LSQFitting(xy *Array) (sol *Array) {
	ex2, ex, exy, ey := 0.0, 0.0, 0.0, 0.0
	for ir := range xy.r {
		ex2 += xy.At(ir, 0) * xy.At(ir, 0)
		ex += xy.At(ir, 0)
		exy += xy.At(ir, 0) * xy.At(ir, 1)
		ey += xy.At(ir, 1)
	}
	a := NewArray(2, 3, []float64{
		float64(xy.r), ex, ey,
		ex, ex2, exy,
	})
	return EMCP(a)
}

// PolyFitting 多项式拟合
//
// 输入:
//
//	[ xy ] (n, 2)维矩阵
//	m: 多项式次数，m < N-1
//
// 输出: sol 解
func PolyFitting(xy *Array, m int) (sol *Array) {
	n := xy.r
	a := ZeroArray(n, m+1)
	for i := range n {
		for j := range m + 1 {
			a.Set(i, j, math.Pow(xy.At(i, 0), float64(j)))
		}
	}
	return CELSQ(a, xy.Of(1, 1))
}
