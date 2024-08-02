package numgo

import (
	"fmt"
)

// Array 矩阵
type Array struct {
	r, c int
	data []float64
}

// CopyOf 创建一个新的副本
func CopyOf(a *Array) (x *Array) {
	x = &Array{
		c:    a.c,
		r:    a.r,
		data: make([]float64, len(a.data)),
	}
	copy(x.data, a.data)
	return
}

// EmptyArray 创建空矩阵
func EmptyArray(r, c int) (a *Array) {
	return &Array{
		r: r,
		c: c,
	}
}

// ZeroArray 创建全为0的矩阵
func ZeroArray(r, c int) (a *Array) {
	return &Array{
		r:    r,
		c:    c,
		data: make([]float64, r*c),
	}
}

// NewArray 创建矩阵
func NewArray(r, c int, data []float64) (a *Array) {
	if r*c > len(data) {
		buf := make([]float64, r*c-len(data))
		data = append(data, buf...)
	} else if r*c < len(data) {
		data = data[:r*c]
	}
	return &Array{
		r:    r,
		c:    c,
		data: data,
	}
}

// Reshape 矩阵变形
func (a *Array) Reshape(r, c int) *Array {
	if r*c > len(a.data) {
		buf := make([]float64, r*c-len(a.data))
		a.data = append(a.data, buf...)
	} else if r*c < len(a.data) {
		a.data = a.data[:r*c]
	}
	a.r, a.c = r, c
	return a
}

// T 矩阵转置
//
// numpy: a.T
func (a *Array) T() *Array {
	data := make([]float64, len(a.data))
	r, c := a.r, a.c
	copy(data, a.data)
	a.Reshape(a.c, a.r)
	for ir := range c {
		for ic := range r {
			a.Set(ir, ic, data[ic*c+ir])
		}
	}
	return a
}

// Dims 获取矩阵形状
func (a *Array) Dims() (r, c int) {
	return a.r, a.c
}

// At 获取坐标值
func (a *Array) At(i, j int) float64 {
	return a.data[i*a.c+j]
}

// Set 设置坐标值
func (a *Array) Set(i, j int, f float64) *Array {
	a.data[i*a.c+j] = f
	return a
}

// Format 字符串输出格式
func (a *Array) Format(fs fmt.State, c rune) {
	a.format("", 0, '.', true, fs, c)
}

// RangeDelete 范围删除行或列
//
// f: 要删除的行或列的开始
// t: 要删除的行或列的结束
// axis:
//
//   - 0: 按行删除
//
//   - 1: 按列删除
func (a *Array) RangeDelete(f, t int, axis int) *Array {
	if axis == 0 {
		a.data = append(a.data[:f*a.r], a.data[t*a.r:]...)
		a.r -= t - f
	} else {
		buf := []float64{}
		for ir := range a.r {
			for ic := range a.c {
				if ic >= f && ic < t {
					continue
				}
				buf = append(buf, a.data[ir*a.r+ic])
			}
		}
		a.c -= t - f
		a.data = buf
	}
	return a
}

// Delete 删除某行或列
//
// n: 要删除的行或列
// axis:
//
//   - 0: 按行删除
//
//   - 1: 按列删除
func (a *Array) Delete(n int, axis int) *Array {
	if axis == 0 {
		a.data = append(a.data[:a.r*n], a.data[a.r*n+a.c:]...)
		a.r--
	} else {
		buf := []float64{}
		for ir := range a.r {
			for ic := range a.c {
				if ic == n {
					continue
				}
				buf = append(buf, a.data[ir*a.r+ic])
			}
		}
		a.data = buf
		a.c--
	}
	return a
}

// Swap 交换行或列
//
// numpy: a[1], a[2] = a[2], a[1]
//
// axis:
//
//   - 0: 按行交换
//
//   - 1: 按列交换
func (a *Array) Swap(f, t int, axis int) *Array {
	if axis == 0 {
		for ic := range a.c {
			a.data[f*a.r+ic], a.data[t*a.r+ic] = a.data[t*a.r+ic], a.data[f*a.r+ic]
		}
	} else if axis == 1 {
		for ir := range a.r {
			a.data[f+a.r*ir], a.data[t+a.r*ir] = a.data[t+a.r*ir], a.data[f+a.r*ir]
		}
	}
	return a
}

// Cat 拼接两个向量
//
// numpy: np.concatenate((a, b), axis)
//
// axis:
//
//   - 0: 按行拼接
//
//   - 1: 按列拼接
func (a *Array) Cat(b *Array, axis int) *Array {
	if axis == 0 {
		a.r += b.r
		a.data = append(a.data, b.data...)
	} else {
		buf := []float64{}
		for i := range a.r {
			for ja := range a.c {
				buf = append(buf, a.data[i*a.c+ja])
			}
			for jb := range b.c {
				buf = append(buf, b.data[i*b.c+jb])
			}
		}
		a.c += b.c
		a.data = buf
	}
	return a
}

// Of 获取行或者列
//
// numpy: a[i], a[:, i]
//
// axis:
//
//   - 0: 按行获取
//
//   - 1: 按列获取
func (a *Array) Of(i int, axis int) *Array {
	buf := []float64{}
	if axis == 0 {
		for ir := range a.r {
			if ir == i {
				for ic := range a.c {
					buf = append(buf, a.At(ir, ic))
				}
				break
			}
		}
		return NewArray(1, a.c, buf)
	} else {
		for ir := range a.r {
			for ic := range a.c {
				if ic == i {
					buf = append(buf, a.At(ir, ic))
					break
				}
			}
		}
		return NewArray(a.r, 1, buf)
	}
}
