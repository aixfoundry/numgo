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
func (x *Array) Reshape(r, c int) *Array {
	if r*c > len(x.data) {
		buf := make([]float64, r*c-len(x.data))
		x.data = append(x.data, buf...)
	} else if r*c < len(x.data) {
		x.data = x.data[:r*c]
	}
	x.r, x.c = r, c
	return x
}

// T 矩阵转置
//
// numpy: a.T
func (x *Array) T() *Array {
	data := make([]float64, len(x.data))
	r, c := x.r, x.c
	copy(data, x.data)
	x.Reshape(x.c, x.r)
	for ir := range c {
		for ic := range r {
			x.Set(ir, ic, data[ic*c+ir])
		}
	}
	return x
}

// Dims 获取矩阵形状
func (x *Array) Dims() (r, c int) {
	return x.r, x.c
}

// At 获取坐标值
func (x *Array) At(i, j int) float64 {
	return x.data[i*x.c+j]
}

// Set 设置坐标值
func (x *Array) Set(i, j int, f float64) *Array {
	x.data[i*x.c+j] = f
	return x
}

// Format 字符串输出格式
func (x *Array) Format(fs fmt.State, c rune) {
	x.format("", 0, '.', true, fs, c)
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
func (x *Array) RangeDelete(f, t int, axis int) *Array {
	if axis == 0 {
		x.data = append(x.data[:f*x.r], x.data[t*x.r:]...)
		x.r -= t - f
	} else {
		buf := []float64{}
		for ir := range x.r {
			for ic := range x.c {
				if ic >= f && ic < t {
					continue
				}
				buf = append(buf, x.data[ir*x.r+ic])
			}
		}
		x.c -= t - f
		x.data = buf
	}
	return x
}

// Delete 删除某行或列
//
// n: 要删除的行或列
// axis:
//
//   - 0: 按行删除
//
//   - 1: 按列删除
func (x *Array) Delete(n int, axis int) *Array {
	if axis == 0 {
		x.data = append(x.data[:x.r*n], x.data[x.r*n+x.c:]...)
		x.r--
	} else {
		buf := []float64{}
		for ir := range x.r {
			for ic := range x.c {
				if ic == n {
					continue
				}
				buf = append(buf, x.data[ir*x.r+ic])
			}
		}
		x.data = buf
		x.c--
	}
	return x
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
func (x *Array) Swap(f, t int, axis int) *Array {
	if axis == 0 {
		for ic := range x.c {
			x.data[f*x.r+ic], x.data[t*x.r+ic] = x.data[t*x.r+ic], x.data[f*x.r+ic]
		}
	} else if axis == 1 {
		for ir := range x.r {
			x.data[f+x.r*ir], x.data[t+x.r*ir] = x.data[t+x.r*ir], x.data[f+x.r*ir]
		}
	}
	return x
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
func (x *Array) Cat(b *Array, axis int) *Array {
	if axis == 0 {
		x.r += b.r
		x.data = append(x.data, b.data...)
	} else {
		buf := []float64{}
		for i := range x.r {
			for ja := range x.c {
				buf = append(buf, x.data[i*x.c+ja])
			}
			for jb := range b.c {
				buf = append(buf, b.data[i*b.c+jb])
			}
		}
		x.c += b.c
		x.data = buf
	}
	return x
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
func (x *Array) Of(i int, axis int) *Array {
	buf := []float64{}
	if axis == 0 {
		for ir := range x.r {
			if ir == i {
				for ic := range x.c {
					buf = append(buf, x.At(ir, ic))
				}
				break
			}
		}
		return NewArray(1, x.c, buf)
	} else {
		for ir := range x.r {
			for ic := range x.c {
				if ic == i {
					buf = append(buf, x.At(ir, ic))
					break
				}
			}
		}
		return NewArray(x.r, 1, buf)
	}
}
