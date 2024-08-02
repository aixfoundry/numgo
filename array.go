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
