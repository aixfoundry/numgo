package numgo

import "fmt"

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

// T 矩阵转置
//
// numpy: a.T
func T(a *Array) (o *Array) {
	o = ZeroArray(a.c, a.r)
	for ir := range a.c {
		for ic := range a.r {
			o.Set(ir, ic, a.At(ic, ir))
		}
	}
	return
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
			a.data[f*a.c+ic], a.data[t*a.c+ic] = a.data[t*a.c+ic], a.data[f*a.c+ic]
		}
	} else if axis == 1 {
		for ir := range a.r {
			a.data[f+a.c*ir], a.data[t+a.c*ir] = a.data[t+a.c*ir], a.data[f+a.c*ir]
		}
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
func Swap(a *Array, f, t int, axis int) (o *Array) {
	o = CopyOf(a)
	if axis == 0 {
		for ic := range o.c {
			o.data[f*o.r+ic], o.data[t*o.r+ic] = o.data[t*o.r+ic], o.data[f*o.r+ic]
		}
	} else if axis == 1 {
		for ir := range o.r {
			o.data[f+o.r*ir], o.data[t+o.r*ir] = o.data[t+o.r*ir], o.data[f+o.r*ir]
		}
	}
	return
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

// Cat 拼接两个向量
//
// numpy: np.concatenate((a, b), axis)
//
// axis:
//
//   - 0: 按行拼接
//
//   - 1: 按列拼接
func Cat(a *Array, b *Array, axis int) (o *Array) {
	if axis == 0 {
		o = EmptyArray(a.r+b.r, a.c)
		for i := range a.r {
			for ja := range a.c {
				o.data = append(o.data, a.data[i*a.c+ja])
			}
		}
		for i := range b.r {
			for jb := range b.c {
				o.data = append(o.data, b.data[i*b.c+jb])
			}
		}
	} else {
		o = EmptyArray(a.r, a.c+b.c)
		for i := range a.r {
			for ja := range a.c {
				o.data = append(o.data, a.data[i*a.c+ja])
			}
			for jb := range b.c {
				o.data = append(o.data, b.data[i*b.c+jb])
			}
		}
	}
	return
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

// Delete 删除某行或列
//
// n: 要删除的行或列
// axis:
//
//   - 0: 按行删除
//
//   - 1: 按列删除
func Delete(a *Array, n int, axis int) *Array {
	buf := []float64{}
	r, c := a.r, a.c
	if axis == 0 {
		buf = append(a.data[:a.r*n], a.data[a.r*n+a.c:]...)
		r = a.r - 1
	} else {
		for ir := range a.r {
			for ic := range a.c {
				if ic == n {
					continue
				}
				buf = append(buf, a.data[ir*a.r+ic])
			}
		}
		c = a.c - 1
	}
	return NewArray(r, c, buf)
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

// RangeDelete 范围删除行或列
//
// f: 要删除的行或列的开始
// t: 要删除的行或列的结束
// axis:
//
//   - 0: 按行删除
//
//   - 1: 按列删除
func RangeDelete(a *Array, f, t int, axis int) *Array {
	buf := []float64{}
	r, c := a.r, a.c
	if axis == 0 {
		buf = append(a.data[:f*a.r], a.data[t*a.r:]...)
		r = a.r - t + f
	} else {
		for ir := range a.r {
			for ic := range a.c {
				if ic >= f && ic < t {
					continue
				}
				buf = append(buf, a.data[ir*a.r+ic])
			}
		}
		c = a.c - t + f
	}
	return NewArray(r, c, buf)
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

// Reshape 矩阵变形
func Reshape(a *Array, r, c int) (o *Array) {
	o = CopyOf(a)
	if r*c > len(o.data) {
		buf := make([]float64, r*c-len(o.data))
		o.data = append(o.data, buf...)
	} else if r*c < len(o.data) {
		o.data = o.data[:r*c]
	}
	o.r, o.c = r, c
	return
}

func ExampleArray_Delete_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Delete(1, 0))
	// Output:
	// ⎡1  2  3⎤
	// ⎣7  8  9⎦
}

func ExampleArray_Delete_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Delete(1, 1))
	// Output:
	// ⎡1  3⎤
	// ⎢4  6⎥
	// ⎣7  9⎦
}

func ExampleArray_RangeDelete_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.RangeDelete(0, 1, 0))
	// Output:
	// ⎡4  5  6⎤
	// ⎣7  8  9⎦
}

func ExampleArray_RangeDelete_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.RangeDelete(0, 1, 1))
	// Output:
	// ⎡2  3⎤
	// ⎢5  6⎥
	// ⎣8  9⎦
}
