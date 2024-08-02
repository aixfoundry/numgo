package numgo

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
