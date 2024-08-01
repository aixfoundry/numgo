package numgo

// T 矩阵转置
//
// numpy: a.T
func T(x *Array) (o *Array) {
	o = ZeroArray(x.c, x.r)
	for ir := range x.c {
		for ic := range x.r {
			o.Set(ir, ic, x.At(ic, ir))
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
func Swap(x *Array, f, t int, axis int) (o *Array) {
	o = CopyOf(x)
	if axis == 0 {
		for ic := range o.c {
			o.data[f*o.c+ic], o.data[t*o.c+ic] = o.data[t*o.c+ic], o.data[f*o.c+ic]
		}
	} else if axis == 1 {
		for ir := range o.r {
			o.data[f*o.r+ir], o.data[t*o.r+ir] = o.data[t*o.r+ir], o.data[f*o.r+ir]
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
func Cat(x *Array, b *Array, axis int) (o *Array) {
	if axis == 0 {
		o = EmptyArray(x.r+b.r, x.c)
		for i := range x.r {
			for ja := range x.c {
				o.data = append(o.data, x.data[i*x.c+ja])
			}
		}
		for i := range b.r {
			for jb := range b.c {
				o.data = append(o.data, b.data[i*b.c+jb])
			}
		}
	} else {
		o = EmptyArray(x.r, x.c+b.c)
		for i := range x.r {
			for ja := range x.c {
				o.data = append(o.data, x.data[i*x.c+ja])
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
func Delete(x *Array, n int, axis int) *Array {
	buf := []float64{}
	r, c := x.r, x.c
	if axis == 0 {
		buf = append(x.data[:x.r*n], x.data[x.r*n+x.c:]...)
		r = x.r - 1
	} else {
		for ir := range x.r {
			for ic := range x.c {
				if ic == n {
					continue
				}
				buf = append(buf, x.data[ir*x.r+ic])
			}
		}
		c = x.c - 1
	}
	return NewArray(c, r, buf)
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
func RangeDelete(x *Array, f, t int, axis int) *Array {
	buf := []float64{}
	r, c := x.r, x.c
	if axis == 0 {
		buf = append(x.data[:f*x.r], x.data[t*x.r:]...)
		r = x.r - 1
	} else {
		for ir := range x.r {
			for ic := range x.c {
				if ic >= f && ic <= t {
					continue
				}
				buf = append(buf, x.data[ir*x.r+ic])
			}
		}
		c = x.c - 1
	}
	return NewArray(c, r, buf)
}
