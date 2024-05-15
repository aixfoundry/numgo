package numgo

import (
	"fmt"
)

// Array matrix
type Array struct {
	r, c int
	data []float64
}

func CopyOf(a *Array) (x *Array) {
	x = &Array{
		c:    a.c,
		r:    a.r,
		data: a.data,
	}
	return
}

func EmptyArray(r, c int) (a *Array) {
	return &Array{
		r: r,
		c: c,
	}
}

func ZeroArray(r, c int) (a *Array) {
	return &Array{
		r:    r,
		c:    c,
		data: make([]float64, r*c),
	}
}

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

func (x *Array) T() (a *Array) {
	a = ZeroArray(x.c, x.r)
	for ir := range x.c {
		for ic := range x.r {
			a.Set(ir, ic, x.At(ic, ir))
		}
	}
	return
}

func (x *Array) Dims() (r, c int) {
	return x.r, x.c
}

func (x *Array) At(i, j int) float64 {
	return x.data[i*x.c+j]
}

func (x *Array) Set(i, j int, f float64) {
	x.data[i*x.c+j] = f
}

func (x *Array) Format(fs fmt.State, c rune) {
	x.format("", 0, '.', true, fs, c)
}

func (x *Array) RangeDelete(f, t int, axis int) (a *Array) {
	buf := []float64{}
	r, c := x.r, x.c
	if axis == 0 {
		for ir := range x.r {
			if ir >= f && ir <= t {
				continue
			}
			for ic := range x.c {
				buf = append(buf, x.data[ir*x.r+ic])
			}
		}
		r = x.r - t + f
	} else {
		for ir := range x.r {
			for ic := range x.c {
				if ic >= f && ic <= t {
					continue
				}
				buf = append(buf, x.data[ir*x.r+ic])
			}
		}
		c = x.c - t + f
	}
	return &Array{
		c:    c,
		r:    r,
		data: buf,
	}
}

func (x *Array) Delete(n int, axis int) (a *Array) {
	buf := []float64{}
	r, c := x.r, x.c
	if axis == 0 {
		for ir := range x.r {
			if ir == n {
				continue
			}
			for ic := range x.c {
				buf = append(buf, x.data[ir*x.r+ic])
			}
		}
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
	return &Array{
		c:    c,
		r:    r,
		data: buf,
	}
}

func (x *Array) Swap(f, t int, axis int) {
	if axis == 0 {
		for ic := range x.c {
			x.data[f*x.c+ic], x.data[t*x.c+ic] = x.data[t*x.c+ic], x.data[f*x.c+ic]
		}
	} else if axis == 1 {
		for ir := range x.r {
			x.data[f*x.r+ir], x.data[t*x.r+ir] = x.data[t*x.r+ir], x.data[f*x.r+ir]
		}
	}
}

func (x *Array) Cat(b *Array, axis int) (ret *Array) {
	if axis == 0 {
		ret = EmptyArray(x.r+b.r, x.c)
		for i := range x.r {
			for ja := range x.c {
				ret.data = append(ret.data, x.data[i*x.c+ja])
			}
		}
		for i := range b.r {
			for jb := range b.c {
				ret.data = append(ret.data, b.data[i*b.c+jb])
			}
		}
	} else {
		ret = EmptyArray(x.r, x.c+b.c)
		for i := range x.r {
			for ja := range x.c {
				ret.data = append(ret.data, x.data[i*x.c+ja])
			}
			for jb := range b.c {
				ret.data = append(ret.data, b.data[i*b.c+jb])
			}
		}
	}
	return
}

func (x *Array) Of(i int, axis int) (ret *Array) {
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
