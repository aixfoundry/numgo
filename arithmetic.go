package numgo

import "math"

func (a *Array) MaxAbs() (sol float64, r, c int) {
	for ir := range a.r {
		for ic := range a.c {
			if math.Abs(sol) < math.Abs(a.data[ir*a.c+ic]) {
				sol = a.data[ir*a.c+ic]
				r, c = ir, ic
			}
		}
	}
	return
}

func MaxAbs(a *Array) (sol float64, r, c int) {
	return a.MaxAbs()
}

func (a *Array) MinAbs() (sol float64, r, c int) {
	for ir := range a.r {
		for ic := range a.c {
			if math.Abs(sol) > math.Abs(a.data[ir*a.c+ic]) {
				sol = a.data[ir*a.c+ic]
				r, c = ir, ic
			}
		}
	}
	return
}

func MinAbs(a *Array) (sol float64, r, c int) {
	return a.MinAbs()
}

func (a *Array) Max() (sol float64, r, c int) {
	for ir := range a.r {
		for ic := range a.c {
			if sol < a.data[ir*a.c+ic] {
				sol = a.data[ir*a.c+ic]
				r, c = ir, ic
			}
		}
	}
	return
}

func Max(a *Array) (sol float64, r, c int) {
	return a.Max()
}

func (a *Array) Min() (sol float64, r, c int) {
	for ir := range a.r {
		for ic := range a.c {
			if sol > a.data[ir*a.c+ic] {
				sol = a.data[ir*a.c+ic]
				r, c = ir, ic
			}
		}
	}
	return
}

func Min(a *Array) (sol float64, r, c int) {
	return a.Min()
}

func (a *Array) Dot(b *Array) *Array {
	if a.c != b.r {
		return nil
	}
	ca := CopyOf(a)
	a = a.Reshape(a.r, b.c)
	for i := range ca.r {
		for j := range b.c {
			for k := range ca.c {
				a.data[b.c*i+j] += ca.At(i, k) * b.At(k, j)
			}
		}
	}
	return a
}

func Dot(a *Array, b *Array) (ret *Array) {
	if a.c != b.r {
		return
	}
	ret = ZeroArray(a.r, b.c)
	for i := range a.r {
		for j := range b.c {
			for k := range a.c {
				ret.data[b.c*i+j] += a.At(i, k) * b.At(k, j)
			}
		}
	}
	return
}
