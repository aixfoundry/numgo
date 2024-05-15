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
