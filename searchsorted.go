package numgo

import "sort"

// SearchSorted finds, for each element of v, the insertion index into the
// 1-D ascending array a. With side="left" it returns the first i with
// a[i] >= v; with side="right" the first i with a[i] > v. Like numpy, a is
// not checked for sortedness or dimensionality. Any other side panics.
func (a *Array[T]) SearchSorted(v *Array[T], side string) []int {
	if side != "left" && side != "right" {
		panic(`numgo: SearchSorted: side must be "left" or "right"`)
	}
	af := a.Ravel()
	vf := v.Ravel()
	out := make([]int, vf.Size())
	for i := range out {
		x := mustFloat(vf.At(i))
		if side == "left" {
			out[i] = sort.Search(af.Size(), func(j int) bool { return mustFloat(af.At(j)) >= x })
		} else {
			out[i] = sort.Search(af.Size(), func(j int) bool { return mustFloat(af.At(j)) > x })
		}
	}
	return out
}

// Digitize converts each element of x to the index of the bin in the sorted
// array a it falls into: right=false is a.SearchSorted(x, "left"), right=true
// a.SearchSorted(x, "right").
func (a *Array[T]) Digitize(x *Array[T], right bool) []int {
	if right {
		return a.SearchSorted(x, "right")
	}
	return a.SearchSorted(x, "left")
}
