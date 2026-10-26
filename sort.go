package numgo

import (
	"math"
	"slices"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// This file holds ordering operations: sorts, partitions and uniqueness.
// Unlike Sum, a sort with NO axes argument uses the LAST axis (numpy's
// axis=-1 default), not every axis; explicit axes sort each listed axis
// independently, one pass per axis in the order given.

// sortLess orders Numbers ascending: real part first, then imaginary for
// complex values (numpy sorts complex lexicographically). A NaN — detected
// via its real part — compares GREATER than everything, so NaN sorts last.
func sortLess[T Number](a, b T) bool {
	na, nb := math.IsNaN(mustFloat(a)), math.IsNaN(mustFloat(b))
	if na || nb {
		return nb && !na
	}
	ra, ia := parts(a)
	rb, ib := parts(b)
	if ra != rb {
		return ra < rb
	}
	return ia < ib
}

// cmpNumber is sortLess as a three-way comparison: NaN equals NaN, so
// equal-run detection and stable ordering treat NaNs as a group.
func cmpNumber[T Number](a, b T) int {
	if sortLess[T](a, b) {
		return -1
	}
	if sortLess[T](b, a) {
		return 1
	}
	return 0
}

// sortTyped sorts one channel slice in place with the concrete numeric
// comparator when T is a built-in type (NaN last, complex by real then imag —
// same total order as cmpNumber); false means "use the generic path" (named
// types). Bit-identical placement because the order is the same.
func sortTyped[S ~float32 | ~float64 | ~complex64 | ~complex128](ch []S) bool {
	switch s := any(ch).(type) {
	case []float64:
		slices.SortFunc(s, func(a, b float64) int {
			na, nb := math.IsNaN(a), math.IsNaN(b)
			if na || nb {
				if nb && !na {
					return -1
				}
				if na && !nb {
					return 1
				}
				return 0
			}
			switch {
			case a < b:
				return -1
			case a > b:
				return 1
			}
			return 0
		})
	case []float32:
		slices.SortFunc(s, func(a, b float32) int {
			na, nb := math.IsNaN(float64(a)), math.IsNaN(float64(b))
			if na || nb {
				if nb && !na {
					return -1
				}
				if na && !nb {
					return 1
				}
				return 0
			}
			switch {
			case a < b:
				return -1
			case a > b:
				return 1
			}
			return 0
		})
	default:
		return false
	}
	return true
}

// sortAxis sorts the channels along one axis of a in place. Channels are
// collected with ReduceAxis; the sorted values go back through Zip, which
// handles arbitrary strides.
func sortAxis[T Number](a *Array[T], axis int) {
	ax := a.resolveAxis(axis)
	n := a.shape[ax]
	if n < 2 || a.Size() == 0 {
		return
	}
	// Contiguous last-axis fast path: sort each row slice in place — no
	// channel collection, no scatter back. Sorting is a permutation, so the
	// values (and NaN placement, same comparator) are identical to the
	// generic path.
	if ax == len(a.shape)-1 && a.IsContiguous() {
		d := a.Data()[:a.Size()]
		for i := 0; i < len(d); i += n {
			if !sortTyped(d[i : i+n]) {
				slices.SortFunc(d[i:i+n], cmpNumber[T])
			}
		}
		return
	}
	channels := make([][]T, a.Size()/n)
	nditer.ReduceAxis(a.layout(), a.data, ax, func(slot, coord int, v T) {
		channels[slot] = append(channels[slot], v)
	})
	sorted := make([]T, 0, a.Size())
	for _, ch := range channels {
		slices.SortFunc(ch, cmpNumber[T])
		sorted = append(sorted, ch...)
	}
	// sorted is slot-major (C order over kept dims), coord-minor within each
	// channel: it is the C-order image of a's shape with axis ax moved last.
	// Map the compact permuted-order strides back onto a's axes.
	others := make([]int, 0, len(a.shape))
	for k := range a.shape {
		if k != ax {
			others = append(others, k)
		}
	}
	perm := append(others, ax)
	permShape := make([]int, len(perm))
	for i, k := range perm {
		permShape[i] = a.shape[k]
	}
	permStrides := nditer.COrderStrides(permShape)
	srcStrides := make([]int, len(a.shape))
	for i, k := range perm {
		srcStrides[k] = permStrides[i]
	}
	src := &nditer.Layout{Shape: a.shape, Strides: srcStrides}
	nditer.Zip(a.layout(), a.data, []*nditer.Layout{src}, [][]T{sorted},
		func(d []T, s [][]T) {
			for i := range d {
				d[i] = s[0][i]
			}
		})
}

// Sort returns a new array with each channel along the given axes
// sorted ascending; with no axes the LAST axis is sorted (numpy axis=-1).
// Each listed axis is sorted independently, one pass per axis in the order
// given. NaN sorts last. Negative axes resolve.
func (a *Array[T]) Sort(axes ...int) *Array[T] {
	out := a.Copy()
	if len(axes) == 0 {
		axes = []int{-1}
	}
	for _, ax := range axes {
		sortAxis(out, ax)
	}
	return out
}

// SortInto sorts the receiver's channels in place (same semantics
// as Sort) and returns it.
func (a *Array[T]) SortInto(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = []int{-1}
	}
	for _, ax := range axes {
		sortAxis(a, ax)
	}
	return a
}

// Argsort returns the coordinates that would sort each channel along
// axis as a flat []int (one entry per element, kept slots in C order).
// The order is STABLE: equal values keep their original order, a Go-friendly
// superset of numpy's non-stable quicksort default. Negative axes resolve.
func (a *Array[T]) Argsort(axis int) []int {
	ax := a.resolveAxis(axis)
	n := a.shape[ax]
	type pair struct {
		val T
		idx int
	}
	channels := make([][]pair, a.Size()/max(n, 1))
	nditer.ReduceAxis(a.layout(), a.data, ax, func(slot, coord int, v T) {
		channels[slot] = append(channels[slot], pair{v, coord})
	})
	out := make([]int, 0, a.Size())
	for _, ch := range channels {
		slices.SortStableFunc(ch, func(x, y pair) int { return cmpNumber[T](x.val, y.val) })
		for _, p := range ch {
			out = append(out, p.idx)
		}
	}
	return out
}

// checkKth validates a partition kth against the channel length, resolving
// negative kth from the end.
func checkKth(kth, n int) {
	k := kth
	if k < 0 {
		k += n
	}
	if k < 0 || k >= n {
		panicKth(kth, n)
	}
}

// Partition returns a new array where the element at position kth
// (negative counts from the end) along each channel of axis is in its sorted
// place, with smaller elements left and larger right (numpy partition).
//
// shortcut: O(n log n) sort-based partition, upgrade to introselect if
// profiling needs O(n).
func (a *Array[T]) Partition(kth int, axis int) *Array[T] {
	checkKth(kth, a.shape[a.resolveAxis(axis)])
	return a.Sort(axis)
}

// ArgPartition returns the indices that would partition each channel
// along axis, as a flat []int (same shortcut and stability as Argsort).
//
// shortcut: O(n log n) sort-based partition, upgrade to introselect if
// profiling needs O(n).
func (a *Array[T]) ArgPartition(kth int, axis int) []int {
	checkKth(kth, a.shape[a.resolveAxis(axis)])
	return a.Argsort(axis)
}

// sortedFlat ravels a in C order and returns a sorted copy.
func sortedFlat[T Number](a *Array[T]) []T {
	vals := slices.Clone(a.Ravel().Data())
	slices.SortFunc(vals, cmpNumber[T])
	return vals
}

// Unique returns the flattened (C-order ravel) unique values, sorted.
// NaN — compared by real part — groups into a single NaN placed last, as
// complex values order by (real, imag).
func (a *Array[T]) Unique() *Array[T] {
	vals := slices.CompactFunc(sortedFlat(a), func(x, y T) bool {
		return cmpNumber[T](x, y) == 0
	})
	shape := []int{len(vals)}
	return newArray(shape, nditer.COrderStrides(shape), 0, vals)
}

// UniqueCounts returns Unique's values together with each value's
// occurrence count.
func (a *Array[T]) UniqueCounts() (*Array[T], []int) {
	sorted := sortedFlat(a)
	var vals []T
	var counts []int
	for i, v := range sorted {
		if i > 0 && cmpNumber[T](sorted[i-1], v) == 0 {
			counts[len(counts)-1]++
			continue
		}
		vals = append(vals, v)
		counts = append(counts, 1)
	}
	shape := []int{len(vals)}
	return newArray(shape, nditer.COrderStrides(shape), 0, vals), counts
}
