package numgo

import (
	"math"
	"slices"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// This file holds the NaN-skipping variants of the reductions (numpy's
// nanfunctions). NaN is detected through the real part, the package-wide
// convention (complex values are NaN only through their real part, matching
// how sort orders them). numpy warns on all-NaN slices and returns NaN; we
// return NaN silently. Zero-size or all-NaN slices yield NaN for
// max/min/mean/var/median and the fold identity (0 for sum/cumsum, 1 for
// prod/cumprod) for the rest — numpy raises for an empty max/min reduction
// instead.

// isNaN reports whether v's real part is NaN.
func isNaN[T Number](v T) bool { return math.IsNaN(mustFloat(v)) }

// nanScalar builds the NaN element of T.
func nanScalar[T Number]() T { return scalarFromParts[T](math.NaN(), 0) }

// nanReplaced copies a with every NaN replaced by ident.
func nanReplaced[T Number](a *Array[T], ident T) *Array[T] {
	out := a.Copy()
	for i, v := range out.data {
		if isNaN(v) {
			out.data[i] = ident
		}
	}
	return out
}

// Pure: NanSum folds addition along axes, treating NaN as 0 (numpy nansum; an
// all-NaN slice therefore sums to 0).
func (a *Array[T]) NanSum(axes ...int) *Array[T] {
	return nanReplaced(a, 0).Sum(axes...)
}

// Pure: NanSumKeepdims folds the NaN-skipping sum along axes, keeping reduced
// axes as size 1.
func (a *Array[T]) NanSumKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.NanSum, axes...)
}

// Pure: NanProd folds multiplication along axes, treating NaN as 1 (numpy
// nanprod; an all-NaN slice therefore multiplies to 1).
func (a *Array[T]) NanProd(axes ...int) *Array[T] {
	return nanReplaced(a, 1).Prod(axes...)
}

// Pure: NanProdKeepdims folds the NaN-skipping product along axes, keeping
// reduced axes as size 1.
func (a *Array[T]) NanProdKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.NanProd, axes...)
}

// Pure: NanMax folds max along axes, skipping NaNs; a slice with no non-NaN
// values (including an empty one) yields NaN.
func (a *Array[T]) NanMax(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	// The NaN identity doubles as the "nothing seen yet" marker: only
	// non-NaN values ever overwrite it.
	return reduced(a, axes, nanScalar[T](), func(acc *T, v T) {
		if !isNaN(v) && (isNaN(*acc) || greater(v, *acc)) {
			*acc = v
		}
	})
}

// Pure: NanMaxKeepdims folds the NaN-skipping max along axes, keeping reduced
// axes as size 1.
func (a *Array[T]) NanMaxKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.NanMax, axes...)
}

// Pure: NanMin folds min along axes, skipping NaNs; a slice with no non-NaN
// values (including an empty one) yields NaN.
func (a *Array[T]) NanMin(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	return reduced(a, axes, nanScalar[T](), func(acc *T, v T) {
		if !isNaN(v) && (isNaN(*acc) || less(v, *acc)) {
			*acc = v
		}
	})
}

// Pure: NanMinKeepdims folds the NaN-skipping min along axes, keeping reduced
// axes as size 1.
func (a *Array[T]) NanMinKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.NanMin, axes...)
}

// Pure: NanMean averages the non-NaN values along axes; a slice without
// non-NaN values yields NaN.
func (a *Array[T]) NanMean(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	sum := reduced(a, axes, 0, func(acc *T, v T) {
		if !isNaN(v) {
			*acc += v
		}
	})
	cnt := a.nanCount(axes)
	for i := range sum.data {
		if cnt.data[i] == 0 {
			sum.data[i] = nanScalar[T]()
		} else {
			sum.data[i] /= cnt.data[i]
		}
	}
	return sum
}

// Pure: NanMeanKeepdims folds the NaN-skipping mean along axes, keeping
// reduced axes as size 1.
func (a *Array[T]) NanMeanKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.NanMean, axes...)
}

// nanCount folds the number of non-NaN values along axes.
func (a *Array[T]) nanCount(axes []int) *Array[T] {
	return reduced(a, axes, 0, func(acc *T, v T) {
		if !isNaN(v) {
			*acc++
		}
	})
}

// Pure: NanVar computes the NaN-skipping population variance (ddof=0) along
// axes. Complex arrays panic as with Var. A slice with fewer non-NaN values
// than ddof+1 yields NaN.
func (a *Array[T]) NanVar(axes ...int) *Array[T] {
	return a.nanvariance(0, a.axesOrAll(axes))
}

// Pure: NanVarKeepdims folds the NaN-skipping variance along axes, keeping
// reduced axes as size 1.
func (a *Array[T]) NanVarKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.NanVar, axes...)
}

// Pure: NanVarDDof computes the NaN-skipping variance along axes with the
// given delta degrees of freedom.
func (a *Array[T]) NanVarDDof(ddof int, axes ...int) *Array[T] {
	return a.nanvariance(ddof, a.axesOrAll(axes))
}

// Pure: NanStd computes the NaN-skipping standard deviation along axes with
// ddof=0.
func (a *Array[T]) NanStd(axes ...int) *Array[T] {
	return a.nanstddev(0, a.axesOrAll(axes))
}

// Pure: NanStdKeepdims folds the NaN-skipping standard deviation along axes,
// keeping reduced axes as size 1.
func (a *Array[T]) NanStdKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.NanStd, axes...)
}

// Pure: NanStdDDof computes the NaN-skipping standard deviation along axes
// with the given delta degrees of freedom.
func (a *Array[T]) NanStdDDof(ddof int, axes ...int) *Array[T] {
	return a.nanstddev(ddof, a.axesOrAll(axes))
}

// nanvariance computes the NaN-skipping variance along axes, accumulating
// sum and sum of squares in T like variance does.
func (a *Array[T]) nanvariance(ddof int, axes []int) *Array[T] {
	if isComplexKind[T]() {
		panicComplexStat("Var/Std")
	}
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	channels, n := channelValues(a, axes)
	shape := reduceShape(a.shape, axes)
	out := make([]T, len(channels))
	for i, ch := range channels {
		var sum, sumsq T
		cnt := 0
		for _, v := range ch {
			if isNaN(v) {
				continue
			}
			re, _ := parts(v)
			sum = scalarFromParts[T](mustFloat(sum)+re, 0)
			sumsq = scalarFromParts[T](mustFloat(sumsq)+re*re, 0)
			cnt++
		}
		if n == 0 || cnt == 0 || cnt-ddof <= 0 {
			out[i] = nanScalar[T]()
			continue
		}
		mean := mustFloat(sum) / float64(cnt)
		out[i] = scalarFromParts[T]((mustFloat(sumsq)-float64(cnt)*mean*mean)/float64(cnt-ddof), 0)
	}
	return newArray(shape, nditer.COrderStrides(shape), 0, out)
}

// nanstddev is nanvariance followed by a square root.
func (a *Array[T]) nanstddev(ddof int, axes []int) *Array[T] {
	v := a.nanvariance(ddof, axes)
	for i, x := range v.data {
		v.data[i] = scalarFromParts[T](math.Sqrt(mustFloat(x)), 0)
	}
	return v
}

// Pure: NanMedian computes the median of the non-NaN values along axes
// (linear interpolation between the two middle values for even counts); a
// slice without non-NaN values yields NaN.
func (a *Array[T]) NanMedian(axes ...int) *Array[T] {
	axes = a.axesOrAll(axes)
	norm := normalizeAxes(axes, a.Ndim())
	channels, n := channelValues(a, norm)
	shape := reduceShape(a.shape, norm)
	out := make([]T, len(channels))
	for i, ch := range channels {
		vals := make([]T, 0, len(ch))
		for _, v := range ch {
			if !isNaN(v) {
				vals = append(vals, v)
			}
		}
		if n == 0 || len(vals) == 0 {
			out[i] = nanScalar[T]()
			continue
		}
		slices.SortFunc(vals, cmpNumber[T])
		out[i] = quantileSorted(vals, 0.5)
	}
	return newArray(shape, nditer.COrderStrides(shape), 0, out)
}

// Pure: NanMedianKeepdims folds the NaN-skipping median along axes, keeping
// reduced axes as size 1.
func (a *Array[T]) NanMedianKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.NanMedian, axes...)
}

// Pure: NanArgMax returns the coordinate of each maximum along axis, skipping
// NaNs (first occurrence wins ties among non-NaN values), one entry per kept
// slot in C order. A slice holding only NaNs panics, as numpy's nanargmax
// raises.
func (a *Array[T]) NanArgMax(axis int) []int {
	return nanArgExtreme(a, axis, greater[T])
}

// Pure: NanArgMin returns the coordinate of each minimum along axis, skipping
// NaNs (first occurrence wins ties among non-NaN values), one entry per kept
// slot in C order. A slice holding only NaNs panics.
func (a *Array[T]) NanArgMin(axis int) []int {
	return nanArgExtreme(a, axis, less[T])
}

// nanArgExtreme is argExtreme with NaNs skipped and an all-NaN panic.
func nanArgExtreme[T Number](a *Array[T], axis int, better func(a, b T) bool) []int {
	ax := a.resolveAxis(axis)
	slots := a.Size() / max(a.shape[ax], 1)
	cur := make([]argBest[T], slots)
	seen := make([]bool, slots)
	nditer.ReduceAxis(a.layout(), a.data, ax, func(slot, coord int, v T) {
		if isNaN(v) {
			return
		}
		if !seen[slot] || better(v, cur[slot].val) {
			cur[slot] = argBest[T]{v, coord}
			seen[slot] = true
		}
	})
	out := make([]int, slots)
	for i, s := range seen {
		if !s {
			panicAllNaN()
		}
		out[i] = cur[i].idx
	}
	return out
}

// Pure: NanCumSum prefixes NaN-free sums along axes (no axes = C-order prefix
// over the raveled array, like CumSum): NaN contributes 0, mirroring numpy's
// nancumsum.
func (a *Array[T]) NanCumSum(axes ...int) *Array[T] {
	return nanReplaced(a, 0).CumSum(axes...)
}

// Pure: NanCumProd prefixes NaN-free products along axes (no axes = C-order
// prefix over the raveled array): NaN contributes 1, mirroring numpy's
// nancumprod.
func (a *Array[T]) NanCumProd(axes ...int) *Array[T] {
	return nanReplaced(a, 1).CumProd(axes...)
}
