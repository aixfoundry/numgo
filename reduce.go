package numgo

import (
	"math"
	"reflect"
	"slices"

	"github.com/kaisawind/numgo/internal/nditer"
)

// This file holds reductions. Every reduction is Pure (new data). A
// whole-array reduction returns a 1-element 1-D array. Reducing a zero-size
// array yields the operation's identity (Sum [0], Prod [1], Max [-Inf],
// Min [+Inf], Mean/Var/Std [NaN]) instead of numpy's error.

// normalizeAxes resolves negative axes against ndim, rejecting invalid ones.
// Duplicates are harmless: Reduce/Accumulate treat axes as a set.
func normalizeAxes(axes []int, ndim int) []int {
	out := make([]int, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax += ndim
		}
		if ax < 0 || ax >= ndim {
			panicAxis(ax, ndim)
		}
		out[i] = ax
	}
	return out
}

// allAxes is the no-axes convention: fold over every axis.
func allAxes(ndim int) []int {
	axes := make([]int, ndim)
	for i := range axes {
		axes[i] = i
	}
	return axes
}

// reduceShape returns the shape with the given axes removed; reducing all axes
// collapses to a 1-element 1-D shape.
func reduceShape(shape []int, axes []int) []int {
	drop := make([]bool, len(shape))
	for _, ax := range axes {
		drop[ax] = true
	}
	out := make([]int, 0, len(shape))
	for i, d := range shape {
		if !drop[i] {
			out = append(out, d)
		}
	}
	if len(out) == 0 {
		return []int{1}
	}
	return out
}

// reduced folds a along axes into a fresh 1-d array prefilled with ident.
// All whole-array/axis reductions route through it.
func reduced[T Number](a *Array[T], axes []int, ident T, fn func(acc *T, v T)) *Array[T] {
	shape := reduceShape(a.shape, axes)
	out := make([]T, nditer.Size(shape))
	for i := range out {
		out[i] = ident
	}
	nditer.Reduce(a.layout(), a.data, axes, out, fn)
	return newArray(shape, nditer.COrderStrides(shape), 0, out)
}

// reducedCount is the number of elements folded per slot for the axes.
func reducedCount[T Number](a *Array[T], axes []int) int {
	n := 1
	for _, ax := range axes {
		n *= a.shape[ax]
	}
	return n
}

// keptShape returns the input shape with the reduced axes set to 1 (keepdims).
func keptShape(shape, axes []int) []int {
	out := slices.Clone(shape)
	for _, ax := range axes {
		out[ax] = 1
	}
	return out
}

// reduceKeepdims folds along axes and re-shapes the result with the reduced
// axes as size 1.
func reduceKeepdims[T Number](a *Array[T], fold func(axes ...int) *Array[T], axes ...int) *Array[T] {
	axes = a.axesOrAll(axes)
	norm := normalizeAxes(axes, a.Ndim())
	shape := slices.Clone(a.shape)
	for _, ax := range norm {
		shape[ax] = 1
	}
	r := fold(axes...)
	return newArray(shape, nditer.COrderStrides(shape), 0, r.data)
}

// Pure: Sum folds addition along axes (no axes = whole array, result shape
// [1]). Negative axes resolve as axis+ndim; invalid axes panic.
func (a *Array[T]) Sum(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	return reduced(a, axes, 0, func(acc *T, v T) { *acc += v })
}

// Pure: SumKeepdims folds addition along axes, keeping reduced axes as size 1.
func (a *Array[T]) SumKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.Sum, axes...)
}

// Pure: Prod folds multiplication along axes (no axes = whole array, [1]).
func (a *Array[T]) Prod(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	return reduced(a, axes, 1, func(acc *T, v T) { *acc *= v })
}

// Pure: ProdKeepdims folds multiplication along axes, keeping reduced axes as
// size 1.
func (a *Array[T]) ProdKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.Prod, axes...)
}

// Pure: Max folds max along axes (no axes = whole array, [1]). The zero-size
// identity is -Inf: every element compares greater, so the fold adopts the
// first element and the empty result stays -Inf.
func (a *Array[T]) Max(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	return reduced(a, axes, minIdent[T](), func(acc *T, v T) {
		if greater(v, *acc) {
			*acc = v
		}
	})
}

// Pure: MaxKeepdims folds max along axes, keeping reduced axes as size 1.
func (a *Array[T]) MaxKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.Max, axes...)
}

// Pure: Min folds min along axes (no axes = whole array, [1]). The zero-size
// identity is +Inf: every element compares less, so the fold adopts the first
// element and the empty result stays +Inf.
func (a *Array[T]) Min(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	return reduced(a, axes, maxIdent[T](), func(acc *T, v T) {
		if less(v, *acc) {
			*acc = v
		}
	})
}

// Pure: MinKeepdims folds min along axes, keeping reduced axes as size 1.
func (a *Array[T]) MinKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.Min, axes...)
}

// maxIdent/minIdent build the ±Inf identities. math.Inf cannot be a constant
// for arbitrary T, so the conversion goes through scalarFromParts.
func maxIdent[T Number]() T { return scalarFromParts[T](math.Inf(1), 0) }

func minIdent[T Number]() T { return scalarFromParts[T](math.Inf(-1), 0) }

// greater/less order Numbers by real part, matching Arange's convention for
// complex operands.
func greater[T Number](a, b T) bool {
	re, _ := parts(a)
	other, _ := parts(b)
	return re > other
}

func less[T Number](a, b T) bool {
	re, _ := parts(a)
	other, _ := parts(b)
	return re < other
}

// Pure: Mean divides the sum along axes by the reduced element count. It
// accumulates in T (documented); the zero-size identity is NaN.
func (a *Array[T]) Mean(axes ...int) *Array[T] {
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	count := reducedCount(a, axes)
	out := reduced(a, axes, 0, func(acc *T, v T) { *acc += v })
	n := scalarFromParts[T](float64(count), 0)
	for i := range out.data {
		out.data[i] /= n
	}
	return out
}

// Pure: MeanKeepdims folds the mean along axes, keeping reduced axes as size 1.
func (a *Array[T]) MeanKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.Mean, axes...)
}

// isComplexKind reports whether T is complex. Var/Std have no T-representable
// value for complex input (numpy raises TypeError likewise), so they panic.
func isComplexKind[T Number]() bool {
	var v T
	k := reflect.TypeOf(v).Kind()
	return k == reflect.Complex64 || k == reflect.Complex128
}

// Pure: Var computes the population variance (ddof=0) along axes. The result
// is real-valued: Var/Std of complex arrays are not representable in T and
// panic (numpy raises TypeError for the same reason). The zero-size identity
// is NaN.
func (a *Array[T]) Var(axes ...int) *Array[T] {
	return a.variance(0, a.axesOrAll(axes))
}

// Pure: VarKeepdims folds the variance along axes, keeping reduced axes as
// size 1.
func (a *Array[T]) VarKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.Var, axes...)
}

// Pure: VarDDof computes the variance along axes with the given delta degrees
// of freedom.
func (a *Array[T]) VarDDof(ddof int, axes ...int) *Array[T] {
	return a.variance(ddof, a.axesOrAll(axes))
}

// Pure: Std computes the standard deviation along axes with ddof=0.
func (a *Array[T]) Std(axes ...int) *Array[T] {
	return a.stddev(0, a.axesOrAll(axes))
}

// Pure: StdKeepdims folds the standard deviation along axes, keeping reduced
// axes as size 1.
func (a *Array[T]) StdKeepdims(axes ...int) *Array[T] {
	return reduceKeepdims(a, a.Std, axes...)
}

// Pure: StdDDof computes the standard deviation along axes with the given
// delta degrees of freedom.
func (a *Array[T]) StdDDof(ddof int, axes ...int) *Array[T] {
	return a.stddev(ddof, a.axesOrAll(axes))
}

// variance computes the float64 variance along axes: (E[x²] - mean²) scaled by
// count/(count-ddof). Only the real path exists; complex T panicked above.
func (a *Array[T]) variance(ddof int, axes []int) *Array[T] {
	if isComplexKind[T]() {
		panicComplexStat("Var/Std")
	}
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
	}
	axes = normalizeAxes(axes, a.Ndim())
	count := reducedCount(a, axes)
	denom := count - ddof
	sum := reduced(a, axes, 0, func(acc *T, v T) {
		re, _ := parts(v)
		*acc = scalarFromParts[T](mustFloat(*acc)+re, 0)
	})
	sumSq := reduced(a, axes, 0, func(acc *T, v T) {
		re, _ := parts(v)
		*acc = scalarFromParts[T](mustFloat(*acc)+re*re, 0)
	})
	shape := reduceShape(a.shape, axes)
	out := make([]T, len(sum.data))
	for i := range out {
		if count == 0 || denom <= 0 {
			// Zero-size fold, or ddof >= count: numpy yields NaN; mirror it.
			out[i] = scalarFromParts[T](math.NaN(), 0)
			continue
		}
		mean := mustFloat(sum.data[i]) / float64(count)
		out[i] = scalarFromParts[T]((mustFloat(sumSq.data[i])-float64(count)*mean*mean)/float64(denom), 0)
	}
	return newArray(shape, nditer.COrderStrides(shape), 0, out)
}

// stddev is variance followed by a square root.
func (a *Array[T]) stddev(ddof int, axes []int) *Array[T] {
	v := a.variance(ddof, axes)
	for i, x := range v.data {
		v.data[i] = scalarFromParts[T](math.Sqrt(mustFloat(x)), 0)
	}
	return v
}

// mustFloat converts a real Number to float64 via parts.
func mustFloat[T Number](v T) float64 {
	re, _ := parts(v)
	return re
}

// argBest tracks the winning value and its coordinate along the folded axis.
type argBest[T Number] struct {
	val T
	idx int
}

// argExtreme is the shared body of ArgMax/ArgMin; better picks the winner.
func argExtreme[T Number](a *Array[T], axis int, better func(a, b T) bool) []int {
	ax := a.resolveAxis(axis)
	slots := a.Size() / a.shape[ax]
	cur := make([]argBest[T], slots)
	seen := make([]bool, slots)
	nditer.ReduceAxis(a.layout(), a.data, axis, func(slot, coord int, v T) {
		if !seen[slot] || better(v, cur[slot].val) {
			cur[slot] = argBest[T]{v, coord}
			seen[slot] = true
		}
	})
	out := make([]int, slots)
	for i, b := range cur {
		out[i] = b.idx
	}
	return out
}

// Pure: ArgMax returns the coordinate of each maximum along axis (first
// occurrence wins ties), one entry per kept slot in C order. It returns a
// plain []int: index results are not element data, and Array's Number
// constraint excludes integer element types.
func (a *Array[T]) ArgMax(axis int) []int {
	return argExtreme(a, axis, greater[T])
}

// Pure: ArgMin returns the coordinate of each minimum along axis (first
// occurrence wins ties), one entry per kept slot in C order.
func (a *Array[T]) ArgMin(axis int) []int {
	return argExtreme(a, axis, less[T])
}

// Pure: Any reports whether any element is nonzero.
func (a *Array[T]) Any() bool {
	for _, v := range a.data {
		re, im := parts(v)
		if re != 0 || im != 0 {
			return true
		}
	}
	return false
}

// Pure: All reports whether every element is nonzero. The zero-size array is
// vacuously all-nonzero.
func (a *Array[T]) All() bool {
	for _, v := range a.data {
		re, im := parts(v)
		if re == 0 && im == 0 {
			return false
		}
	}
	return true
}

// Pure: CumSum prefixes sums along axes (no axes = C-order prefix over the
// raveled array, numpy's cumsum default). The result has the input's shape.
func (a *Array[T]) CumSum(axes ...int) *Array[T] {
	return a.accumulate(axes, func(acc *T, v T) { *acc += v })
}

// Pure: CumProd prefixes products along axes (no axes = C-order prefix over
// the raveled array). The result has the input's shape.
func (a *Array[T]) CumProd(axes ...int) *Array[T] {
	return a.accumulate(axes, func(acc *T, v T) { *acc *= v })
}

// accumulate routes CumSum/CumProd through nditer.Accumulate. With no axes the
// result is 1-d: numpy's cumsum default folds the raveled array.
func (a *Array[T]) accumulate(axes []int, fn func(acc *T, v T)) *Array[T] {
	shape := a.shape
	if len(axes) == 0 {
		axes = allAxes(a.Ndim())
		shape = []int{a.Size()}
	}
	axes = normalizeAxes(axes, a.Ndim())
	out := make([]T, a.Size())
	nditer.Accumulate(a.layout(), a.data, axes, out, fn)
	return newArray(slices.Clone(shape), nditer.COrderStrides(shape), 0, out)
}
