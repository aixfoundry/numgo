package ma

import (
	"math"
	"slices"

	"github.com/aixfoundry/numgo"
)

// Reductions: masked elements are ignored. A channel with no unmasked
// elements yields NaN (masked), standing in for numpy's masked element +
// RuntimeWarning. All are Pure. Axis conventions follow the root package:
// no axes = fold everything (result shape [1]), negative axes resolve as
// axis+ndim, invalid axes panic.

// fold shares the walk-and-fold body: it accumulates into the kept slots of a
// fresh output shaped like the reduction result.
type fold[T numgo.Number] struct {
	out   *numgo.Array[T]
	mask  []bool
	count []int
	shape []int
	norm  []int
}

// beginFold allocates the reduction output for m over norm axes.
func beginFold[T numgo.Number](m *Masked[T], axes []int) *fold[T] {
	norm := normAxes(axes, m.Ndim())
	shape := reduceShape(m.Shape(), norm)
	n := sizeOf(shape)
	return &fold[T]{
		out:   numgo.Zeros[T](shape...),
		mask:  make([]bool, n),
		count: make([]int, n),
		shape: shape,
		norm:  norm,
	}
}

// add folds one unmasked element into its kept slot; fn mutates the slot.
// Slots are addressed through the C-contiguous output's flat data.
func (f *fold[T]) add(m *Masked[T], coords []int, fn func(acc *T, v T)) {
	i := flat(m.Shape(), coords)
	if m.mask[i] {
		return
	}
	kept := keptCoords(coords, m.Shape(), f.norm)
	k := flat(f.shape, kept)
	if f.count[k] == 0 {
		f.out.Data()[k] = m.a.Data()[i]
	} else {
		acc := f.out.Data()[k]
		fn(&acc, m.a.Data()[i])
		f.out.Data()[k] = acc
	}
	f.count[k]++
}

// finish marks slots with no unmasked elements as masked NaN.
func (f *fold[T]) finish() *Masked[T] {
	for k, c := range f.count {
		if c == 0 {
			f.out.Data()[k] = nan[T]()
			f.mask[k] = true
		}
	}
	return newMasked(f.out, f.mask)
}

// Sum folds addition along axes over unmasked elements.
func (m *Masked[T]) Sum(axes ...int) *Masked[T] {
	if len(axes) == 0 {
		axes = allAxes(m.Ndim())
	}
	f := beginFold(m, axes)
	walk(m.Shape(), func(coords []int) { f.add(m, coords, func(acc *T, v T) { *acc += v }) })
	return f.finish()
}

// SumKeepdims folds the sum keeping reduced axes as size 1.
func (m *Masked[T]) SumKeepdims(axes ...int) *Masked[T] {
	return m.keepdims(m.Sum, axes...)
}

// Mean divides the unmasked sum by the unmasked count along axes.
func (m *Masked[T]) Mean(axes ...int) *Masked[T] {
	if len(axes) == 0 {
		axes = allAxes(m.Ndim())
	}
	f := beginFold(m, axes)
	walk(m.Shape(), func(coords []int) { f.add(m, coords, func(acc *T, v T) { *acc += v }) })
	for k, c := range f.count {
		if c != 0 {
			f.out.Data()[k] = scale(f.out.Data()[k], 1/float64(c))
		}
	}
	return f.finish()
}

// MeanKeepdims folds the mean keeping reduced axes as size 1.
func (m *Masked[T]) MeanKeepdims(axes ...int) *Masked[T] {
	return m.keepdims(m.Mean, axes...)
}

// keepdims re-shapes an axes-fold result with the reduced axes as size 1
// (root reduceKeepdims convention).
func (m *Masked[T]) keepdims(fold func(axes ...int) *Masked[T], axes ...int) *Masked[T] {
	if len(axes) == 0 {
		axes = allAxes(m.Ndim())
	}
	norm := normAxes(axes, m.Ndim())
	r := fold(axes...)
	shape := make([]int, m.Ndim())
	copy(shape, m.Shape())
	for _, ax := range norm {
		shape[ax] = 1
	}
	return newMasked(r.a.Reshape(shape...), r.mask)
}

// Var computes the unmasked population variance (ddof=0) along axes.
// Complex input panics like the root package (no real result in T).
func (m *Masked[T]) Var(axes ...int) *Masked[T] {
	return m.variance(0, axes)
}

// VarDDof computes the unmasked variance with the given ddof.
func (m *Masked[T]) VarDDof(ddof int, axes ...int) *Masked[T] {
	return m.variance(ddof, axes)
}

// Std computes the unmasked standard deviation along axes (ddof=0).
func (m *Masked[T]) Std(axes ...int) *Masked[T] {
	return m.stddev(0, axes)
}

// StdDDof computes the unmasked standard deviation with the given ddof.
func (m *Masked[T]) StdDDof(ddof int, axes ...int) *Masked[T] {
	return m.stddev(ddof, axes)
}

// variance folds unmasked sums and squares in float64 (real T only).
func (m *Masked[T]) variance(ddof int, axes []int) *Masked[T] {
	if isComplexKind[T]() {
		panic("numgo: Var/Std are not defined for complex arrays")
	}
	if len(axes) == 0 {
		axes = allAxes(m.Ndim())
	}
	norm := normAxes(axes, m.Ndim())
	shape := reduceShape(m.Shape(), norm)
	n := sizeOf(shape)
	sum := make([]float64, n)
	sumSq := make([]float64, n)
	count := make([]int, n)
	walk(m.Shape(), func(coords []int) {
		i := flat(m.Shape(), coords)
		if m.mask[i] {
			return
		}
		k := flat(shape, keptCoords(coords, m.Shape(), norm))
		v := realOf(m.a.At(coords...))
		sum[k] += v
		sumSq[k] += v * v
		count[k]++
	})
	out := numgo.Zeros[T](shape...)
	mask := make([]bool, n)
	for k, c := range count {
		denom := c - ddof
		if c == 0 || denom <= 0 {
			out.Data()[k] = nan[T]()
			mask[k] = true
			continue
		}
		mean := sum[k] / float64(c)
		out.Data()[k] = fromParts[T]((sumSq[k]-float64(c)*mean*mean)/float64(denom), 0)
	}
	return newMasked(out, mask)
}

// stddev is variance followed by a square root.
func (m *Masked[T]) stddev(ddof int, axes []int) *Masked[T] {
	v := m.variance(ddof, axes)
	for i, x := range v.a.Data() {
		v.a.Data()[i] = fromParts[T](math.Sqrt(realOf(x)), 0)
	}
	return v
}

// extreme folds max/min; an all-masked channel yields masked NaN.
func (m *Masked[T]) extreme(axes []int, better func(x, best T) bool) *Masked[T] {
	if len(axes) == 0 {
		axes = allAxes(m.Ndim())
	}
	norm := normAxes(axes, m.Ndim())
	shape := reduceShape(m.Shape(), norm)
	n := sizeOf(shape)
	out := numgo.Zeros[T](shape...)
	mask := make([]bool, n)
	seen := make([]bool, n)
	walk(m.Shape(), func(coords []int) {
		i := flat(m.Shape(), coords)
		if m.mask[i] {
			return
		}
		kept := keptCoords(coords, m.Shape(), norm)
		k := flat(shape, kept)
		v := m.a.Data()[i]
		if !seen[k] || better(v, out.Data()[k]) {
			out.Data()[k] = v
			seen[k] = true
		}
	})
	for k, ok := range seen {
		if !ok {
			out.Data()[k] = nan[T]()
			mask[k] = true
		}
	}
	return newMasked(out, mask)
}

// Max folds max along axes over unmasked elements (ties → first).
func (m *Masked[T]) Max(axes ...int) *Masked[T] {
	return m.extreme(axes, func(x, best T) bool { return realOf(x) > realOf(best) })
}

// MaxKeepdims folds max keeping reduced axes as size 1.
func (m *Masked[T]) MaxKeepdims(axes ...int) *Masked[T] {
	return m.keepdims(m.Max, axes...)
}

// Min folds min along axes over unmasked elements (ties → first).
func (m *Masked[T]) Min(axes ...int) *Masked[T] {
	return m.extreme(axes, func(x, best T) bool { return realOf(x) < realOf(best) })
}

// MinKeepdims folds min keeping reduced axes as size 1.
func (m *Masked[T]) MinKeepdims(axes ...int) *Masked[T] {
	return m.keepdims(m.Min, axes...)
}

// argExtreme shares the ArgMax/ArgMin body over one axis. All-masked
// channels report -1 (numpy returns a masked index instead).
func (m *Masked[T]) argExtreme(axis int, better func(x, best T) bool) []int {
	shape := m.Shape()
	ax := axis
	if ax < 0 {
		ax += len(shape)
	}
	if ax < 0 || ax >= len(shape) {
		panicAxis(axis, len(shape))
	}
	keptShape := reduceShape(shape, []int{ax})
	slots := sizeOf(keptShape)
	best := make([]T, slots)
	coord := make([]int, slots)
	seen := make([]bool, slots)
	walk(shape, func(coords []int) {
		i := flat(shape, coords)
		if m.mask[i] {
			return
		}
		k := flat(keptShape, keptCoords(coords, shape, []int{ax}))
		v := m.a.At(coords...)
		if !seen[k] || better(v, best[k]) {
			best[k] = v
			coord[k] = coords[ax]
			seen[k] = true
		}
	})
	for k := range coord {
		if !seen[k] {
			coord[k] = -1
		}
	}
	return coord
}

// ArgMax returns the coordinate of each unmasked maximum along axis
// (first occurrence wins ties); an all-masked channel gives -1 (numpy returns
// a masked index).
func (m *Masked[T]) ArgMax(axis int) []int {
	return m.argExtreme(axis, func(x, best T) bool { return realOf(x) > realOf(best) })
}

// ArgMin returns the coordinate of each unmasked minimum along axis
// (first occurrence wins ties); an all-masked channel gives -1.
func (m *Masked[T]) ArgMin(axis int) []int {
	return m.argExtreme(axis, func(x, best T) bool { return realOf(x) < realOf(best) })
}

// Any reports whether any UNMASKED element is nonzero. All-masked input
// returns false (a bool cannot carry numpy's masked answer).
func (m *Masked[T]) Any() bool {
	for i, bad := range m.mask {
		if bad {
			continue
		}
		re, im := parts(m.a.Data()[i])
		if re != 0 || im != 0 {
			return true
		}
	}
	return false
}

// All reports whether every UNMASKED element is nonzero; masked
// elements are skipped, so all-masked input is vacuously true.
func (m *Masked[T]) All() bool {
	for i, bad := range m.mask {
		if bad {
			continue
		}
		re, im := parts(m.a.Data()[i])
		if re == 0 && im == 0 {
			return false
		}
	}
	return true
}

// CumSum prefixes unmasked sums along axes (no axes = C-order prefix
// over the raveled array); masked positions carry the running value forward
// without contributing (numpy ma cumsum).
func (m *Masked[T]) CumSum(axes ...int) *Masked[T] {
	return m.accumulate(axes, func(acc *T, v T) { *acc += v })
}

// CumProd prefixes unmasked products along axes (same convention).
func (m *Masked[T]) CumProd(axes ...int) *Masked[T] {
	return m.accumulate(axes, func(acc *T, v T) { *acc *= v })
}

// accumulate walks the array in C order, carrying per-channel accumulators
// stored in the output itself (numpy cumsum shape = input shape; with no axes
// the result is 1-D over the raveled array).
func (m *Masked[T]) accumulate(axes []int, fn func(acc *T, v T)) *Masked[T] {
	srcShape := m.Shape()
	dstShape := srcShape
	if len(axes) == 0 {
		axes = allAxes(m.Ndim())
		dstShape = []int{m.Size()}
	}
	norm := normAxes(axes, m.Ndim())
	out := numgo.Zeros[T](dstShape...)
	mask := make([]bool, m.Size())
	// Channel id: the KEPT (non-folded) coordinates only — all positions in
	// one accumulation channel share it.
	acc := make(map[int]T)
	walk(srcShape, func(coords []int) {
		i := flat(srcShape, coords)
		ch := 0
		for k, c := range coords {
			if !isFolded(norm, k) {
				ch = ch*srcShape[k] + c
			}
		}
		dst := make([]int, len(dstShape))
		copy(dst, coords)
		if len(dst) > len(dstShape) {
			dst = dst[len(dst)-len(dstShape):]
		}
		di := flat(dstShape, dst)
		if m.mask[i] {
			if cur, ok := acc[ch]; ok {
				out.Data()[di] = cur
			}
			mask[di] = true
			return
		}
		if cur, ok := acc[ch]; ok {
			next := cur
			fn(&next, m.a.Data()[i])
			acc[ch] = next
			out.Data()[di] = next
		} else {
			acc[ch] = m.a.Data()[i]
			out.Data()[di] = m.a.Data()[i]
		}
	})
	return newMasked(out, mask)
}

// keptCoords drops the folded axes from coords.
func keptCoords(coords, shape []int, norm []int) []int {
	drop := make([]bool, len(shape))
	for _, ax := range norm {
		drop[ax] = true
	}
	out := make([]int, 0, len(coords))
	for i, c := range coords {
		if !drop[i] {
			out = append(out, c)
		}
	}
	return out
}

func isFolded(norm []int, ax int) bool {
	return slices.Contains(norm, ax)
}

// nan builds NaN as T.
func nan[T numgo.Number]() T {
	return fromParts[T](math.NaN(), 0)
}

// scale multiplies a real T by a float factor.
func scale[T numgo.Number](v T, f float64) T {
	re, im := parts(v)
	return fromParts[T](re*f, im*f)
}
