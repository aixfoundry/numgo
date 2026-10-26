package numgo

import (
	"slices"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// This file holds the conditional selection family. cond arguments are
// flattened C-order []bool masks, matching SetMask/MaskSelect and the
// comparison predicates in compare.go.

// advance steps coords through shape in C order, wrapping to all zeros when
// the walk completes.
func advance(coords, shape []int) {
	for k := len(coords) - 1; k >= 0; k-- {
		coords[k]++
		if coords[k] < shape[k] {
			return
		}
		coords[k] = 0
	}
}

// Where selects element-wise between x and y from a flattened C-order mask
// (numpy np.where): cond[k] true yields x's element at broadcast position k,
// false yields y's. len(cond) must equal the broadcast size; the output has
// the broadcast shape and C-order layout.
func Where[T Number](cond []bool, x, y *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(x, y)
	n := nditer.Size(shape)
	if len(cond) != n {
		panicCompressMask(len(cond), n)
	}
	dst := newArray(shape, nditer.COrderStrides(shape), 0, make([]T, n))
	coords := make([]int, len(shape))
	for k := 0; k < n; k++ {
		if cond[k] {
			dst.data[k] = x.data[nditer.Index(la, coords)]
		} else {
			dst.data[k] = y.data[nditer.Index(lb, coords)]
		}
		advance(coords, shape)
	}
	return dst
}

// NonZero returns the flat C-order indices of the nonzero elements (numpy
// np.flatnonzero, the raveled form of np.nonzero). Zero means == 0 in every
// component, so NaN and non-real complex values count as nonzero, like numpy.
func (a *Array[T]) NonZero() []int {
	var out []int
	coords := make([]int, len(a.shape))
	for k := 0; k < a.Size(); k++ {
		if a.At(coords...) != 0 {
			out = append(out, k)
		}
		advance(coords, a.shape)
	}
	return out
}

// ArgWhere returns the full coordinates of every nonzero element in C order
// (numpy np.argwhere); nonzero follows the same component-wise rule as
// NonZero.
func (a *Array[T]) ArgWhere() [][]int {
	var out [][]int
	coords := make([]int, len(a.shape))
	for k := 0; k < a.Size(); k++ {
		if a.At(coords...) != 0 {
			out = append(out, slices.Clone(coords))
		}
		advance(coords, a.shape)
	}
	return out
}
