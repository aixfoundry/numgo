package numgo

import (
	"slices"

	"github.com/aixfoundry/numgo/internal/nditer"
	"github.com/aixfoundry/numgo/internal/simdvec"
)

// Pure: Add returns a new array with the element-wise sum of a and b, broadcasting
// both operands with full numpy right-alignment.
func (a *Array[T]) Add(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	dst := newArray(shape, nditer.COrderStrides(shape), 0, make([]T, nditer.Size(shape)))
	// SIMD fast path: float64 add with all three arrays C-contiguous over the
	// same shape. a[i]+b[i] is exact IEEE and order-free, so the result is
	// bit-identical to the generic path; any other layout keeps the nditer
	// route. Only Add takes it — Sub/Mul/Div stay on zip2.
	if dst.Size() > 0 && simdAddF64(dst, la, a.data, lb, b.data, shape) {
		return dst
	}
	return zip2Into(dst, la, a.data, lb, b.data, shape, func(x, y T) T { return x + y })
}

// Pure: Sub returns a new array with the element-wise difference of a and b.
func (a *Array[T]) Sub(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, func(x, y T) T { return x - y })
}

// Pure: Mul returns a new array with the element-wise product of a and b.
func (a *Array[T]) Mul(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, func(x, y T) T { return x * y })
}

// Pure: Div returns a new array with the element-wise quotient of a and b. Division
// by zero follows IEEE (Inf/NaN), never panics.
func (a *Array[T]) Div(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, func(x, y T) T { return x / y })
}

// In-place: AddInto writes the broadcast sum of a and b into dst and returns dst. dst
// must already have the broadcast result shape.
func (a *Array[T]) AddInto(dst *Array[T], b *Array[T]) *Array[T] {
	return into2(dst, a, b, func(x, y T) T { return x + y })
}

// In-place: SubInto writes the broadcast difference of a and b into dst and returns dst.
func (a *Array[T]) SubInto(dst *Array[T], b *Array[T]) *Array[T] {
	return into2(dst, a, b, func(x, y T) T { return x - y })
}

// In-place: MulInto writes the broadcast product of a and b into dst and returns dst.
func (a *Array[T]) MulInto(dst *Array[T], b *Array[T]) *Array[T] {
	return into2(dst, a, b, func(x, y T) T { return x * y })
}

// In-place: DivInto writes the broadcast quotient of a and b into dst and returns dst.
func (a *Array[T]) DivInto(dst *Array[T], b *Array[T]) *Array[T] {
	return into2(dst, a, b, func(x, y T) T { return x / y })
}

// Pure: AddValue returns a new array with v added to every element.
func (a *Array[T]) AddValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return x + v })
}

// Pure: SubValue returns a new array with v subtracted from every element.
func (a *Array[T]) SubValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return x - v })
}

// Pure: MulValue returns a new array with every element multiplied by v.
func (a *Array[T]) MulValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return x * v })
}

// Pure: DivValue returns a new array with every element divided by v. Division by
// zero follows IEEE (Inf/NaN), never panics.
func (a *Array[T]) DivValue(v T) *Array[T] {
	return mapValues(a, func(x T) T { return x / v })
}

// broadcastPair aligns a and b, panicking with the numpy message when their
// shapes are incompatible.
func broadcastPair[T Number](a, b *Array[T]) (*nditer.Layout, *nditer.Layout, []int) {
	shape, strides, err := nditer.Broadcast(a.shape, b.shape)
	if err != nil {
		panicBroadcast(a.shape, b.shape)
	}
	return &nditer.Layout{Shape: shape, Strides: strides[0]},
		&nditer.Layout{Shape: shape, Strides: strides[1]}, shape
}

// zip2 broadcasts a and b, allocates fresh storage, then feeds an op over them.
func zip2[T Number](la *nditer.Layout, adata []T, lb *nditer.Layout, bdata []T,
	shape []int, op func(x, y T) T) *Array[T] {
	dst := newArray(shape, nditer.COrderStrides(shape), 0, make([]T, nditer.Size(shape)))
	if dst.Size() == 0 {
		return dst
	}
	return zip2Into(dst, la, adata, lb, bdata, shape, op)
}

// zip2Into feeds op over broadcast operands into the preallocated dst.
func zip2Into[T Number](dst *Array[T], la *nditer.Layout, adata []T,
	lb *nditer.Layout, bdata []T, shape []int, op func(x, y T) T) *Array[T] {
	nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{la, lb}, [][]T{adata, bdata},
		func(d []T, src [][]T) {
			for i := range d {
				d[i] = op(src[0][i], src[1][i])
			}
		})
	return dst
}

// simdAddF64 runs simdvec.AddF64 when dst, a and b are all C-contiguous full
// slices of float64 over the same shape; otherwise it reports false and the
// caller falls back to the generic nditer path.
func simdAddF64[T Number](dst *Array[T], la *nditer.Layout, adata []T,
	lb *nditer.Layout, bdata []T, shape []int) bool {
	n := dst.Size()
	a, aok := any(adata).([]float64)
	b, bok := any(bdata).([]float64)
	d, dok := any(dst.data).([]float64)
	if !aok || !bok || !dok {
		return false
	}
	if !contigFull(la, a, shape, n) || !contigFull(lb, b, shape, n) ||
		!nditer.IsContiguous(dst.layout()) || len(dst.data) < n+dst.offset {
		return false
	}
	simdvec.AddF64(d[dst.offset:dst.offset+n], a[la.Offset:la.Offset+n], b[lb.Offset:lb.Offset+n])
	return true
}

// contigFull reports whether l addresses exactly shape in C order with no
// broadcasting and data has room for the whole run.
func contigFull[T any](l *nditer.Layout, data []T, shape []int, n int) bool {
	return nditer.ContiguousSame(l, shape) && len(data) >= n+l.Offset
}

// into2 broadcasts a and b and writes the op result into dst.
func into2[T Number](dst, a, b *Array[T], op func(x, y T) T) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	if !slices.Equal(dst.shape, shape) {
		panicInto(dst.shape, shape)
	}
	if dst.Size() == 0 {
		return dst
	}
	nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{la, lb}, [][]T{a.data, b.data},
		func(d []T, src [][]T) {
			for i := range d {
				d[i] = op(src[0][i], src[1][i])
			}
		})
	return dst
}

// mapValues copies a and applies f to every element.
func mapValues[T Number](a *Array[T], f func(T) T) *Array[T] {
	out := newArray(a.Shape(), nditer.COrderStrides(a.shape), 0, make([]T, a.Size()))
	if out.Size() == 0 {
		return out
	}
	nditer.Zip(out.layout(), out.data, []*nditer.Layout{a.layout()}, [][]T{a.data},
		func(d []T, src [][]T) {
			for i := range d {
				d[i] = f(src[0][i])
			}
		})
	return out
}
