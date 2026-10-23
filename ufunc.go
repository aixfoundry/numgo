package numgo

import (
	"slices"

	"github.com/kaisawind/numgo/internal/nditer"
)

// Pure: Add returns a new array with the element-wise sum of a and b, broadcasting
// both operands with full numpy right-alignment.
func (a *Array[T]) Add(b *Array[T]) *Array[T] {
	la, lb, shape := broadcastPair(a, b)
	return zip2(la, a.data, lb, b.data, shape, func(x, y T) T { return x + y })
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

// zip2 broadcasts a and b, then feeds an op over them into fresh storage.
func zip2[T Number](la *nditer.Layout, adata []T, lb *nditer.Layout, bdata []T,
	shape []int, op func(x, y T) T) *Array[T] {
	dst := newArray(shape, nditer.COrderStrides(shape), 0, make([]T, nditer.Size(shape)))
	if dst.Size() == 0 {
		return dst
	}
	nditer.Zip(dst.layout(), dst.data, []*nditer.Layout{la, lb}, [][]T{adata, bdata},
		func(d []T, src [][]T) {
			d[0] = op(src[0][0], src[1][0])
		})
	return dst
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
			d[0] = op(src[0][0], src[1][0])
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
			d[0] = f(src[0][0])
		})
	return out
}
