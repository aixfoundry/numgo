package numgo

import (
	"math"
	"slices"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// Zeros returns a new array of the given shape filled with zeros.
func Zeros[T Number](shape ...int) *Array[T] {
	return newArray(slices.Clone(shape), nditer.COrderStrides(shape), 0, make([]T, nditer.Size(shape)))
}

// Ones returns a new array of the given shape filled with ones.
func Ones[T Number](shape ...int) *Array[T] {
	a := Zeros[T](shape...)
	for i := range a.data {
		a.data[i] = 1
	}
	return a
}

// Full returns a new array of the given shape filled with v.
func Full[T Number](v T, shape ...int) *Array[T] {
	a := Zeros[T](shape...)
	for i := range a.data {
		a.data[i] = v
	}
	return a
}

// Arange returns evenly spaced values in [start, stop), stepping by step.
// Complex operands are ordered by their real part.
func Arange[T Number](start, stop, step T) *Array[T] {
	startRe, startIm := parts(start)
	stopRe, _ := parts(stop)
	stepRe, stepIm := parts(step)
	if stepRe == 0 {
		panicZeroStep()
	}
	n := 0
	if stepRe > 0 && stopRe > startRe {
		n = int(math.Ceil((stopRe - startRe) / stepRe))
	} else if stepRe < 0 && stopRe < startRe {
		n = int(math.Ceil((startRe - stopRe) / (-stepRe)))
	}
	data := make([]T, n)
	for i := range data {
		data[i] = scalarFromParts[T](startRe+float64(i)*stepRe, startIm+float64(i)*stepIm)
	}
	return New1DArray(data)
}

// Linspace returns num evenly spaced values from start to stop inclusive.
func Linspace[T Number](start, stop T, num int) *Array[T] {
	if num < 0 {
		panicSamples(num)
	}
	data := make([]T, num)
	if num == 0 {
		return New1DArray(data)
	}
	if num == 1 {
		data[0] = start
		return New1DArray(data)
	}
	startRe, startIm := parts(start)
	stopRe, stopIm := parts(stop)
	for i := range data {
		t := float64(i) / float64(num-1)
		data[i] = scalarFromParts[T](startRe+(stopRe-startRe)*t, startIm+(stopIm-startIm)*t)
	}
	return New1DArray(data)
}

// Eye returns the n-by-n identity matrix.
func Eye[T Number](n int) *Array[T] {
	a := Zeros[T](n, n)
	for i := range n {
		a.data[nditer.Index(a.layout(), []int{i, i})] = 1
	}
	return a
}

// FromSlice returns a new array holding a copy of data. With no shape the
// result is 1-D; otherwise the element count must match.
func FromSlice[T Number](data []T, shape ...int) *Array[T] {
	if len(shape) == 0 {
		shape = []int{len(data)}
	}
	if nditer.Size(shape) != len(data) {
		panicReshape(len(data), shape)
	}
	return newArray(slices.Clone(shape), nditer.COrderStrides(shape), 0, slices.Clone(data))
}

// New1DArray returns a new 1-D array holding a copy of data.
func New1DArray[T Number](data []T) *Array[T] {
	return FromSlice(data, len(data))
}

// New2DArray returns a new 2-D array holding a copy of rows; it panics on
// jagged input.
func New2DArray[T Number](rows [][]T) *Array[T] {
	if len(rows) == 0 || len(rows[0]) == 0 {
		return Zeros[T](0, 0)
	}
	cols := len(rows[0])
	data := make([]T, 0, len(rows)*cols)
	for _, row := range rows {
		if len(row) != cols {
			panicJagged(len(row), cols)
		}
		data = append(data, row...)
	}
	return FromSlice(data, len(rows), cols)
}

// Identity returns the n-by-n matrix with ones on the diagonal. Eye with a
// single argument; no Empty constructor exists because Go's make already zeroes.
func Identity[T Number](n int) *Array[T] {
	return Eye[T](n)
}

// FromFunction fills a new array in C order, calling f with the coordinates
// of each element.
func FromFunction[T Number](shape []int, f func(coords ...int) T) *Array[T] {
	a := Zeros[T](shape...)
	if a.Size() == 0 {
		return a
	}
	coords := make([]int, len(a.shape))
	for i := range a.data {
		a.data[i] = f(coords...)
		for k := len(coords) - 1; k >= 0; k-- {
			coords[k]++
			if coords[k] < a.shape[k] {
				break
			}
			coords[k] = 0
		}
	}
	return a
}

// MeshGrid returns len(as) arrays with 'ij' (matrix) indexing semantics:
// result[k] has shape len(as[0]) × … × len(as[n-1]) and varies only along
// dimension k, taking as[k]'s values there.
func MeshGrid[T Number](as ...*Array[T]) []*Array[T] {
	shape := make([]int, len(as))
	for k, a := range as {
		if a.Ndim() != 1 {
			panicMeshGrid()
		}
		shape[k] = a.Size()
	}
	out := make([]*Array[T], len(as))
	for k, a := range as {
		v := a // capture for the closure below
		out[k] = FromFunction(shape, func(coords ...int) T {
			return v.At(coords[k])
		})
	}
	return out
}
