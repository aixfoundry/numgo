package numgo

import (
	"fmt"
	"iter"
	"slices"
)

// Float is the element type constraint for Array.
// Integers are excluded on purpose: Mean/Div on an int array would silently
// truncate (Go has no numpy-style dtype promotion), so only floats are safe.
type Float interface {
	~float32 | ~float64
}

// toF64 converts v to float64 for math functions; call sites convert back to T.
func toF64[T Float](v T) float64 { return float64(v) }

// Array represents a 2D matrix with rows (r) and columns (c).
// Data is stored in row-major order.
//
// Example:
//
//	arr := NewArray(2, 2, []float64{1, 2, 3, 4})
type Array[T Float] struct {
	r, c int
	data []T
}

// Copy creates a new copy of the given Array.
//
// Example:
//
//	copyArr := Copy(arr)
func Copy[T Float](a *Array[T]) *Array[T] {
	return a.Copy()
}

// Copy creates a new Array that is a deep copy of the original Array.
//
// Example:
//
//	copied := original.Copy()
func (a *Array[T]) Copy() *Array[T] {
	return &Array[T]{
		r:    a.r,
		c:    a.c,
		data: slices.Clone(a.data),
	}
}

// ZeroArray creates a new Array with specified rows and columns filled with zeros.
//
// Example:
//
//	zeroArr := ZeroArray(2, 3)
func ZeroArray[T Float](r, c int) (a *Array[T]) {
	return &Array[T]{
		r:    r,
		c:    c,
		data: make([]T, r*c),
	}
}

// RangeArray creates a new Array with a range of values.
// Depending on the number of arguments provided:
//
//   - No arguments: Returns a zeroed Array with dimensions (0, 0).
//   - One argument: Returns a 1D Array with the specified number of elements.
//   - Two arguments: Returns a 1D Array from start to stop (exclusive).
//   - Three arguments: Returns a 1D Array from start to stop (exclusive) with a specified step.
//
// Examples:
//
//	RangeArray[int]()    // Returns a zeroed Array with dimensions [].
//	RangeArray(5)        // Returns a 1D Array with elements [0, 1, 2, 3, 4].
//	RangeArray(1.0, 5.0) // Returns a 1D Array with elements [1, 2, 3, 4].
//	RangeArray(1, 5, 2)  // Returns a 1D Array with elements [1, 3].
func RangeArray[T Float](args ...T) (a *Array[T]) {
	toI := func(v T) int { return int(v) }
	switch len(args) {
	case 0:
		a = ZeroArray[T](0, 0)
	case 1:
		data := make([]T, toI(args[0]))
		for i := range data {
			data[i] = T(i)
		}
		a = New1DArray(data)
	case 2:
		start, stop := toI(args[0]), toI(args[1])
		data := make([]T, 0, max(stop-start, 0))
		for i := start; i < stop; i++ {
			data = append(data, T(i))
		}
		a = New1DArray(data)
	case 3:
		start, stop, step := toI(args[0]), toI(args[1]), toI(args[2])
		if step == 0 {
			panic("numgo: RangeArray: step must not be zero")
		}
		data := make([]T, 0, max((stop-start)/step, 0))
		for i := start; i < stop; i += step {
			data = append(data, T(i))
		}
		a = New1DArray(data)
	default:
		a = ZeroArray[T](0, 0)
	}
	return
}

// NewValueArray creates a new Array with dimensions r (rows) and c (columns),
// initializing all elements to the specified value.
//
// Example:
//
//	a := NewValueArray(2, 3, 5.0)
func NewValueArray[T Float](r, c int, value T) (a *Array[T]) {
	a = ZeroArray[T](r, c)
	for i := range a.data {
		a.data[i] = value
	}
	return
}

// NewArray creates a new Array with specified rows, columns, and data.
// If the data length is less than r*c, it is padded with zeros.
// If the data length is more than r*c, it is truncated.
// The data slice is copied.
//
// Example:
//
//	arr := NewArray(2, 2, []float64{1, 2, 3, 4})
func NewArray[T Float](r, c int, data []T) (a *Array[T]) {
	buf := make([]T, r*c)
	copy(buf, data)
	return &Array[T]{
		r:    r,
		c:    c,
		data: buf,
	}
}

// New1DArray creates a new 1D array from a slice.
// The slice is copied, so later changes to the input do not affect the Array.
//
// Example:
//
//	data := []float64{1.0, 2.0, 3.0}
//	array := New1DArray(data)
func New1DArray[T Float](data []T) (a *Array[T]) {
	return &Array[T]{
		r:    1,
		c:    len(data),
		data: slices.Clone(data),
	}
}

// New2DArray creates a new 2D array from a slice of slices.
// It flattens the 2D data into a 1D slice internally.
// Panics if the rows have unequal lengths.
//
// Example:
//
//	data := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
//	array := New2DArray(data)
func New2DArray[T Float](data [][]T) (a *Array[T]) {
	if len(data) == 0 || len(data[0]) == 0 {
		a = ZeroArray[T](0, 0)
		return
	}
	buf := make([]T, 0, len(data)*len(data[0]))
	for _, row := range data {
		if len(row) != len(data[0]) {
			panic(fmt.Sprintf("numgo: New2DArray: jagged input: row has %d elements, want %d", len(row), len(data[0])))
		}
		buf = append(buf, row...)
	}
	return &Array[T]{
		r:    len(data),
		c:    len(data[0]),
		data: buf,
	}
}

// Dims returns the dimensions (rows, columns) of the Array.
//
// Example:
//
//	rows, cols := arr.Dims()
func (a *Array[T]) Dims() (r, c int) {
	return a.r, a.c
}

// checkIndex panics with a numpy-style message when (i, j) is out of bounds.
func (a *Array[T]) checkIndex(i, j int) {
	if i < 0 || i >= a.r || j < 0 || j >= a.c {
		panic(fmt.Sprintf("numgo: index (%d,%d) is out of bounds for shape (%d,%d)", i, j, a.r, a.c))
	}
}

// At returns the value at the specified row (i) and column (j).
// Panics if the index is out of bounds.
//
// Example:
//
//	value := arr.At(0, 1)
func (a *Array[T]) At(i, j int) T {
	a.checkIndex(i, j)
	return a.data[i*a.c+j]
}

// Set sets the value at the specified row (i) and column (j).
// Panics if the index is out of bounds.
//
// Example:
//
//	arr.Set(0, 1, 5)
func (a *Array[T]) Set(i, j int, f T) *Array[T] {
	a.checkIndex(i, j)
	a.data[i*a.c+j] = f
	return a
}

// Data returns the underlying data slice of the Array.
//
// Example:
//
//	data := arr.Data()
func (a *Array[T]) Data() []T {
	return a.data
}

// Of returns a sub-array based on the specified axis and index.
// It returns a column if axis is 0, and a row if axis is 1.
// Panics if axis is not 0 or 1.
//
// Example:
//
//	col := arr.Of(0, 0) // returns the first column
//	row := arr.Of(1, 1) // returns the second row
//
// numpy: a[:, i], a[i]
func (a *Array[T]) Of(i int, axis int) *Array[T] {
	switch axis {
	case 0:
		if i < 0 || i >= a.c {
			panic(fmt.Sprintf("numgo: index (%d) is out of bounds for axis 1 with size %d", i, a.c))
		}
		buf := make([]T, a.r)
		for ir := range a.r {
			buf[ir] = a.data[ir*a.c+i]
		}
		return NewArray(a.r, 1, buf)
	case 1:
		if i < 0 || i >= a.r {
			panic(fmt.Sprintf("numgo: index (%d) is out of bounds for axis 0 with size %d", i, a.r))
		}
		return NewArray(1, a.c, a.data[i*a.c:(i+1)*a.c])
	default:
		panic(fmt.Sprintf("numgo: axis must be 0 or 1, got %d", axis))
	}
}

// Row returns a new Array that contains the i-th row of the original array.
//
// Example:
//
//	row := a.Row(2)
func (a *Array[T]) Row(i int) *Array[T] {
	return a.Of(i, 1)
}

// Col returns a new Array that contains the i-th column of the original array.
//
// Example:
//
//	col := a.Col(1)
func (a *Array[T]) Col(i int) *Array[T] {
	return a.Of(i, 0)
}

// All returns a sequence that iterates over all elements of the Array.
// Each element is represented by a tuple of its row and column indices and its value.
//
// Example:
//
//	for k, v := range a.All() {
//	    fmt.Println(k, v)
//	}
func (a *Array[T]) All() iter.Seq2[[2]int, T] {
	return func(yield func([2]int, T) bool) {
		for ir := range a.r {
			for ic := range a.c {
				if !yield([2]int{ir, ic}, a.data[ir*a.c+ic]) {
					return
				}
			}
		}
	}
}

// All returns a sequence that iterates over all elements of the Array.
//
// Example:
//
//	for k, v := range All(a) {
//	    fmt.Println(k, v)
//	}
func All[T Float](a *Array[T]) iter.Seq2[[2]int, T] {
	return a.All()
}

// zip applies op element-wise between a and b, broadcasting b against a
// when b is a single row, a single column, or a 1x1 array (numpy semantics).
// Panics when the shapes cannot be broadcast together.
func (a *Array[T]) zip(b *Array[T], op func(x, y T) T) *Array[T] {
	if !((a.r == b.r || b.r == 1) && (a.c == b.c || b.c == 1)) {
		panic(fmt.Sprintf("numgo: operands could not be broadcast together with shapes (%d,%d) (%d,%d)", a.r, a.c, b.r, b.c))
	}
	ret := ZeroArray[T](a.r, a.c)
	for i := range a.r {
		bi := i
		if b.r == 1 {
			bi = 0
		}
		for j := range a.c {
			bj := j
			if b.c == 1 {
				bj = 0
			}
			ret.data[i*a.c+j] = op(a.data[i*a.c+j], b.data[bi*b.c+bj])
		}
	}
	return ret
}

// zipValue applies op element-wise between a and a scalar.
func (a *Array[T]) zipValue(value T, op func(x, y T) T) *Array[T] {
	ret := ZeroArray[T](a.r, a.c)
	for i, v := range a.data {
		ret.data[i] = op(v, value)
	}
	return ret
}
