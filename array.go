package numgo

import (
	"iter"
)

// Array represents a 2D matrix with rows (r) and columns (c).
// Example:
//
//	arr := NewArray(2, 2, []float64{1, 2, 3, 4})
type Array struct {
	r, c int       // number of rows and columns
	data []float64 // data stored in row-major order
}

// Copy creates a new copy of the given Array.
// Example:
//
//	copyArr := Copy(arr)
func Copy(a *Array) *Array {
	return a.Copy()
}

// Copy creates a new Array that is a deep copy of the original Array.
// It returns a pointer to the new Array.
//
// Example:
//
//	original := &Array{c: 3, r: 2, data: []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}}
//	copied := original.Copy()
//	// Now `copied` is a separate instance with the same data as `original`
func (a *Array) Copy() *Array {
	x := &Array{
		c:    a.c,
		r:    a.r,
		data: make([]float64, len(a.data)),
	}
	copy(x.data, a.data)
	return x
}

// Reshape changes the dimensions of the Array to the specified rows and columns.
// EmptyArray creates a new Array with specified rows and columns filled with zeros.
// Example:
//
//	emptyArr := EmptyArray(2, 3)
func EmptyArray(r, c int) (a *Array) {
	return &Array{
		r: r,
		c: c,
	}
}

// ZeroArray creates a new Array with specified rows and columns filled with zeros.
// Example:
//
//	zeroArr := ZeroArray(2, 3)
func ZeroArray(r, c int) (a *Array) {
	return &Array{
		r:    r,
		c:    c,
		data: make([]float64, r*c),
	}
}

// RangeArray creates a new Array with a range of values.
// Depending on the number of arguments provided:
// - No arguments: Returns a zeroed Array with dimensions (0, 0).
// - One argument: Returns a 1D Array with the specified number of elements.
// - Two arguments: Returns a 1D Array from start to stop (exclusive).
// - Three arguments: Returns a 1D Array from start to stop (exclusive) with a specified step.
//
// Examples:
//
//	RangeArray() // Returns a zeroed Array with dimensions [].
//	RangeArray(5) // Returns a 1D Array with elements [0, 1, 2, 3, 4].
//	RangeArray(1, 5) // Returns a 1D Array with elements [1, 2, 3, 4].
//	RangeArray(1, 5, 2) // Returns a 1D Array with elements [1, 3].
func RangeArray(args ...int) (a *Array) {
	if len(args) == 0 {
		a = ZeroArray(0, 0)
		return
	} else if len(args) == 1 {
		data := []float64{}
		for i := range args[0] {
			data = append(data, float64(i))
		}
		a = New1DArray(data)
		return
	} else if len(args) == 2 {
		start, stop := args[0], args[1]
		data := []float64{}
		for i := start; i < stop; i++ {
			data = append(data, float64(i))
		}
		a = New1DArray(data)
	} else if len(args) == 3 {
		start, stop, step := args[0], args[1], args[2]
		data := []float64{}
		for i := start; i < stop; i += step {
			data = append(data, float64(i))
		}
		a = New1DArray(data)
	} else {
		a = ZeroArray(0, 0)
	}
	return
}

// NewValueArray creates a new Array with dimensions r (rows) and c (columns),
// initializing all elements to the specified value.
//
// Example:
//
//	a := NewValueArray(2, 3, 5.0)
//	// a will be:
//	// [[5.0 5.0 5.0]
//	//  [5.0 5.0 5.0]]
func NewValueArray(r, c int, value float64) (a *Array) {
	a = ZeroArray(r, c)
	for i := range a.data {
		a.data[i] = value
	}
	return
}

// NewArray creates a new Array with specified rows, columns, and data.
// If the data length is less than r*c, it is padded with zeros.
// If the data length is more than r*c, it is truncated.
//
// Example:
//
//	arr := NewArray(2, 2, []float64{1, 2, 3, 4})
func NewArray(r, c int, data []float64) (a *Array) {
	if r*c > len(data) {
		buf := make([]float64, r*c-len(data))
		data = append(data, buf...)
	} else if r*c < len(data) {
		data = data[:r*c]
	}
	return &Array{
		r:    r,
		c:    c,
		data: data,
	}
}

// New1DArray creates a new 1D array from a slice of float64.
//
// Example:
//
//	data := []float64{1.0, 2.0, 3.0}
//	array := New1DArray(data)
func New1DArray(data []float64) (a *Array) {
	return &Array{
		r:    1,
		c:    len(data),
		data: data,
	}
}

// New2DArray creates a new 2D array from a slice of slices of float64.
// It flattens the 2D data into a 1D slice internally.
//
// Example:
//
//	data := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
//	array := New2DArray(data)
func New2DArray(data [][]float64) (a *Array) {
	if (len(data) == 0) || (len(data[0]) == 0) {
		a = EmptyArray(0, 0)
		return
	}
	buf := []float64{}
	for _, row := range data {
		buf = append(buf, row...)
	}
	return &Array{
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
func (a *Array) Dims() (r, c int) {
	return a.r, a.c
}

// At returns the value at the specified row (i) and column (j).
//
// Example:
//
//	value := arr.At(0, 1)
func (a *Array) At(i, j int) float64 {
	return a.data[i*a.c+j]
}

// Set sets the value at the specified row (i) and column (j).
//
// Example:
//
//	arr.Set(0, 1, 5)
func (a *Array) Set(i, j int, f float64) *Array {
	a.data[i*a.c+j] = f
	return a
}

// Data returns the underlying data slice of the Array.
//
// Example:
//
//	data := arr.Data()
func (a *Array) Data() []float64 {
	return a.data
}

// Of returns a sub-array based on the specified axis and index.
// It returns a column if axis is 0, and a row if axis is 1.
//
// Example:
//
//	row := arr.Of(0, 0) // returns the first column
//	col := arr.Of(1, 1) // returns the second row
//
// numpy: a[i], a[:, i]
//
// axis:
//
//   - 0: by column
//
//   - 1: by row
func (a *Array) Of(i int, axis int) (ret *Array) {
	buf := []float64{}
	if axis == 0 {
		for ir := range a.r {
			for ic := range a.c {
				if ic == i {
					buf = append(buf, a.At(ir, ic))
					break
				}
			}
		}
		ret = NewArray(a.r, 1, buf)
	} else if axis == 1 {
		for ir := range a.r {
			if ir == i {
				for ic := range a.c {
					buf = append(buf, a.At(ir, ic))
				}
				break
			}
		}
		ret = NewArray(1, a.c, buf)
	}
	return
}

// Row returns a new Array that contains the i-th row of the original array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}, {5, 6}}) // Assume a is a valid Array with multiple rows
//	row := a.Row(2)   // row will contain the elements of the third row
func (a *Array) Row(i int) *Array {
	return a.Of(i, 1)
}

// Col returns a new Array that contains the i-th column of the original array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}, {5, 6}}) // Assume a is a valid Array with multiple columns
//	col := a.Col(1)   // col will contain the elements of the second column
func (a *Array) Col(i int) *Array {
	return a.Of(i, 0)
}

// All returns a sequence that iterates over all elements of the Array.
// Each element is represented by a tuple of its row and column indices and its value.
//
// Example:
//
//	a := NewArray(...) // Assume NewArray creates a new Array
//	seq := a.All()
//
//	seq(func(indices [2]int, value float64) bool {
//	    fmt.Printf("Index: %v, Value: %v\n", indices, value)
//	    return true // Return false to stop iteration early
//	})
func (a *Array) All() iter.Seq2[[2]int, float64] {
	return func(yield func([2]int, float64) bool) {
		for ir := range a.r {
			for ic := range a.c {
				if !yield([2]int{ir, ic}, a.At(ir, ic)) {
					return
				}
			}
		}
	}
}

// All returns a sequence that iterates over all elements of the Array.
// Each element is represented by a tuple of its row and column indices and its value.
//
// Example:
//
//	a := NewArray(...) // Assume NewArray creates a new Array
//	seq := All(a)
//
//	seq(func(indices [2]int, value float64) bool {
//	    fmt.Printf("Index: %v, Value: %v\n", indices, value)
//	    return true // Return false to stop iteration early
//	})
func All(a *Array) iter.Seq2[[2]int, float64] {
	return a.All()
}
