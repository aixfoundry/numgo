package numgo

import (
	"fmt"
)

// Array represents a 2D matrix with rows (r) and columns (c).
// Example:
//
//	arr := NewArray(2, 2, []float64{1, 2, 3, 4})
type Array struct {
	r, c int       // number of rows and columns
	data []float64 // data stored in row-major order
}

// CopyOf creates a new copy of the given Array.
// Example:
//
//	copyArr := CopyOf(arr)
func CopyOf(a *Array) (x *Array) {
	x = &Array{
		c:    a.c,
		r:    a.r,
		data: make([]float64, len(a.data)),
	}
	copy(x.data, a.data)
	return
}

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
// Example:
//
//	rows, cols := arr.Dims()
func (a *Array) Dims() (r, c int) {
	return a.r, a.c
}

// At returns the value at the specified row (i) and column (j).
// Example:
//
//	value := arr.At(0, 1)
func (a *Array) At(i, j int) float64 {
	return a.data[i*a.c+j]
}

// Set sets the value at the specified row (i) and column (j).
// Example:
//
//	arr.Set(0, 1, 5)
func (a *Array) Set(i, j int, f float64) *Array {
	a.data[i*a.c+j] = f
	return a
}

// Data returns the underlying data slice of the Array.
// Example:
//
//	data := arr.Data()
func (a *Array) Data() []float64 {
	return a.data
}

// Format provides a custom string representation of the Array.
// Example:
//
//	fmt.Printf("%v", arr)
func (a *Array) Format(fs fmt.State, c rune) {
	a.format("", 0, '.', true, fs, c)
}

// Of returns a sub-array based on the specified axis and index.
// It returns a row if axis is 0, and a column if axis is 1.
// Example:
//
//	row := arr.Of(0, 0) // returns the first row
//	col := arr.Of(1, 1) // returns the second column
//
// numpy: a[i], a[:, i]
//
// axis:
//
//   - 0: by row
//
//   - 1: by column
func (a *Array) Of(i int, axis int) *Array {
	buf := []float64{}
	if axis == 0 {
		for ir := range a.r {
			if ir == i {
				for ic := range a.c {
					buf = append(buf, a.At(ir, ic))
				}
				break
			}
		}
		return NewArray(1, a.c, buf)
	} else {
		for ir := range a.r {
			for ic := range a.c {
				if ic == i {
					buf = append(buf, a.At(ir, ic))
					break
				}
			}
		}
		return NewArray(a.r, 1, buf)
	}
}
