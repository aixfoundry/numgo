package numgo

import (
	"fmt"
	"slices"
)

func checkAxis(axis int) {
	if axis != 0 && axis != 1 {
		panic(fmt.Sprintf("numgo: axis must be 0 or 1, got %d", axis))
	}
}

// T returns the transpose of the matrix.
// Example:
//
//	a := NewArray(2, 3, []float64{1, 2, 3, 4, 5, 6})
//	a.T() // Result: 1 4 2 5 3 6 (reshaped to 3x2)
func (a *Array[T]) T() *Array[T] {
	data := slices.Clone(a.data)
	r, c := a.r, a.c
	a.Reshape(a.c, a.r)
	for ir := range c {
		for ic := range r {
			a.Set(ir, ic, data[ic*c+ir])
		}
	}
	return a
}

// T returns the transpose of the matrix.
// Example:
//
//	a := NewArray(2, 3, []float64{1, 2, 3, 4, 5, 6})
//	T(a) // Result: 1 4 2 5 3 6 (new 3x2 array)
func T[T Float](a *Array[T]) (o *Array[T]) {
	o = ZeroArray[T](a.c, a.r)
	for ir := range a.c {
		for ic := range a.r {
			o.Set(ir, ic, a.At(ic, ir))
		}
	}
	return
}

// Swap exchanges two rows or columns.
// axis:
//   - 0: swap rows
//   - 1: swap columns
//
// Example:
//
//	a := NewArray(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
//	a.Swap(0, 1, 0) // Swaps first and second rows
func (a *Array[T]) Swap(f, t int, axis int) *Array[T] {
	checkAxis(axis)
	if axis == 0 {
		if f < 0 || f >= a.r || t < 0 || t >= a.r {
			panic(fmt.Sprintf("numgo: index (%d,%d) is out of bounds for axis 0 with size %d", f, t, a.r))
		}
		for ic := range a.c {
			a.data[f*a.c+ic], a.data[t*a.c+ic] = a.data[t*a.c+ic], a.data[f*a.c+ic]
		}
	} else {
		if f < 0 || f >= a.c || t < 0 || t >= a.c {
			panic(fmt.Sprintf("numgo: index (%d,%d) is out of bounds for axis 1 with size %d", f, t, a.c))
		}
		for ir := range a.r {
			a.data[f+a.c*ir], a.data[t+a.c*ir] = a.data[t+a.c*ir], a.data[f+a.c*ir]
		}
	}
	return a
}

// Swap exchanges two rows or columns.
// axis:
//   - 0: swap rows
//   - 1: swap columns
//
// Example:
//
//	a := NewArray(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
//	Swap(a, 0, 1, 0) // Returns new array with first and second rows swapped
func Swap[T Float](a *Array[T], f, t int, axis int) (o *Array[T]) {
	o = Copy(a)
	return o.Swap(f, t, axis)
}

// Cat concatenates two arrays along the specified axis.
// axis:
//   - 0: concatenate along rows
//   - 1: concatenate along columns
//
// Example:
//
//	a := NewArray(2, 3, []float64{1, 2, 3, 4, 5, 6})
//	b := NewArray(2, 3, []float64{7, 8, 9, 10, 11, 12})
//	a.Cat(b, 0) // Concatenates along rows, resulting in a 4x3 array
func (a *Array[T]) Cat(b *Array[T], axis int) *Array[T] {
	checkAxis(axis)
	if axis == 0 {
		if a.c != b.c {
			panic(fmt.Sprintf("numgo: cannot concatenate arrays with shapes (%d,%d) and (%d,%d) along axis 0", a.r, a.c, b.r, b.c))
		}
		a.r += b.r
		a.data = append(a.data, b.data...)
	} else {
		if a.r != b.r {
			panic(fmt.Sprintf("numgo: cannot concatenate arrays with shapes (%d,%d) and (%d,%d) along axis 1", a.r, a.c, b.r, b.c))
		}
		buf := []T{}
		for i := range a.r {
			for ja := range a.c {
				buf = append(buf, a.data[i*a.c+ja])
			}
			for jb := range b.c {
				buf = append(buf, b.data[i*b.c+jb])
			}
		}
		a.c += b.c
		a.data = buf
	}
	return a
}

// Cat concatenates two arrays along the specified axis.
// axis:
//   - 0: concatenate along rows
//   - 1: concatenate along columns
//
// Example:
//
//	a := NewArray(2, 3, []float64{1, 2, 3, 4, 5, 6})
//	b := NewArray(2, 3, []float64{7, 8, 9, 10, 11, 12})
//	Cat(a, b, 0) // Returns new 4x3 array concatenated along rows
func Cat[T Float](a *Array[T], b *Array[T], axis int) *Array[T] {
	checkAxis(axis)
	if axis == 0 {
		if a.c != b.c {
			panic(fmt.Sprintf("numgo: cannot concatenate arrays with shapes (%d,%d) and (%d,%d) along axis 0", a.r, a.c, b.r, b.c))
		}
		return NewArray(a.r+b.r, a.c, slices.Concat(a.data, b.data))
	}
	if a.r != b.r {
		panic(fmt.Sprintf("numgo: cannot concatenate arrays with shapes (%d,%d) and (%d,%d) along axis 1", a.r, a.c, b.r, b.c))
	}
	buf := make([]T, 0, a.r*(a.c+b.c))
	for i := range a.r {
		buf = append(buf, a.data[i*a.c:(i+1)*a.c]...)
		buf = append(buf, b.data[i*b.c:(i+1)*b.c]...)
	}
	return NewArray(a.r, a.c+b.c, buf)
}

// Delete removes a specified row or column.
// n: index of the row or column to remove
// axis:
//   - 0: remove row
//   - 1: remove column
//
// Example:
//
//	a := NewArray(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
//	a.Delete(0, 0) // Removes the first row
func (a *Array[T]) Delete(n int, axis int) *Array[T] {
	checkAxis(axis)
	if axis == 0 {
		if n < 0 || n >= a.r {
			panic(fmt.Sprintf("numgo: index (%d) is out of bounds for axis 0 with size %d", n, a.r))
		}
		a.data = append(a.data[:n*a.c], a.data[(n+1)*a.c:]...)
		a.r--
	} else {
		if n < 0 || n >= a.c {
			panic(fmt.Sprintf("numgo: index (%d) is out of bounds for axis 1 with size %d", n, a.c))
		}
		buf := []T{}
		for ir := range a.r {
			for ic := range a.c {
				if ic == n {
					continue
				}
				buf = append(buf, a.data[ir*a.c+ic])
			}
		}
		a.data = buf
		a.c--
	}
	return a
}

// Delete removes a specified row or column.
// n: index of the row or column to remove
// axis:
//   - 0: remove row
//   - 1: remove column
//
// Example:
//
//	a := NewArray(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
//	Delete(a, 0, 0) // Returns new array with the first row removed
func Delete[T Float](a *Array[T], n int, axis int) *Array[T] {
	checkAxis(axis)
	buf := []T{}
	r, c := a.r, a.c
	if axis == 0 {
		if n < 0 || n >= a.r {
			panic(fmt.Sprintf("numgo: index (%d) is out of bounds for axis 0 with size %d", n, a.r))
		}
		buf = slices.Concat(a.data[:n*a.c], a.data[(n+1)*a.c:])
		r = a.r - 1
	} else {
		if n < 0 || n >= a.c {
			panic(fmt.Sprintf("numgo: index (%d) is out of bounds for axis 1 with size %d", n, a.c))
		}
		for ir := range a.r {
			for ic := range a.c {
				if ic == n {
					continue
				}
				buf = append(buf, a.data[ir*a.c+ic])
			}
		}
		c = a.c - 1
	}
	return NewArray(r, c, buf)
}

// RangeDelete removes a range of rows or columns.
// f: start index of the row or column to remove
// t: end index of the row or column to remove
// axis:
//   - 0: remove rows
//   - 1: remove columns
//
// Example:
//
//	a := NewArray(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
//	a.RangeDelete(0, 1, 0) // Removes the first row
func (a *Array[T]) RangeDelete(f, t int, axis int) *Array[T] {
	checkAxis(axis)
	if axis == 0 {
		if f < 0 || t > a.r || f > t {
			panic(fmt.Sprintf("numgo: range (%d,%d) is out of bounds for axis 0 with size %d", f, t, a.r))
		}
		a.data = append(a.data[:f*a.c], a.data[t*a.c:]...)
		a.r -= t - f
	} else {
		if f < 0 || t > a.c || f > t {
			panic(fmt.Sprintf("numgo: range (%d,%d) is out of bounds for axis 1 with size %d", f, t, a.c))
		}
		buf := []T{}
		for ir := range a.r {
			for ic := range a.c {
				if ic >= f && ic < t {
					continue
				}
				buf = append(buf, a.data[ir*a.c+ic])
			}
		}
		a.data = buf
		a.c -= t - f
	}
	return a
}

// RangeDelete removes a range of rows or columns.
// f: start index of the row or column to remove
// t: end index of the row or column to remove
// axis:
//   - 0: remove rows
//   - 1: remove columns
//
// Example:
//
//	a := NewArray(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
//	RangeDelete(a, 0, 1, 0) // Returns new array with the first row removed
func RangeDelete[T Float](a *Array[T], f, t int, axis int) *Array[T] {
	checkAxis(axis)
	buf := []T{}
	r, c := a.r, a.c
	if axis == 0 {
		if f < 0 || t > a.r || f > t {
			panic(fmt.Sprintf("numgo: range (%d,%d) is out of bounds for axis 0 with size %d", f, t, a.r))
		}
		buf = slices.Concat(a.data[:f*a.c], a.data[t*a.c:])
		r = a.r - (t - f)
	} else {
		if f < 0 || t > a.c || f > t {
			panic(fmt.Sprintf("numgo: range (%d,%d) is out of bounds for axis 1 with size %d", f, t, a.c))
		}
		for k, v := range a.All() {
			if k[1] < f || k[1] >= t {
				buf = append(buf, v)
			}
		}
		c = a.c - (t - f)
	}
	return NewArray(r, c, buf)
}

// Reshape changes the dimensions of the array.
//
// Example:
//
//	a := NewArray(2, 3, []float64{1, 2, 3, 4, 5, 6})
//	a.Reshape(3, 2) // Reshapes to 3x2
func (a *Array[T]) Reshape(r, c int) *Array[T] {
	if r == -1 && c > 0 {
		r = max(len(a.data)/c, 1)
	} else if c == -1 && r > 0 {
		c = max(len(a.data)/r, 1)
	}
	if r*c > len(a.data) {
		buf := make([]T, r*c-len(a.data))
		a.data = append(a.data, buf...)
	} else if r*c < len(a.data) {
		a.data = a.data[:r*c]
	}
	a.r, a.c = r, c
	return a
}

// Reshape changes the dimensions of the array.
//
// Example:
//
//	a := NewArray(2, 3, []float64{1, 2, 3, 4, 5, 6})
//	Reshape(a, 3, 2) // Returns new 3x2 array
func Reshape[T Float](a *Array[T], r, c int) *Array[T] {
	o := Copy(a)
	return o.Reshape(r, c)
}

// Equal checks if two Array instances are equal by comparing their dimensions and data.
// It returns true if both the row and column dimensions are the same and all corresponding elements are equal.
//
// Example:
//
//	a := NewArray(2, 2, []float64{1, 2, 3, 4})
//	b := NewArray(2, 2, []float64{1, 2, 3, 4})
//	fmt.Println(a.Equal(b)) // Output: true
func (a *Array[T]) Equal(b *Array[T]) bool {
	if a.r != b.r || a.c != b.c {
		return false
	}
	if len(a.data) != len(b.data) {
		return false
	}
	for i := range a.data {
		if a.data[i] != b.data[i] {
			return false
		}
	}
	return true
}

// Equal is a standalone function that checks if two Array instances are equal by calling the Equal method of the first Array.
// Example:
//
//	a := NewArray(2, 2, []float64{1, 2, 3, 4})
//	b := NewArray(2, 2, []float64{1, 2, 3, 4})
//	fmt.Println(Equal(a, b)) // Output: true
func Equal[T Float](a *Array[T], b *Array[T]) bool {
	return a.Equal(b)
}

// Filter applies a filtering function to each element of the Array, returning a new Array
// containing only the elements for which the function returns true.
//
// Example:
//
//	a := New1DArray([]float64{1.0, 2.0, 3.0, 4.0})
//	filteredArray := a.Filter(func(v float64) bool { return v > 2.0 })
//	fmt.Println(filteredArray.data) // Output: [3 4]
func (a *Array[T]) Filter(f func(r, c int, v T) bool) (ret *Array[T]) {
	data := []T{}
	for k, v := range a.All() {
		if f(k[0], k[1], v) {
			data = append(data, v)
		}
	}
	ret = New1DArray(data)
	return
}

// Filter applies a filtering function to each element of the Array, returning a new Array
// containing only the elements for which the function returns true.
//
// Example:
//
//	a := New1DArray([]float64{1.0, 2.0, 3.0, 4.0})
//	filteredArray := Filter(a, func(v float64) bool { return v > 2.0 })
//	fmt.Println(filteredArray.data) // Output: [3 4]
func Filter[T Float](a *Array[T], f func(r, c int, v T) bool) *Array[T] {
	return a.Filter(f)
}
