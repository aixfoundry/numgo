package numgo

// T returns the transpose of the matrix.
// Example:
//
//	a := NewArray(2, 3, []float64{1, 2, 3, 4, 5, 6})
//	a.T() // Result: 1 4 2 5 3 6 (reshaped to 3x2)
func (a *Array) T() *Array {
	data := make([]float64, len(a.data))
	r, c := a.r, a.c
	copy(data, a.data)
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
func T(a *Array) (o *Array) {
	o = ZeroArray(a.c, a.r)
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
func (a *Array) Swap(f, t int, axis int) *Array {
	if axis == 0 {
		for ic := range a.c {
			a.data[f*a.c+ic], a.data[t*a.c+ic] = a.data[t*a.c+ic], a.data[f*a.c+ic]
		}
	} else if axis == 1 {
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
func Swap(a *Array, f, t int, axis int) (o *Array) {
	o = Copy(a)
	if axis == 0 {
		for ic := range o.c {
			o.data[f*o.r+ic], o.data[t*o.r+ic] = o.data[t*o.r+ic], o.data[f*o.r+ic]
		}
	} else if axis == 1 {
		for ir := range o.r {
			o.data[f+o.r*ir], o.data[t+o.r*ir] = o.data[t+o.r*ir], o.data[f+o.r*ir]
		}
	}
	return
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
func (a *Array) Cat(b *Array, axis int) *Array {
	if axis == 0 {
		a.r += b.r
		a.data = append(a.data, b.data...)
	} else {
		buf := []float64{}
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
func Cat(a *Array, b *Array, axis int) (o *Array) {
	if axis == 0 {
		o = EmptyArray(a.r+b.r, a.c)
		for i := range a.r {
			for ja := range a.c {
				o.data = append(o.data, a.data[i*a.c+ja])
			}
		}
		for i := range b.r {
			for jb := range b.c {
				o.data = append(o.data, b.data[i*b.c+jb])
			}
		}
	} else {
		o = EmptyArray(a.r, a.c+b.c)
		for i := range a.r {
			for ja := range a.c {
				o.data = append(o.data, a.data[i*a.c+ja])
			}
			for jb := range b.c {
				o.data = append(o.data, b.data[i*b.c+jb])
			}
		}
	}
	return
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
func (a *Array) Delete(n int, axis int) *Array {
	if axis == 0 {
		a.data = append(a.data[:a.r*n], a.data[a.r*n+a.c:]...)
		a.r--
	} else {
		buf := []float64{}
		for ir := range a.r {
			for ic := range a.c {
				if ic == n {
					continue
				}
				buf = append(buf, a.data[ir*a.r+ic])
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
func Delete(a *Array, n int, axis int) *Array {
	buf := []float64{}
	r, c := a.r, a.c
	if axis == 0 {
		buf = append(a.data[:a.r*n], a.data[a.r*n+a.c:]...)
		r = a.r - 1
	} else {
		for ir := range a.r {
			for ic := range a.c {
				if ic == n {
					continue
				}
				buf = append(buf, a.data[ir*a.r+ic])
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
func (a *Array) RangeDelete(f, t int, axis int) *Array {
	if axis == 0 {
		a.data = append(a.data[:f*a.r], a.data[t*a.r:]...)
		a.r -= t - f
	} else {
		buf := []float64{}
		for ir := range a.r {
			for ic := range a.c {
				if ic >= f && ic < t {
					continue
				}
				buf = append(buf, a.data[ir*a.r+ic])
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
func RangeDelete(a *Array, f, t int, axis int) *Array {
	buf := []float64{}
	r, c := a.r, a.c
	if axis == 0 {
		buf = append(a.data[:f*a.c], a.data[t*a.c:]...)
		r = a.r - (t - f)
	} else {
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
func (a *Array) Reshape(r, c int) *Array {
	if r == -1 {
		r = max(len(a.data)/c, 1)
	} else if c == -1 {
		c = max(len(a.data)/r, 1)
	}
	if r*c > len(a.data) {
		buf := make([]float64, r*c-len(a.data))
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
func Reshape(a *Array, r, c int) *Array {
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
func (a *Array) Equal(b *Array) bool {
	if a.r != b.r || a.c != b.c {
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
func Equal(a *Array, b *Array) bool {
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
func (a *Array) Filter(f func(r, c int, v float64) bool) (ret *Array) {
	data := []float64{}
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
func Filter(a *Array, f func(r, c int, v float64) bool) *Array {
	return a.Filter(f)
}
