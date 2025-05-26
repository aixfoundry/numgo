package numgo

import (
	"math"
)

// MaxAbs returns the element with the maximum absolute value in the array and its position.
// It returns the value, row index, and column index of the element.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := a.MaxAbs()
//	fmt.Println(sol, r, c) // Output: 4 1 1
func (a *Array) MaxAbs() (sol float64, r, c int) {
	sol, r, c = math.SmallestNonzeroFloat64, -1, -1
	if len(a.data) == 0 {
		return
	}
	for ir := range a.r {
		for ic := range a.c {
			if math.Abs(sol) < math.Abs(a.data[ir*a.c+ic]) {
				sol = a.data[ir*a.c+ic]
				r, c = ir, ic
			}
		}
	}
	return
}

// MaxAbs is a wrapper function that calls the MaxAbs method of the Array struct.
// It returns the element with the maximum absolute value in the array and its position.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := MaxAbs(a)
//	fmt.Println(sol, r, c) // Output: 4 1 1
func MaxAbs(a *Array) (sol float64, r, c int) {
	return a.MaxAbs()
}

// MinAbs returns the element with the minimum absolute value in the array and its position.
// It returns the value, row index, and column index of the element.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := a.MinAbs()
//	fmt.Println(sol, r, c) // Output: -1 0 0
func (a *Array) MinAbs() (sol float64, r, c int) {
	sol, r, c = math.MaxFloat64, -1, -1
	if len(a.data) == 0 {
		return
	}
	for ir := range a.r {
		for ic := range a.c {
			v := a.data[ir*a.c+ic]
			if math.Abs(sol) > math.Abs(v) {
				sol = v
				r, c = ir, ic
			}
		}
	}
	return
}

// MinAbs is a wrapper function that calls the MinAbs method of the Array struct.
// It returns the element with the minimum absolute value in the array and its position.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := MinAbs(a)
//	fmt.Println(sol, r, c) // Output: -1 0 0
func MinAbs(a *Array) (sol float64, r, c int) {
	return a.MinAbs()
}

// Max returns the maximum element in the array and its position.
// It returns the value, row index, and column index of the element.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := a.Max()
//	fmt.Println(sol, r, c) // Output: 4 1 1
func (a *Array) Max() (sol float64, r, c int) {
	sol, r, c = math.SmallestNonzeroFloat64, -1, -1
	if len(a.data) == 0 {
		return
	}
	for ir := range a.r {
		for ic := range a.c {
			if sol < a.data[ir*a.c+ic] {
				sol = a.data[ir*a.c+ic]
				r, c = ir, ic
			}
		}
	}
	return
}

// Max is a wrapper function that calls the Max method of the Array struct.
// It returns the maximum element in the array and its position.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := Max(a)
//	fmt.Println(sol, r, c) // Output: 4 1 1
func Max(a *Array) (sol float64, r, c int) {
	return a.Max()
}

// Min returns the minimum element in the array and its position.
// It returns the value, row index, and column index of the element.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := a.Min()
//	fmt.Println(sol, r, c) // Output: -3 1 0
func (a *Array) Min() (sol float64, r, c int) {
	sol, r, c = math.MaxFloat64, -1, -1
	if len(a.data) == 0 {
		return
	}
	for ir := range a.r {
		for ic := range a.c {
			if sol > a.data[ir*a.c+ic] {
				sol = a.data[ir*a.c+ic]
				r, c = ir, ic
			}
		}
	}
	return
}

// Min is a wrapper function that calls the Min method of the Array struct.
// It returns the minimum element in the array and its position.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := Min(a)
//	fmt.Println(sol, r, c) // Output: -3 1 0
func Min(a *Array) (sol float64, r, c int) {
	return a.Min()
}

// Dot computes the dot product of two arrays.
// It returns a new Array representing the result of the dot product.
// If the number of columns in the first array is not equal to the number of rows in the second array, it returns nil.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{2, 0}, {1, 2}})
//	result := a.Dot(b)
//	fmt.Println(result) // Output: [[4 4], [10 8]]
func (a *Array) Dot(b *Array) (ret *Array) {
	if a.c != b.r {
		return
	}
	ret = ZeroArray(a.r, b.c)
	for i := range a.r {
		for j := range b.c {
			for k := range a.c {
				ret.data[b.c*i+j] += a.At(i, k) * b.At(k, j)
			}
		}
	}
	return
}

// Dot computes the dot product of two arrays.
// It returns a new Array representing the result of the dot product.
// If the number of columns in the first array is not equal to the number of rows in the second array, it returns nil.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{2, 0}, {1, 2}})
//	result := Dot(a, b)
//	fmt.Println(result) // Output: [[4 4], [10 8]]
func Dot(a *Array, b *Array) (ret *Array) {
	if a.c != b.r {
		return
	}
	ret = ZeroArray(a.r, b.c)
	for i := range a.r {
		for j := range b.c {
			for k := range a.c {
				ret.data[b.c*i+j] += a.At(i, k) * b.At(k, j)
			}
		}
	}
	return
}

// Mean calculates the mean of the elements along the specified axis of the array.
// If axis is 0, it calculates the mean for each column and returns a 1xN array.
// If axis is 1, it calculates the mean for each row and returns an Nx1 array.
// If the array is empty or has zero dimensions, it returns nil.
//
// Example usage:
//
//	a := NewArray([][]float64{{1, 2, 3}, {4, 5, 6}})
//	mean := a.Mean(0) // mean will be a *Array with data [[2.5, 3.5, 4.5]]
//	mean := a.Mean(1) // mean will be a *Array with data [[2], [5]]
func (a *Array) Mean(axis int) (ret *Array) {
	if len(a.data) == 0 || a.r == 0 || a.c == 0 {
		return
	}
	if axis == 0 {
		ret = ZeroArray(1, a.c)
		for ic := range a.c {
			ret.Set(0, ic, a.Of(ic, 1).Avg())
		}
	} else if axis == 1 {
		ret = ZeroArray(1, a.r)
		for ir := range a.r {
			ret.Set(0, ir, a.Of(ir, 0).Avg())
		}
	}
	return
}

// Mean is a utility function that calls the Mean method on the Array receiver.
// It calculates the mean of the elements along the specified axis of the array.
//
// Example usage:
//
//	a := NewArray([][]float64{{1, 2, 3}, {4, 5, 6}})
//	mean := Mean(a, 0) // mean will be a *Array with data [[2.5, 3.5, 4.5]]
//	mean := Mean(a, 1) // mean will be a *Array with data [[2], [5]]
func Mean(a *Array, axis int) (ret *Array) {
	return a.Mean(axis)
}

// Avg computes the average of all elements in the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	avg := a.Avg()
//	fmt.Println(avg) // Output: 2.5
func (a *Array) Avg() (ret float64) {
	return a.Sum() / float64(len(a.data))
}

// Avg is a wrapper function that calls the Avg method of the Array struct.
// It computes the average of all elements in the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	avg := Avg(a)
//	fmt.Println(avg) // Output: 2.5
func Avg(a *Array) (ret float64) {
	return a.Avg()
}

// Sum computes the sum of all elements in the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	sum := a.Sum()
//	fmt.Println(sum) // Output: 10
func (a *Array) Sum() (ret float64) {
	ret = 0
	for _, v := range a.data {
		ret += v
	}
	return
}

// Sum is a wrapper function that calls the Sum method of the Array struct.
// It computes the sum of all elements in the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	sum := Sum(a)
//	fmt.Println(sum) // Output: 10
func Sum(a *Array) (ret float64) {
	return a.Sum()
}

// MovingAvg computes the moving average of the array along the specified axis.
// It returns a new Array with the moving averages.
// If the window or step is less than or equal to 0, or the window is greater than the length of the array, it returns an array with the overall average.
//
// Example:
//
//	a := New1DArray([]float64{1, 2, 3, 4, 5})
//	result := a.MovingAvg(3, 1, 0)
//	fmt.Println(result) // Output: [2 3 4]
func (a *Array) MovingAvg(window int, step int, axis int) (ret *Array) {
	data := []float64{}
	if window <= 0 || step <= 0 || window > len(a.data) {
		data = append(data, a.Avg())
		ret = New1DArray(data)
		return
	}
	if axis == 0 {
		i := 0
		l := len(a.data)
		for {
			if i+window >= l {
				if i < l {
					sum := 0.0
					for _, v := range a.data[i:] {
						sum += v
					}
					data = append(data, sum/float64(l-i))
				}
				break
			}
			sum := 0.0
			for _, v := range a.data[i : i+window] {
				sum += v
			}
			data = append(data, sum/float64(window))
			i += step
		}
		ret = New1DArray(data)
	} else if axis == 1 {
		ca := CopyOf(a).T()
		ret = ca.MovingAvg(window, step, 0)
	}
	return
}

// MovingAvg is a wrapper function that calls the MovingAvg method of the Array struct.
// It computes the moving average of the array along the specified axis.
// It returns a new Array with the moving averages.
//
// Example:
//
//	a := New1DArray([]float64{1, 2, 3, 4, 5})
//	result := MovingAvg(a, 3, 1, 0)
//	fmt.Println(result) // Output: [2 3 4]
func MovingAvg(a *Array, window int, step int, axis int) (ret *Array) {
	return a.MovingAvg(window, step, axis)
}

// Sub subtracts two arrays element-wise. If the dimensions of the arrays do not match,
// it checks for broadcasting conditions where one array is a single row or column.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	b := NewArray([][]float64{{1, 1}})
//	result := a.Sub(b) // result is [[0, 1], [2, 3]]
func (a *Array) Sub(b *Array) (ret *Array) {
	if a.r == b.r && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)-b.At(ir, ic))
			}
		}
	} else if a.r != b.r && b.r == 1 && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)-b.At(0, ic))
			}
		}
	} else if a.c != b.c && b.c == 1 && a.r == b.r {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)-b.At(ir, 0))
			}
		}
	} else if b.c == 1 && b.r == 1 {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)-b.At(0, 0))
			}
		}
	}
	return
}

// Sub subtracts two arrays element-wise. It is a wrapper around the Array.Sub method.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	b := NewArray([][]float64{{1, 1}})
//	result := Sub(a, b) // result is [[0, 1], [2, 3]]
func Sub(a *Array, b *Array) (ret *Array) {
	return a.Sub(b)
}

// Add adds two arrays element-wise. If the dimensions of the arrays do not match,
// it checks for broadcasting conditions where one array is a single row or column.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	b := NewArray([][]float64{{1, 1}})
//	result := a.Add(b) // result is [[2, 3], [4, 5]]
func (a *Array) Add(b *Array) (ret *Array) {
	if a.r == b.r && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)+b.At(ir, ic))
			}
		}
	} else if a.r != b.r && b.r == 1 && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)+b.At(0, ic))
			}
		}
	} else if a.c != b.c && b.c == 1 && a.r == b.r {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)+b.At(ir, 0))
			}
		}
	} else if b.c == 1 && b.r == 1 {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)+b.At(0, 0))
			}
		}
	}
	return
}

// Add adds two arrays element-wise. It is a wrapper around the Array.Add method.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	b := NewArray([][]float64{{1, 1}})
//	result := Add(a, b) // result is [[2, 3], [4, 5]]
func Add(a *Array, b *Array) (ret *Array) {
	return a.Add(b)
}

// Mul multiplies two arrays element-wise. If the dimensions of the arrays do not match,
// it checks for broadcasting conditions where one array is a single row or column.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	b := NewArray([][]float64{{2, 2}})
//	result := a.Mul(b) // result is [[2, 4], [6, 8]]
func (a *Array) Mul(b *Array) (ret *Array) {
	if a.r == b.r && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)*b.At(ir, ic))
			}
		}
	} else if a.r != b.r && b.r == 1 && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)*b.At(0, ic))
			}
		}
	} else if a.c != b.c && b.c == 1 && a.r == b.r {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)*b.At(ir, 0))
			}
		}
	} else if b.c == 1 && b.r == 1 {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				ret.Set(ir, ic, a.At(ir, ic)*b.At(0, 0))
			}
		}
	}
	return
}

// Mul multiplies two arrays element-wise. It is a wrapper around the Array.Mul method.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	b := NewArray([][]float64{{2, 2}})
//	result := Mul(a, b) // result is [[2, 4], [6, 8]]
func Mul(a *Array, b *Array) (ret *Array) {
	return a.Mul(b)
}

// Div divides two arrays element-wise. If the dimensions of the arrays do not match,
// it checks for broadcasting conditions where one array is a single row or column.
// Division by zero results in NaN for that element.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	b := NewArray([][]float64{{1, 1}})
//	result := a.Div(b) // result is [[1, 2], [3, 4]]
func (a *Array) Div(b *Array) (ret *Array) {
	if a.r == b.r && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				if b.At(ir, ic) == 0 {
					ret.Set(ir, ic, math.NaN())
				} else {
					ret.Set(ir, ic, a.At(ir, ic)/b.At(ir, ic))
				}
			}
		}
	} else if a.r != b.r && b.r == 1 && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				if b.At(0, ic) == 0 {
					ret.Set(ir, ic, math.NaN())
				} else {
					ret.Set(ir, ic, a.At(ir, ic)/b.At(0, ic))
				}
			}
		}
	} else if a.c != b.c && b.c == 1 && a.r == b.r {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				if b.At(ir, 0) == 0 {
					ret.Set(ir, ic, math.NaN())
				} else {
					ret.Set(ir, ic, a.At(ir, ic)/b.At(ir, 0))
				}
			}
		}
	} else if b.c == 1 && b.r == 1 {
		ret = ZeroArray(a.r, a.c)
		for ir := range a.r {
			for ic := range a.c {
				if b.At(0, 0) == 0 {
					ret.Set(ir, ic, math.NaN())
				} else {
					ret.Set(ir, ic, a.At(ir, ic)/b.At(0, 0))
				}
			}
		}
	}
	return
}

// Div divides two arrays element-wise. It is a wrapper around the Array.Div method.
// Division by zero results in NaN for that element.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	b := NewArray([][]float64{{1, 1}})
//	result := Div(a, b) // result is [[1, 2], [3, 4]]
func Div(a *Array, b *Array) (ret *Array) {
	return a.Div(b)
}

// DivValue divides each element of the array by a given value.
// If the value is 0, it sets the result to NaN to avoid division by zero.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	result := a.DivValue(2)
//	// result will be a new Array with elements [[0.5, 1], [1.5, 2]]
func (a *Array) DivValue(value float64) (ret *Array) {
	ret = ZeroArray(a.r, a.c)
	for ir := range a.r {
		for ic := range a.c {
			if value == 0 {
				ret.Set(ir, ic, math.NaN())
			} else {
				ret.Set(ir, ic, a.At(ir, ic)/value)
			}
		}
	}
	return
}

// DivValue is a wrapper function that calls the DivValue method of the Array struct.
// Example usage is the same as DivValue method.
func DivValue(a *Array, value float64) (ret *Array) {
	return a.DivValue(value)
}

// MulValue multiplies each element of the array by a given value.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	result := a.MulValue(2)
//	// result will be a new Array with elements [[2, 4], [6, 8]]
func (a *Array) MulValue(value float64) (ret *Array) {
	ret = ZeroArray(a.r, a.c)
	for ir := range a.r {
		for ic := range a.c {
			ret.Set(ir, ic, a.At(ir, ic)*value)
		}
	}
	return
}

// MulValue is a wrapper function that calls the MulValue method of the Array struct.
// Example usage is the same as MulValue method.
func MulValue(a *Array, value float64) (ret *Array) {
	return a.MulValue(value)
}

// SubValue subtracts a given value from each element of the array.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	result := a.SubValue(1)
//	// result will be a new Array with elements [[0, 1], [2, 3]]
func (a *Array) SubValue(value float64) (ret *Array) {
	ret = ZeroArray(a.r, a.c)
	for ir := range a.r {
		for ic := range a.c {
			ret.Set(ir, ic, a.At(ir, ic)-value)
		}
	}
	return
}

// SubValue is a wrapper function that calls the SubValue method of the Array struct.
// Example usage is the same as SubValue method.
func SubValue(a *Array, value float64) (ret *Array) {
	return a.SubValue(value)
}

// AddValue adds a given value to each element of the array.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2}, {3, 4}})
//	result := a.AddValue(1)
//	// result will be a new Array with elements [[2, 3], [4, 5]]
func (a *Array) AddValue(value float64) (ret *Array) {
	ret = ZeroArray(a.r, a.c)
	for ir := range a.r {
		for ic := range a.c {
			ret.Set(ir, ic, a.At(ir, ic)+value)
		}
	}
	return
}

// AddValue is a wrapper function that calls the AddValue method of the Array struct.
// Example usage is the same as AddValue method.
func AddValue(a *Array, value float64) (ret *Array) {
	return a.AddValue(value)
}
