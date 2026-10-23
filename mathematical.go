package numgo

import (
	"fmt"
	"math"
	"slices"
)

// Abs returns a new Array where each element is the absolute value of the corresponding element in the input Array.
//
// Example:
//
//	a := NewArray(2, 2, []float64{-1.0, 2.0, -3.0, 4.0})
//	absA := a.Abs()
//	fmt.Println(absA) // Output: [1 2 3 4]
func (a *Array[T]) Abs() (ret *Array[T]) {
	ret = ZeroArray[T](a.r, a.c)
	for i, v := range a.data {
		ret.data[i] = T(math.Abs(toF64(v)))
	}
	return ret
}

// Abs returns a new Array where each element is the absolute value of the corresponding element in the input Array.
// This is a convenience function that calls the Abs method on the Array.
//
// Example:
//
//	a := NewArray(2, 2, []float64{-1.0, 2.0, -3.0, 4.0})
//	absA := Abs(a)
//	fmt.Println(absA) // Output: [1 2 3 4]
func Abs[T Float](a *Array[T]) *Array[T] {
	return a.Abs()
}

// MaxAbs returns the element with the maximum absolute value in the array and its position.
// It returns the value, row index, and column index of the element.
// For an empty array it returns (0, -1, -1); an all-zero array has no valid position.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := a.MaxAbs()
//	fmt.Println(sol, r, c) // Output: 4 1 1
func (a *Array[T]) MaxAbs() (sol T, r, c int) {
	r, c = -1, -1
	if len(a.data) == 0 {
		return
	}
	for k, v := range a.All() {
		vv := T(math.Abs(toF64(v)))
		if T(math.Abs(toF64(sol))) < vv {
			sol = vv
			r, c = k[0], k[1]
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
func MaxAbs[T Float](a *Array[T]) (sol T, r, c int) {
	return a.MaxAbs()
}

// MinAbs returns the element with the minimum absolute value in the array and its position.
// It returns the value, row index, and column index of the element.
// For an empty array it returns (0, -1, -1).
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := a.MinAbs()
//	fmt.Println(sol, r, c) // Output: 1 0 0
func (a *Array[T]) MinAbs() (sol T, r, c int) {
	sol, r, c = T(math.Inf(1)), -1, -1
	if len(a.data) == 0 {
		sol = 0
		return
	}
	for k, v := range a.All() {
		vv := T(math.Abs(toF64(v)))
		if T(math.Abs(toF64(sol))) > vv {
			sol = vv
			r, c = k[0], k[1]
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
//	fmt.Println(sol, r, c) // Output: 1 0 0
func MinAbs[T Float](a *Array[T]) (sol T, r, c int) {
	return a.MinAbs()
}

// Max returns the maximum element in the array and its position.
// It returns the value, row index, and column index of the element.
// An empty array reduces to a 1x1 array containing NaN.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol := a.Max(-1)
//	fmt.Println(sol) // Output: [4]
func (a *Array[T]) Max(axis int) (ret *Array[T]) {
	if len(a.data) == 0 {
		return New1DArray([]T{T(math.NaN())})
	}
	if axis == 0 {
		ret = ZeroArray[T](1, a.c)
		for j := range a.c {
			maxVal := T(math.Inf(-1))
			for i := range a.r {
				if a.At(i, j) > maxVal {
					maxVal = a.At(i, j)
				}
			}
			ret.Set(0, j, maxVal)
		}
	} else if axis == 1 {
		ret = ZeroArray[T](a.r, 1)
		for i := range a.r {
			maxVal := T(math.Inf(-1))
			for j := range a.c {
				if a.At(i, j) > maxVal {
					maxVal = a.At(i, j)
				}
			}
			ret.Set(i, 0, maxVal)
		}
	} else {
		maxVal := T(math.Inf(-1))
		for _, v := range a.All() {
			if maxVal < v {
				maxVal = v
			}
		}
		ret = New1DArray([]T{maxVal})
	}
	return
}

// Max is a wrapper function that calls the Max method of the Array struct.
// It returns the maximum element in the array and its position.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol := Max(a, -1)
//	fmt.Println(sol) // Output: [4]
func Max[T Float](a *Array[T], axis int) *Array[T] {
	return a.Max(axis)
}

// Min returns the minimum element in the array and its position.
// It returns the value, row index, and column index of the element.
// An empty array reduces to a 1x1 array containing NaN.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := a.Min()
//	fmt.Println(sol, r, c) // Output: -3 1 0
func (a *Array[T]) Min(axis int) (ret *Array[T]) {
	if len(a.data) == 0 {
		return New1DArray([]T{T(math.NaN())})
	}
	if axis == 0 {
		ret = ZeroArray[T](1, a.c)
		for j := range a.c {
			minVal := T(math.Inf(1))
			for i := range a.r {
				if a.At(i, j) < minVal {
					minVal = a.At(i, j)
				}
			}
			ret.Set(0, j, minVal)
		}
	} else if axis == 1 {
		ret = ZeroArray[T](a.r, 1)
		for i := range a.r {
			minVal := T(math.Inf(1))
			for j := range a.c {
				if a.At(i, j) < minVal {
					minVal = a.At(i, j)
				}
			}
			ret.Set(i, 0, minVal)
		}
	} else {
		minVal := T(math.Inf(1))
		for _, v := range a.All() {
			if minVal > v {
				minVal = v
			}
		}
		ret = New1DArray([]T{minVal})
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
func Min[T Float](a *Array[T], axis int) *Array[T] {
	return a.Min(axis)
}

// Dot computes the dot product of two arrays.
// It returns a new Array representing the result of the dot product.
// It panics if the number of columns in the first array is not equal to the number of rows in the second array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{2, 0}, {1, 2}})
//	result := a.Dot(b)
//	fmt.Println(result) // Output: [[4 4], [10 8]]
func (a *Array[T]) Dot(b *Array[T]) (ret *Array[T]) {
	if a.c != b.r {
		panic(fmt.Sprintf("numgo: Dot: shape mismatch (%d,%d) x (%d,%d)", a.r, a.c, b.r, b.c))
	}
	ret = ZeroArray[T](a.r, b.c)
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
// It panics if the number of columns in the first array is not equal to the number of rows in the second array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{2, 0}, {1, 2}})
//	result := Dot(a, b)
//	fmt.Println(result) // Output: [[4 4], [10 8]]
func Dot[T Float](a, b *Array[T]) *Array[T] {
	return a.Dot(b)
}

// Median calculates the median of the Array along the specified axis.
// If axis is 0, it calculates the median for each column.
// If axis is 1, it calculates the median for each row.
// If axis is -1, it calculates the median of the entire array.
// It returns a new Array containing the median values.
// An empty array reduces to a 1x1 array containing NaN.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 3, 5}, {2, 4, 6}})
//	median := a.Median(0) // median will be a 1x3 Array with values [1.5, 3.5, 5.5]
func (a *Array[T]) Median(axis int) (ret *Array[T]) {
	if len(a.data) == 0 {
		return New1DArray([]T{T(math.NaN())})
	}
	if axis == 0 {
		ret = ZeroArray[T](1, a.c)
		for ic := range a.c {
			col := a.Col(ic).Sort(axis)
			if col.r%2 == 0 {
				ret.Set(0, ic, (col.At(col.r/2-1, 0)+col.At(col.r/2, 0))/2)
			} else {
				ret.Set(0, ic, col.At(col.r/2, 0))
			}
		}
	} else if axis == 1 {
		ret = ZeroArray[T](a.r, 1)
		for ir := range a.r {
			row := a.Row(ir).Sort(axis)
			if row.c%2 == 0 {
				ret.Set(ir, 0, (row.At(0, row.c/2-1)+row.At(0, row.c/2))/2)
			} else {
				ret.Set(ir, 0, row.At(0, row.c/2))
			}
		}
	} else {
		var median T
		data := slices.Clone(a.Data())
		slices.Sort(data)
		if len(data)%2 == 0 {
			median = (data[len(data)/2-1] + data[len(data)/2]) / 2
		} else {
			median = data[len(data)/2]
		}
		ret = New1DArray([]T{median})
	}
	return
}

// Median is a convenience function that calls the Median method on the Array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 3, 5}, {2, 4, 6}})
//	median := Median(a, 0) // median will be a 1x3 Array with values [1.5, 3.5, 5.5]
func Median[T Float](a *Array[T], axis int) *Array[T] {
	return a.Median(axis)
}

// Sort sorts the elements of the Array along the specified axis.
// If axis is 0, it sorts each column.
// If axis is 1, it sorts each row.
// If axis is -1, it sorts the entire array.
// It returns the sorted Array.
//
// Example:
//
//	a := New2DArray([][]float64{{3, 1, 2}, {6, 5, 4}})
//	sorted := a.Sort(0) // sorted will be a 2x3 Array with values [[3, 1, 2], [6, 5, 4]]
func (a *Array[T]) Sort(axis int) *Array[T] {
	if axis == 0 {
		for ic := range a.c {
			col := a.Col(ic).Data()
			slices.Sort(col)
			for ir := range a.r {
				a.Set(ir, ic, col[ir])
			}
		}
	} else if axis == 1 {
		for ir := range a.r {
			row := a.Row(ir).Data()
			slices.Sort(row)
			for ic := range a.c {
				a.Set(ir, ic, row[ic])
			}
		}
	} else {
		slices.Sort(a.data)
	}
	return a
}

// Sort is a convenience function that calls the Sort method on the Array.
//
// Example:
//
//	a := New2DArray([][]float64{{3, 1, 2}, {6, 5, 4}})
//	sorted := Sort(a, 0) // sorted will be a 2x3 Array with values [[3, 1, 2], [6, 5, 4]]
func Sort[T Float](a *Array[T], axis int) *Array[T] {
	return a.Sort(axis)
}

// Mean calculates the mean of the elements along the specified axis of the array.
// If axis is 0, it calculates the mean for each column and returns a 1xN array.
// If axis is 1, it calculates the mean for each row and returns an Nx1 array.
// If the array is empty, it returns a 1x1 array containing NaN.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}})
//	mean := a.Mean(0) // mean will be a 1x3 Array with values [2.5, 3.5, 4.5]
//	mean := a.Mean(1) // mean will be a 3x1 Array with values [[2], [5]]
func (a *Array[T]) Mean(axis int) (ret *Array[T]) {
	if len(a.data) == 0 || a.r == 0 || a.c == 0 {
		return New1DArray([]T{T(math.NaN())})
	}
	var divisor int
	if axis == 0 {
		divisor = a.r
	} else if axis == 1 {
		divisor = a.c
	} else {
		divisor = a.r * a.c
	}
	return a.Sum(axis).DivValue(T(divisor))
}

// Mean is a utility function that calls the Mean method on the Array receiver.
// It calculates the mean of the elements along the specified axis of the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}})
//	mean := Mean(a, 0) // mean will be a 1x3 Array with values [2.5, 3.5, 4.5]
//	mean := Mean(a, 1) // mean will be a 3x1 Array with values [[2], [5]]
func Mean[T Float](a *Array[T], axis int) *Array[T] {
	return a.Mean(axis)
}

// Avg computes the average of all elements in the array.
// An empty array returns NaN.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	avg := a.Avg()
//	fmt.Println(avg) // Output: 2.5
func (a *Array[T]) Avg() T {
	return a.Mean(-1).At(0, 0)
}

// Avg is a wrapper function that calls the Avg method on the Array struct.
// It computes the average of all elements in the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	avg := Avg(a)
//	fmt.Println(avg) // Output: 2.5
func Avg[T Float](a *Array[T]) T {
	return a.Avg()
}

// Sum computes the sum of all elements in the array.
// If axis is 0, it returns a 1xN array; if axis is 1, it returns an Nx1 array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	sum := a.Sum(-1)
//	fmt.Println(sum) // Output: 10
func (a *Array[T]) Sum(axis int) (ret *Array[T]) {
	if axis == 0 {
		ret = ZeroArray[T](1, a.c)
		for ic := range a.c {
			ret.Set(0, ic, a.Of(ic, axis).Sum(-1).At(0, 0))
		}
	} else if axis == 1 {
		ret = ZeroArray[T](a.r, 1)
		for ir := range a.r {
			ret.Set(ir, 0, a.Of(ir, axis).Sum(-1).At(0, 0))
		}
	} else {
		var sum T
		for _, v := range a.data {
			sum += v
		}
		ret = New1DArray([]T{sum})
	}
	return
}

// Sum is a wrapper function that calls the Sum method on the Array struct.
// It computes the sum of all elements in the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	sum := Sum(a, -1)
//	fmt.Println(sum) // Output: 10
func Sum[T Float](a *Array[T], axis int) *Array[T] {
	return a.Sum(axis)
}

// MovingAvg computes the moving average of the array along the specified axis.
// It returns a new Array with the moving averages.
// If the window or step is less than or equal to 0, or the window is greater than the length of the array, it returns an array with the overall average.
// It panics if axis is not 0 or 1.
//
// Example:
//
//	a := New1DArray([]float64{1, 2, 3, 4, 5})
//	result := a.MovingAvg(3, 1, 0)
//	fmt.Println(result) // Output: [2 3 4]
func (a *Array[T]) MovingAvg(window int, step int, axis int) (ret *Array[T]) {
	checkAxis(axis)
	if window <= 0 || step <= 0 || window > len(a.data) {
		return New1DArray([]T{a.Mean(-1).At(0, 0)})
	}
	if axis == 1 {
		return Copy(a).T().MovingAvg(window, step, 0)
	}
	data := []T{}
	i := 0
	l := len(a.data)
	for {
		if i+window >= l {
			if i < l {
				var sum T
				for _, v := range a.data[i:] {
					sum += v
				}
				data = append(data, sum/T(l-i))
			}
			break
		}
		var sum T
		for _, v := range a.data[i : i+window] {
			sum += v
		}
		data = append(data, sum/T(window))
		i += step
	}
	return New1DArray(data)
}

// MovingAvg is a wrapper function that calls the MovingAvg method on the Array struct.
// It computes the moving average of the array along the specified axis.
// It returns a new Array with the moving averages.
//
// Example:
//
//	a := New1DArray([]float64{1, 2, 3, 4, 5})
//	result := MovingAvg(a, 3, 1, 0)
//	fmt.Println(result) // Output: [2 3 4]
func MovingAvg[T Float](a *Array[T], window int, step int, axis int) *Array[T] {
	return a.MovingAvg(window, step, axis)
}

// Sub subtracts two arrays element-wise, broadcasting a single row, column or 1x1 array.
// It panics if the shapes cannot be broadcast together.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{1, 1}})
//	result := a.Sub(b) // result is [[0, 1], [2, 3]]
func (a *Array[T]) Sub(b *Array[T]) *Array[T] {
	return a.zip(b, func(x, y T) T { return x - y })
}

// Sub subtracts two arrays element-wise. It is a wrapper around the Array.Sub method.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{1, 1}})
//	result := Sub(a, b) // result is [[0, 1], [2, 3]]
func Sub[T Float](a, b *Array[T]) *Array[T] {
	return a.Sub(b)
}

// Add adds two arrays element-wise, broadcasting a single row, column or 1x1 array.
// It panics if the shapes cannot be broadcast together.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{1, 1}})
//	result := a.Add(b) // result is [[2, 3], [4, 5]]
func (a *Array[T]) Add(b *Array[T]) *Array[T] {
	return a.zip(b, func(x, y T) T { return x + y })
}

// Add adds two arrays element-wise. It is a wrapper around the Array.Add method.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{1, 1}})
//	result := Add(a, b) // result is [[2, 3], [4, 5]]
func Add[T Float](a, b *Array[T]) *Array[T] {
	return a.Add(b)
}

// Mul multiplies two arrays element-wise, broadcasting a single row, column or 1x1 array.
// It panics if the shapes cannot be broadcast together.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{2, 2}})
//	result := a.Mul(b) // result is [[2, 4], [6, 8]]
func (a *Array[T]) Mul(b *Array[T]) *Array[T] {
	return a.zip(b, func(x, y T) T { return x * y })
}

// Mul multiplies two arrays element-wise. It is a wrapper around the Array.Mul method.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{2, 2}})
//	result := Mul(a, b) // result is [[2, 4], [6, 8]]
func Mul[T Float](a, b *Array[T]) *Array[T] {
	return a.Mul(b)
}

// Div divides two arrays element-wise, broadcasting a single row, column or 1x1 array.
// Division by zero follows IEEE semantics (Inf/NaN). It panics if the shapes cannot be broadcast together.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{1, 1}})
//	result := a.Div(b) // result is [[1, 2], [3, 4]]
func (a *Array[T]) Div(b *Array[T]) *Array[T] {
	return a.zip(b, func(x, y T) T { return x / y })
}

// Div divides two arrays element-wise. It is a wrapper around the Array.Div method.
// Division by zero follows IEEE semantics (Inf/NaN).
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{1, 1}})
//	result := Div(a, b) // result is [[1, 2], [3, 4]]
func Div[T Float](a, b *Array[T]) *Array[T] {
	return a.Div(b)
}

// DivValue divides each element of the array by a given value.
// Division by zero follows IEEE semantics (Inf/NaN).
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	result := a.DivValue(2)
//	// result will be a new Array with elements [[0.5, 1], [1.5, 2]]
func (a *Array[T]) DivValue(value T) *Array[T] {
	return a.zipValue(value, func(x, y T) T { return x / y })
}

// DivValue is a wrapper function that calls the DivValue method of the Array struct.
// Example usage is the same as DivValue method.
func DivValue[T Float](a *Array[T], value T) *Array[T] {
	return a.DivValue(value)
}

// MulValue multiplies each element of the array by a given value.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	result := a.MulValue(2)
//	// result will be a new Array with elements [[2, 4], [6, 8]]
func (a *Array[T]) MulValue(value T) *Array[T] {
	return a.zipValue(value, func(x, y T) T { return x * y })
}

// MulValue is a wrapper function that calls the MulValue method of the Array struct.
// Example usage is the same as MulValue method.
func MulValue[T Float](a *Array[T], value T) *Array[T] {
	return a.MulValue(value)
}

// SubValue subtracts a given value from each element of the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	result := a.SubValue(1)
//	// result will be a new Array with elements [[0, 1], [2, 3]]
func (a *Array[T]) SubValue(value T) *Array[T] {
	return a.zipValue(value, func(x, y T) T { return x - y })
}

// SubValue is a wrapper function that calls the SubValue method of the Array struct.
// Example usage is the same as SubValue method.
func SubValue[T Float](a *Array[T], value T) *Array[T] {
	return a.SubValue(value)
}

// AddValue adds a given value to each element of the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	result := a.AddValue(1)
//	// result will be a new Array with elements [[2, 3], [4, 5]]
func (a *Array[T]) AddValue(value T) *Array[T] {
	return a.zipValue(value, func(x, y T) T { return x + y })
}

// AddValue is a wrapper function that calls the AddValue method of the Array struct.
// Example usage is the same as AddValue method.
func AddValue[T Float](a *Array[T], value T) *Array[T] {
	return a.AddValue(value)
}

// Atan2 computes the element-wise arctangent of y/x for arrays y and x.
// The function returns a new array containing the results.
// It panics if the shapes cannot be broadcast together.
//
// Example:
//
//	y := New2DArray([][]float64{{1, 2}, {3, 4}})
//	x := New2DArray([][]float64{{1, 1}, {1, 1}})
//	result := Atan2(y, x)
//	fmt.Println(result) // Output: Array with elements [0.7853981633974483, 1.1071487177940904, 1.2490457723982544, 1.3258176636680326]
func Atan2[T Float](y, x *Array[T]) *Array[T] {
	return y.zip(x, func(a, b T) T { return T(math.Atan2(toF64(a), toF64(b))) })
}

// Pow performs element-wise power operation between two arrays,
// broadcasting a single row, column or 1x1 array.
// It panics if the shapes cannot be broadcast together.
//
// Example:
//
//	a := New2DArray([][]float64{{2, 3}, {4, 5}})
//	b := New2DArray([][]float64{{2, 3}, {1, 2}})
//	result := a.Pow(b)
//	// result will be a new Array with values {{4, 27}, {4, 25}}
func (a *Array[T]) Pow(b *Array[T]) *Array[T] {
	return a.zip(b, func(x, y T) T { return T(math.Pow(toF64(x), toF64(y))) })
}

// Pow is a helper function that calls the Pow method on the Array a with the Array b.
//
// Example:
//
//	a := New2DArray([][]float64{{2, 3}, {4, 5}})
//	b := New2DArray([][]float64{{2, 3}, {1, 2}})
//	result := Pow(a, b)
//	// result will be a new Array with values {{4, 27}, {4, 25}}
func Pow[T Float](a, b *Array[T]) *Array[T] {
	return a.Pow(b)
}

// PowValue raises each element of the array a to the power of the scalar b.
//
// Example:
//
//	a := New2DArray([][]float64{{2, 3}, {4, 5}})
//	b := 2.0
//	result := a.PowValue(b)
//	// result will be a new Array with values {{4, 9}, {16, 25}}
func (a *Array[T]) PowValue(b T) *Array[T] {
	return a.zipValue(b, func(x, y T) T { return T(math.Pow(toF64(x), toF64(y))) })
}

// PowValue is a helper function that calls the PowValue method on the Array a with the scalar b.
//
// Example:
//
//	a := New2DArray([][]float64{{2, 3}, {4, 5}})
//	b := 2.0
//	result := PowValue(a, b)
//	// result will be a new Array with values {{4, 9}, {16, 25}}
func PowValue[T Float](a *Array[T], b T) *Array[T] {
	return a.PowValue(b)
}
