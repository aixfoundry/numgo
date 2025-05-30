package numgo

import (
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
func (a *Array) Abs() (ret *Array) {
	ret = ZeroArray(a.r, a.c)
	for i, v := range a.data {
		ret.data[i] = math.Abs(v)
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
func Abs(a *Array) (ret *Array) {
	return a.Abs()
}

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
	for k, v := range a.All() {
		vv := math.Abs(v)
		if math.Abs(sol) < vv {
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
	for k, v := range a.All() {
		vv := math.Abs(v)
		if math.Abs(sol) > vv {
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
//	sol := a.Max(-1)
//	fmt.Println(sol) // Output: [4]
func (a *Array) Max(axis int) (ret *Array) {
	if axis == 0 {
		ret = ZeroArray(1, a.c)
		for j := range a.c {
			maxVal := -math.MaxFloat64
			for i := range a.r {
				if a.At(i, j) > maxVal {
					maxVal = a.At(i, j)
				}
			}
			ret.Set(0, j, maxVal)
		}
	} else if axis == 1 {
		ret = ZeroArray(a.r, 1)
		for i := range a.r {
			maxVal := -math.MaxFloat64
			for j := range a.c {
				if a.At(i, j) > maxVal {
					maxVal = a.At(i, j)
				}
			}
			ret.Set(i, 0, maxVal)
		}
	} else {
		maxVal := -math.MaxFloat64
		for _, v := range a.All() {
			if maxVal < v {
				maxVal = v
			}
		}
		ret = New1DArray([]float64{maxVal})
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
func Max(a *Array, axis int) *Array {
	return a.Max(axis)
}

// Min returns the minimum element in the array and its position.
// It returns the value, row index, and column index of the element.
//
// Example:
//
//	a := New2DArray([][]float64{{-1, 2}, {-3, 4}})
//	sol, r, c := a.Min()
//	fmt.Println(sol, r, c) // Output: -3 1 0
func (a *Array) Min(axis int) (ret *Array) {
	if axis == 0 {
		ret = ZeroArray(1, a.c)
		for j := range a.c {
			minVal := math.MaxFloat64
			for i := range a.r {
				if a.At(i, j) < minVal {
					minVal = a.At(i, j)
				}
			}
			ret.Set(0, j, minVal)
		}
	} else if axis == 1 {
		ret = ZeroArray(a.r, 1)
		for i := range a.r {
			minVal := math.MaxFloat64
			for j := range a.c {
				if a.At(i, j) < minVal {
					minVal = a.At(i, j)
				}
			}
			ret.Set(i, 0, minVal)
		}
	} else {
		minVal := math.MaxFloat64
		for _, v := range a.All() {
			if minVal > v {
				minVal = v
			}
		}
		ret = New1DArray([]float64{minVal})
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
func Min(a *Array, axis int) *Array {
	return a.Min(axis)
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

// Median calculates the median of the Array along the specified axis.
// If axis is 0, it calculates the median for each column.
// If axis is 1, it calculates the median for each row.
// If axis is -1, it calculates the median of the entire array.
// It returns a new Array containing the median values.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 3, 5}, {2, 4, 6}})
//	median := a.Median(0) // median will be a 1x3 Array with values [1.5, 3.5, 5.5]
func (a *Array) Median(axis int) (ret *Array) {
	if axis == 0 {
		ret = ZeroArray(1, a.c)
		for ic := range a.c {
			col := a.Col(ic).Sort(axis)
			if col.r%2 == 0 {
				ret.Set(0, ic, (col.At(col.r/2-1, 0)+col.At(col.r/2, 0))/2)
			} else {
				ret.Set(0, ic, col.At(col.r/2, 0))
			}
		}
	} else if axis == 1 {
		ret = ZeroArray(a.r, 1)
		for ir := range a.r {
			row := a.Row(ir).Sort(axis)
			if row.c%2 == 0 {
				ret.Set(ir, 0, (row.At(0, row.c/2-1)+row.At(0, row.c/2))/2)
			} else {
				ret.Set(ir, 0, row.At(0, row.c/2))
			}
		}
	} else {
		median := float64(0)
		data := a.Data()
		slices.Sort(data)
		if len(data)%2 == 0 {
			median = (data[len(data)/2-1] + data[len(data)/2]) / 2
		} else {
			median = data[len(data)/2]
		}
		ret = New1DArray([]float64{median})
	}
	return
}

// Median is a convenience function that calls the Median method on the Array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 3, 5}, {2, 4, 6}})
//	median := Median(a, 0) // median will be a 1x3 Array with values [1.5, 3.5, 5.5]
func Median(a *Array, axis int) *Array {
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
func (a *Array) Sort(axis int) *Array {
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
func Sort(a *Array, axis int) *Array {
	return a.Sort(axis)
}

// Mean calculates the mean of the elements along the specified axis of the array.
// If axis is 0, it calculates the mean for each column and returns a 1xN array.
// If axis is 1, it calculates the mean for each row and returns an Nx1 array.
// If the array is empty or has zero dimensions, it returns nil.
//
// Example:
//
//	a := NewArray([][]float64{{1, 2, 3}, {4, 5, 6}})
//	mean := a.Mean(0) // mean will be a *Array with data [[2.5, 3.5, 4.5]]
//	mean := a.Mean(1) // mean will be a *Array with data [[2], [5]]
func (a *Array) Mean(axis int) (ret *Array) {
	if len(a.data) == 0 || a.r == 0 || a.c == 0 {
		return
	}
	divisor := float64(a.r * a.c)
	if axis == 0 {
		divisor = float64(a.r)
	} else if axis == 1 {
		divisor = float64(a.c)
	} else {
		divisor = float64(a.r * a.c)
	}
	ret = a.Sum(axis).DivValue(divisor)
	return
}

// Mean is a utility function that calls the Mean method on the Array receiver.
// It calculates the mean of the elements along the specified axis of the array.
//
// Example:
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
	return a.Mean(-1).At(0, 0)
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
//	sum := a.Sum(-1)
//	fmt.Println(sum) // Output: 10
func (a *Array) Sum(axis int) (ret *Array) {
	if axis == 0 {
		ret = ZeroArray(1, a.c)
		for ic := range a.c {
			ret.Set(0, ic, a.Of(ic, axis).Sum(-1).At(0, 0))
		}
	} else if axis == 1 {
		ret = ZeroArray(1, a.r)
		for ir := range a.r {
			ret.Set(0, ir, a.Of(ir, axis).Sum(-1).At(0, 0))
		}
	} else {
		sum := float64(0)
		for _, v := range a.data {
			sum += v
		}
		ret = New1DArray([]float64{sum})
	}
	return
}

// Sum is a wrapper function that calls the Sum method of the Array struct.
// It computes the sum of all elements in the array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	sum := Sum(a, -1)
//	fmt.Println(sum) // Output: 10
func Sum(a *Array, axis int) (ret *Array) {
	return a.Sum(axis)
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
		data = append(data, a.Mean(-1).At(0, 0))
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
		ca := Copy(a).T()
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
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v-b.At(k[0], k[1]))
		}
	} else if a.r != b.r && b.r == 1 && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v-b.At(0, k[1]))
		}
	} else if a.c != b.c && b.c == 1 && a.r == b.r {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v-b.At(k[0], 0))
		}
	} else if b.c == 1 && b.r == 1 {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v-b.At(0, 0))
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
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v+b.At(k[0], k[1]))
		}
	} else if a.r != b.r && b.r == 1 && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v+b.At(0, k[1]))
		}
	} else if a.c != b.c && b.c == 1 && a.r == b.r {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v+b.At(k[0], 0))
		}
	} else if b.c == 1 && b.r == 1 {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v+b.At(0, 0))
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
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v*b.At(k[0], k[1]))
		}
	} else if a.r != b.r && b.r == 1 && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v*b.At(0, k[1]))
		}
	} else if a.c != b.c && b.c == 1 && a.r == b.r {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v*b.At(k[0], 0))
		}
	} else if b.c == 1 && b.r == 1 {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], v*b.At(0, 0))
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
		for k, v := range a.All() {
			if b.At(k[0], k[1]) == 0 {
				ret.Set(k[0], k[1], math.NaN())
			} else {
				ret.Set(k[0], k[1], v/b.At(k[0], k[1]))
			}
		}
	} else if a.r != b.r && b.r == 1 && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			if b.At(0, k[1]) == 0 {
				ret.Set(k[0], k[1], math.NaN())
			} else {
				ret.Set(k[0], k[1], v/b.At(0, k[1]))
			}
		}
	} else if a.c != b.c && b.c == 1 && a.r == b.r {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			if b.At(k[0], 0) == 0 {
				ret.Set(k[0], k[1], math.NaN())
			} else {
				ret.Set(k[0], k[1], v/b.At(k[0], 0))
			}
		}
	} else if b.c == 1 && b.r == 1 {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			if b.At(0, 0) == 0 {
				ret.Set(k[0], k[1], math.NaN())
			} else {
				ret.Set(k[0], k[1], v/b.At(0, 0))
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
	for k, v := range a.All() {
		if value == 0 {
			ret.Set(k[0], k[1], math.NaN())
		} else {
			ret.Set(k[0], k[1], v/value)
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
	for k, v := range a.All() {
		ret.Set(k[0], k[1], v*value)
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
	for k, v := range a.All() {
		ret.Set(k[0], k[1], v-value)
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
	for k, v := range a.All() {
		ret.Set(k[0], k[1], v+value)
	}
	return
}

// AddValue is a wrapper function that calls the AddValue method of the Array struct.
// Example usage is the same as AddValue method.
func AddValue(a *Array, value float64) (ret *Array) {
	return a.AddValue(value)
}

// Atan2 computes the element-wise arctangent of y/x for arrays y and x.
// The function returns a new array containing the results.
// If the dimensions of y and x do not match, the function returns nil.
//
// Example:
//
//	y := New2DArray([][]float64{{1, 2}, {3, 4}})
//	x := New2DArray([][]float64{{1, 1}, {1, 1}})
//	result := Atan2(y, x)
//	fmt.Println(result) // Output: Array with elements [0.7853981633974483, 1.1071487177940904, 1.2490457723982544, 1.3258176636680326]
func Atan2(y *Array, x *Array) (ret *Array) {
	if y.r != x.r || y.c != x.c {
		return
	}
	ret = ZeroArray(y.r, y.c)
	for k, v := range y.All() {
		ret.Set(k[0], k[1], math.Atan2(v, x.At(k[0], k[1])))
	}
	return ret
}

// Pow performs element-wise power operation between two arrays.
// If the dimensions of a and b match, it raises each element of a to the power of the corresponding element in b.
// If b is a row vector and a and b have the same number of columns, it raises each element of a to the power of the corresponding element in b's row.
// If b is a column vector and a and b have the same number of rows, it raises each element of a to the power of the corresponding element in b's column.
// If b is a scalar (1x1 matrix), it raises each element of a to the power of that scalar.
//
// Example:
//
//	a := New2DArray([][]float64{{2, 3}, {4, 5}})
//	b := New2DArray([][]float64{{2, 3}, {1, 2}})
//	result := a.Pow(b)
//	// result will be a new Array with values {{4, 27}, {4, 25}}
func (a *Array) Pow(b *Array) (ret *Array) {
	if a.r == b.r && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], math.Pow(v, b.At(k[0], k[1])))
		}
	} else if a.r != b.r && b.r == 1 && a.c == b.c {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], math.Pow(v, b.At(0, k[1])))
		}
	} else if a.c != b.c && b.c == 1 && a.r == b.r {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], math.Pow(v, b.At(k[0], 0)))
		}
	} else if b.c == 1 && b.r == 1 {
		ret = ZeroArray(a.r, a.c)
		for k, v := range a.All() {
			ret.Set(k[0], k[1], math.Pow(v, b.At(0, 0)))
		}
	}
	return
}

// Pow is a helper function that calls the Pow method on the Array a with the Array b.
//
// Example:
//
//	a := New2DArray([][]float64{{2, 3}, {4, 5}})
//	b := New2DArray([][]float64{{2, 3}, {1, 2}})
//	result := Pow(a, b)
//	// result will be a new Array with values {{4, 27}, {4, 25}}
func Pow(a *Array, b *Array) *Array {
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
func (a *Array) PowValue(b float64) (ret *Array) {
	ret = ZeroArray(a.r, a.c)
	for k, v := range a.All() {
		ret.Set(k[0], k[1], math.Pow(v, b))
	}
	return
}

// PowValue is a helper function that calls the PowValue method on the Array a with the scalar b.
//
// Example:
//
//	a := New2DArray([][]float64{{2, 3}, {4, 5}})
//	b := 2.0
//	result := PowValue(a, b)
//	// result will be a new Array with values {{4, 9}, {16, 25}}
func PowValue(a *Array, b float64) *Array {
	return a.PowValue(b)
}
