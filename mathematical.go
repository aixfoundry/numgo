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
func (a *Array) MovingAvg(window int, step int, axios int) (ret *Array) {
	data := []float64{}
	if window <= 0 || step <= 0 || window > len(a.data) {
		data = append(data, a.Avg())
		ret = New1DArray(data)
		return
	}
	if axios == 0 {
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
	} else if axios == 1 {
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
func MovingAvg(a *Array, window int, step int, axios int) (ret *Array) {
	return a.MovingAvg(window, step, axios)
}
