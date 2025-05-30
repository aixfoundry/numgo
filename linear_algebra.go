package numgo

import "math"

// EMCP performs Gaussian elimination with partial pivoting (column main element elimination) on an augmented matrix.
//
// The function solves the system of linear equations A x = b, where A is a square matrix.
//
// Parameters:
//
//	a - an augmented matrix of size (n, n+1), where the first n columns represent matrix A and the last column represents vector b.
//
// Returns:
//
//	sol - a solution vector of size (1, n) if the system has a unique solution.
//
// Example:
//
//	a := NewArray(3, 4, []float64{1, 2, -1, 8, -4, -5, 1, 10, 1, 0, 3, 3})
//	sol := EMCP(a)
//	// sol should be [2, -1, 3]
func EMCP(a *Array) (sol *Array) {
	if a.c-1 != a.r {
		return
	}
	x := Copy(a)
	n := x.r // n is the number of equations
	// Convert to upper triangular form
	for i := range n { // Traverse all rows
		// Find the pivot element in the current column and swap rows if necessary
		cols := []float64{}
		for ic := i; ic < n; ic++ { // Traverse unprocessed rows
			cols = append(cols, x.At(ic, i)) // Collect elements from column i
		}
		acol := NewArray(1, len(cols), cols)
		_, _, c := acol.MaxAbs()
		if c+i != i {
			x.Swap(c+i, i, 0)
		}
		for j := i + 1; j < n; j++ {
			mul := x.At(j, i) / x.At(i, i)
			for k := i; k < x.c; k++ {
				x.Set(j, k, x.At(j, k)-x.At(i, k)*mul)
			}
		}
	}
	buf := make([]float64, n)
	buf[n-1] = x.At(n-1, x.c-1) / x.At(n-1, n-1)
	for i := n - 2; i >= 0; i-- {
		t := 0.0
		for j := i + 1; j < n; j++ {
			t += x.At(i, j) * buf[j]
		}
		buf[i] = (x.At(i, x.c-1) - t) / x.At(i, i)
	}
	return &Array{
		r:    1,
		c:    n,
		data: buf,
	}
}

// Lstsq computes the least-squares solution to a linear matrix equation using the EMCP method.
//
// It solves the system of linear equations A x = b, where A is a matrix and b is a vector.
//
// Parameters:
//
//	a - the coefficient matrix of size (m, n)
//	b - the constant vector of size (m, 1)
//
// Returns:
//
//	sol - the least-squares solution vector of size (1, n)
//
// Example:
//
//	a := NewArray(2, 3, []float64{1, 0, 1, 1, 1, 0})
//	b := NewArray(2, 1, []float64{3, 4})
//	sol := Lstsq(a, b)
//	// sol should be [2, 1]
func Lstsq(a *Array, b *Array) (sol *Array) {
	if a.r != b.r {
		return
	}
	at := T(a)
	aa := Dot(at, a)
	ab := Dot(at, b)
	return EMCP(Cat(aa, ab, 1))
}

// LsqFit performs linear least squares fitting for a set of points using the EMCP method.
//
// It finds the best fit line y = mx + c for a given set of points (x, y).
//
// Parameters:
//
//	xy - an array of size (n, 2) where each row represents a point (x, y)
//
// Returns:
//
//	sol - a solution vector [c, m] of size (1, 2) representing the intercept and slope of the best fit line
//
// Example:
//
//	xy := NewArray(3, 2, []float64{1, 2, 2, 3, 3, 4})
//	sol := LsqFit(xy)
//	// sol should be [1, 1]
func LsqFit(xy *Array) (sol *Array) {
	ex2, ex, exy, ey := 0.0, 0.0, 0.0, 0.0
	for ir := range xy.r {
		ex2 += xy.At(ir, 0) * xy.At(ir, 0)
		ex += xy.At(ir, 0)
		exy += xy.At(ir, 0) * xy.At(ir, 1)
		ey += xy.At(ir, 1)
	}
	a := NewArray(2, 3, []float64{
		float64(xy.r), ex, ey,
		ex, ex2, exy,
	})
	return EMCP(a)
}

// FrobeniusNorm calculates the Frobenius norm of the array along the specified axis.
// If axis is 0, it calculates the norm for each column.
// If axis is 1, it calculates the norm for each row.
// If axis is -1, it calculates the norm for the entire array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	norm := a.FrobeniusNorm(0) // norm will be a 1x2 array with values [sqrt(10), sqrt(25)]
//	norm := a.FrobeniusNorm(1) // norm will be a 2x1 array with values [sqrt(5), sqrt(25)]
//	norm := a.FrobeniusNorm(-1) // norm will be a 1x1 array with value [sqrt(30)]
func (a *Array) FrobeniusNorm(axis int) (ret *Array) {
	if axis == 0 {
		ret = ZeroArray(1, a.c)
		for ic := range a.c {
			ret.Set(0, ic, a.Col(ic).FrobeniusNorm(-1).At(0, 0))
		}
	} else if axis == 1 {
		ret = ZeroArray(a.r, 1)
		for ir := range a.r {
			ret.Set(ir, 0, a.Row(ir).FrobeniusNorm(-1).At(0, 0))
		}
	} else {
		sum := 0.0
		for _, v := range a.data {
			sum += v * v
		}
		ret = New1DArray([]float64{math.Sqrt(sum)})
	}
	return
}

// FrobeniusNorm is a wrapper function that calls the FrobeniusNorm method of the Array struct.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	norm := FrobeniusNorm(a, 0) // norm will be a 1x2 array with values [sqrt(10), sqrt(25)]
func FrobeniusNorm(a *Array, axis int) (ret *Array) {
	return a.FrobeniusNorm(axis)
}

// Norm1 calculates the maximum absolute row sum of the array.
// It returns a new 1D array containing the maximum norm value.
//
// Example:
//
//	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
//	norm := a.Norm1(-1) // norm will be a 1D array with value [9]
func (a *Array) Norm1(axis int) (ret *Array) {
	if axis == 0 {
		ret = ZeroArray(1, a.c)
		for ic := range a.c {
			norm := a.Col(ic).Abs().Sum(-1).At(0, 0)
			ret.Set(0, ic, norm)
		}
	} else if axis == 1 {
		ret = ZeroArray(a.r, 1)
		for ir := range a.r {
			norm := a.Row(ir).Abs().Sum(-1).At(0, 0)
			ret.Set(ir, 0, norm)
		}
	} else {
		norm, max := 0.0, 0.0
		for ic := range a.c {
			norm = a.Col(ic).Abs().Sum(-1).At(0, 0)
			if norm > max {
				max = norm
			}
		}
		ret = New1DArray([]float64{max})
	}

	return
}

// Norm1 is a wrapper function that calls the Norm1 method on the Array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
//	norm := Norm1(a, -1) // norm will be a 1D array with value [9]
func Norm1(a *Array, axis int) *Array {
	return a.Norm1(axis)
}

// Norm2 calculates the Frobenius norm of the array along the specified axis.
// It returns a new array representing the norm along the specified axis.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	norm := a.Norm2(0) // norm will be a 1D array with value [sqrt(10), sqrt(25)]
func (a *Array) Norm2(axis int) *Array {
	return a.FrobeniusNorm(axis)
}

// Norm2 is a wrapper function that calls the Norm2 method on the Array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	norm := Norm2(a, 0) // norm will be a 1D array with value [sqrt(10), sqrt(25)]
func Norm2(a *Array, axis int) *Array {
	return a.Norm2(axis)
}

// NormInf calculates the maximum absolute column sum of the array.
// It returns a new 1D array containing the maximum norm value.
//
// Example:
//
//	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
//	norm := a.NormInf(-1) // norm will be a 1D array with value [15]
func (a *Array) NormInf(axis int) (ret *Array) {
	if axis == 0 {
		ret = ZeroArray(1, a.c)
		for ic := range a.c {
			sol, _, _ := a.Col(ic).MaxAbs()
			ret.Set(0, ic, sol)
		}
	} else if axis == 1 {
		ret = ZeroArray(a.r, 1)
		for ir := range a.r {
			sol, _, _ := a.Row(ir).MaxAbs()
			ret.Set(ir, 0, sol)
		}
	} else {
		norm, max := 0.0, 0.0
		for ir := range a.r {
			norm = a.Row(ir).Abs().Sum(-1).At(0, 0)
			if norm > max {
				max = norm
			}
		}
		ret = New1DArray([]float64{max})
	}

	return
}

// NormInf is a wrapper function that calls the NormInf method on the Array.
//
// Example:
//
//	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
//	norm := NormInf(a) // norm will be a 1D array with value [15]
func NormInf(a *Array, axis int) *Array {
	return a.NormInf(axis)
}

// Norm calculates the norm of the array based on the given order and axis.
// It supports the following norms:
// - ord = math.Inf(1): Maximum absolute column sum (L∞ norm)
// - ord = 2: Frobenius norm
// - ord = 1: Maximum absolute row sum (L1 norm)
//
// Examples:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	fmt.Println(a.Norm(math.Inf(1), 0)) // Output: [6]
//	fmt.Println(a.Norm(2, 0))          // Output: [5.477225575051661]
//	fmt.Println(a.Norm(1, 0))          // Output: [4]
func (a *Array) Norm(ord float64, axis int) (ret *Array) {
	switch ord {
	case math.Inf(1):
		ret = a.NormInf(axis)
	case 2:
		ret = a.FrobeniusNorm(axis)
	case 1:
		ret = a.Norm1(axis)
	}
	return ret
}

// Norm is a convenience function that calls the Norm method of the Array receiver.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	fmt.Println(Norm(a, math.Inf(1), 0)) // Output: [6]
func Norm(a *Array, ord float64, axis int) *Array {
	return a.Norm(ord, axis)
}

// Histogram2d computes the 2D histogram of the input data arrays x and y.
// It returns the histogram array, and the bin edges along the x and y axes.
// The bins parameter specifies the number of bins along each dimension.
// The range_ parameter allows specifying the range of values to consider for each dimension.
// If range_ is not provided, the function will use the minimum and maximum values of x and y.
//
// Example:
//
//	x := New1DArray([]float64{1, 2, 2.5, 3, 3.5, 4})
//	y := New1DArray([]float64{2, 3, 3.5, 4, 4.5, 5})
//	bins := []int{3, 3}
//	hist, xEdges, yEdges := Histogram2d(x, y, bins, [][]float64{})
//
// The example above computes a 3x3 histogram for the given x and y data.
func Histogram2d(x, y *Array, bins []int, range_ [][]float64) (hist *Array, xEdges *Array, yEdges *Array) {
	if len(bins) != 2 {
		panic("Invalid bins")
	}
	x, y = x.Reshape(1, x.r*x.c), y.Reshape(1, y.r*y.c)
	var xMin, xMax, yMin, yMax float64
	if len(range_) == 2 && len(range_[0]) == 2 && len(range_[1]) == 2 {
		xMin = range_[0][0]
		xMax = range_[0][1]
		yMin = range_[1][0]
		yMax = range_[1][1]
	} else {
		xMin = x.Min(-1).At(0, 0)
		xMax = x.Max(-1).At(0, 0)
		yMin = y.Min(-1).At(0, 0)
		yMax = y.Max(-1).At(0, 0)
	}

	xBinWidth := (xMax - xMin) / float64(bins[0])
	yBinWidth := (yMax - yMin) / float64(bins[1])

	hist = ZeroArray(bins[0], bins[1])

	for i := range x.c {
		if len(range_) > 0 {
			if x.At(0, i) < xMin || x.At(0, i) > xMax || y.At(0, i) < yMin || y.At(0, i) > yMax {
				continue
			}
		}
		// Determine the bin index for x and y
		xIdx := int((x.At(0, i) - xMin) / xBinWidth)
		yIdx := int((y.At(0, i) - yMin) / yBinWidth)
		// Handle edge case where value is exactly at the upper limit
		if xIdx == bins[0] {
			xIdx--
		}
		if yIdx == bins[1] {
			yIdx--
		}
		if xIdx >= 0 && xIdx < bins[0] && yIdx >= 0 && yIdx < bins[1] {
			v := hist.At(xIdx, yIdx)
			hist.Set(xIdx, yIdx, v+1)
		}
	}

	// Create the bin edges arrays
	edges := []float64{}
	for i := 0; i <= bins[0]; i++ {
		edges = append(edges, xMin+float64(i)*xBinWidth)
	}
	xEdges = New1DArray(edges)
	edges = []float64{}
	for i := 0; i <= bins[1]; i++ {
		edges = append(edges, yMin+float64(i)*yBinWidth)
	}
	yEdges = New1DArray(edges)
	return
}
