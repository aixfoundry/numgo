package numgo

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
	x := CopyOf(a)
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
