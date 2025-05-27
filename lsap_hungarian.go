package numgo

import (
	"fmt"
)

// matching recursively adjusts the results map to resolve column conflicts
// by replacing values with more optimal ones from the ncost matrix.
// It ensures that each column is assigned only once and the total cost is minimized.
//
// Example:
//
//	Given results = {0: {0: 1.0}, 1: {0: 2.0}} and ncost = Array with values
//	that might lead to a conflict in column 0, the function may modify results to
//	{0: {1: 0.5}, 1: {0: 2.0}} if 0.5 is a better value than 1.0 for column 1.
func matching(results map[int]map[int]float64, ncost *Array) {
	for ir, irow := range results {
		for ic := range irow {
			// Check for column conflicts
			for jr, jrow := range results {
				for jc := range jrow {
					if ir != jr && ic == jc {
						for vv, miv := range ncost.Row(jr).All() {
							mik := vv[1]
							thereIs := false
							// Check if the column is already occupied
							for _, rrv := range results {
								if _, ok := rrv[mik]; ok {
									thereIs = true
									break
								}
							}
							// Find an available column to replace
							if !thereIs {
								// Choose the better value to replace
								if results[jr][jc] < results[ir][jc] {
									delete(results[jr], jc)
									results[jr][mik] = miv
								} else {
									delete(results[ir], jc)
									results[ir][mik] = miv
								}
								matching(results, ncost)
							}
						}
					}
				}
			}
		}
	}
}

// HungarianSolveMax solves the assignment problem for a given cost matrix
// to maximize the total cost. It returns the values, row indices, and column indices
// of the optimal assignment.
//
// Example:
//
//	cost = Array{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
//	vals, row, col = HungarianSolveMax(cost)
//	vals might be Array{9, 5, 1} representing the maximum values from each row
//	row might be Array{2, 1, 0} representing the row indices of the selected values
//	col might be Array{2, 1, 0} representing the column indices of the selected values
func HungarianSolveMax(cost *Array) (vals *Array, row *Array, col *Array) {
	n, m := cost.r, cost.c
	ncost := CopyOf(cost)
	// Ensure the cost matrix is square by padding with the maximum value if necessary
	if n > m {
		max := ncost.Max(-1).At(0, 0)
		pad := NewValueArray(n, n-m, max)
		ncost = ncost.Cat(pad, 1)
	} else if n < m {
		max := ncost.Max(-1).At(0, 0)
		pad := NewValueArray(m-n, m, max)
		ncost = ncost.Cat(pad, 0)
	}

	// Reduce the cost matrix to make it easier to find the optimal assignment
	reduced := ncost.Sub(ncost.Max(1)).MulValue(-1) // Row max reduction for max cost
	reduced = reduced.Sub(reduced.Min(1))           // Further row reduction
	reduced = reduced.Sub(reduced.Min(0))           // Column reduction

	results := map[int]map[int]float64{}
	n = ncost.r
	// Mark all zero positions in the reduced cost matrix
	for k, v := range reduced.All() {
		if v == 0 {
			if results[k[0]] == nil {
				results[k[0]] = make(map[int]float64, n)
			}
			results[k[0]][k[1]] = ncost.At(k[0], k[1])
		}
	}

	// Resolve conflicts where multiple zeros exist in the same column
	for ir, irow := range results {
		for ic := range irow {
			// Only process columns with multiple elements
			if len(irow) > 1 {
				for jr, jrow := range results {
					for jc := range jrow {
						// Check for column conflict (same column in different rows)
						if ir != jr && ic == jc {
							// Remove conflicting elements from the row with more elements
							if len(jrow) > len(irow) {
								delete(results[jr], jc)
							} else {
								delete(results[ir], ic)
							}
						}
					}
				}
			}
		}
	}

	// Clean up columns that still have multiple elements after conflict resolution
	for ir, irow := range results {
		for ic := range irow {
			if len(irow) > 1 {
				delete(results[ir], ic)
			}
		}
	}
	matching(results, ncost)
	fmt.Println(results)
	data := []float64{}
	ridx := []float64{}
	cidx := []float64{}
	for ir, irow := range results {
		ridx = append(ridx, float64(ir))
		for ic, jcol := range irow {
			cidx = append(cidx, float64(ic))
			data = append(data, jcol)
		}
	}
	vals = New1DArray(data)
	row = New1DArray(ridx)
	col = New1DArray(cidx)
	return
}
