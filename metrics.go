package numgo

import (
	"math"
)

// ShapeContext computes the shape context descriptor for a given array of points.
// It calculates the distribution of points around each point in terms of their normalized distances and angles.
// The function returns a new array containing the concatenated histograms for all points.
//
// Parameters:
//
//	a - The input array of points.
//	radiusNum - The number of bins for the radial distance in the histogram.
//	angleNum - The number of bins for the angle in the histogram.
//
// Returns:
//
//	ret - A new array containing the shape context descriptors.
//
// Example:
//
//	points := New2DArray([][]float64{{0, 0}, {1, 0}, {0, 1}})
//	shapeContext := ShapeContext(points, 5.0, 8.0)
func ShapeContext(a *Array, radiusNum float64, angleNum float64) (ret *Array) {
	n := a.r
	data := [][]float64{}
	for i := range n {
		p := a.Row(i)
		coords := a.Sub(p)
		r := coords.Norm(2, 1)
		angle := Atan2(coords.Col(1), coords.Col(0))
		normR := r.Div(r.Filter(func(r, c int, v float64) bool { return v != 0 }).Median(-1))
		hist, _, _ := Histogram2d(normR, angle, []int{int(radiusNum), int(angleNum)}, [][]float64{{0, 2}, {-math.Pi, math.Pi}})
		data = append(data, hist.data)
	}
	ret = New2DArray(data)
	return
}

const EPSILON = 1e-10

// ChiSquareDistance calculates the chi-square distance between two arrays.
// This distance measure is often used in histogram comparison.
//
// Parameters:
//
//	a - The first input array.
//	b - The second input array.
//
// Returns:
//
//	ret - The chi-square distance between the two arrays.
//
// Example:
//
//	array1 := New1DArray([]float64{1, 2, 3})
//	array2 := New1DArray([]float64{3, 2, 1})
//	distance := ChiSquareDistance(array1, array2)
func ChiSquareDistance(a *Array, b *Array) (ret float64) {
	ret = a.Sub(b).PowValue(2).Div(a.Add(b).AddValue(EPSILON)).Sum(-1).At(0, 0) * 0.5
	return
}

// CostMatrix calculates the cost matrix between two arrays 'a' and 'b' using the Chi-Square distance.
// It returns a new 2D array where each element (i, j) is the Chi-Square distance between the i-th row of 'a' and the j-th row of 'b'.
//
// Example:
//
//	a := New2DArray([][]float64{{1, 2}, {3, 4}})
//	b := New2DArray([][]float64{{5, 6}, {7, 8}})
//	costMatrix := CostMatrix(a, b)
//	fmt.Println(costMatrix) // Output will be a 2x2 matrix with Chi-Square distances
func CostMatrix(a *Array, b *Array) (ret *Array) {
	n := a.r
	m := b.r
	data := make([][]float64, n)
	for i := range n {
		data[i] = make([]float64, m)
		for j := range m {
			data[i][j] = ChiSquareDistance(a.Row(i), b.Row(j))
		}
	}
	ret = New2DArray(data)
	return
}
