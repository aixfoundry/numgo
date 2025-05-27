package numgo

import (
	"fmt"
)

func ExampleChiSquareDistance() {
	array1 := New1DArray([]float64{1, 2, 3})
	array2 := New1DArray([]float64{3, 2, 1})
	distance := ChiSquareDistance(array1, array2)
	fmt.Println(distance)
	// Output:
	// 0.999999999975
}

func ExampleShapeContext() {
	points := New2DArray([][]float64{{1, 3}, {2, 4}})
	sc := ShapeContext(points, 5.0, 12.0)
	fmt.Println(sc)
	// Output:
	// ⎡0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0⎤
	// ⎣0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0⎦
}

func ExampleCostMatrix() {
	a := New2DArray([][]float64{{1, 2}, {3, 4}})
	b := New2DArray([][]float64{{5, 6}, {7, 8}})
	costMatrix := CostMatrix(a, b)
	fmt.Println(costMatrix) // Output will be a 2x2 matrix with Chi-Square distances
	// Output:
	// ⎡2.333333333298611  4.049999999953875⎤
	// ⎣0.449999999994875  1.466666666653111⎦
}
