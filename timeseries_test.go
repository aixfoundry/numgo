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
