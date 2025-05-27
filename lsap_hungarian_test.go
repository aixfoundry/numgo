package numgo

import "fmt"

func ExampleHungarianSolveMax() {
	a := New2DArray([][]float64{
		{6, 2, 3, 4, 5},
		{3, 8, 2, 8, 1},
		{9, 9, 5, 4, 2},
		{6, 7, 3, 4, 3},
		{1, 2, 6, 4, 9},
	})
	vals, row, col := HungarianSolveMax(a)
	fmt.Println(row)
	fmt.Println(col)
	fmt.Println(vals)
}
