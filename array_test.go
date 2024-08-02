package numgo

import (
	"fmt"
)

func ExampleArray_Of_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Of(0, 0))
	// Output:
	// [1  2  3]
}

func ExampleArray_Of_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Of(0, 1))
	// Output:
	// ⎡1⎤
	// ⎢4⎥
	// ⎣7⎦
}

func ExampleArray_Dims() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	r, c := a.Dims()
	fmt.Printf("%v %v", r, c)
	// Output:
	// 3 3
}

func ExampleArray_At() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.At(1, 1))
	// Output:
	// 5
}

func ExampleArray_Set() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Set(1, 1, 11))
	// Output:
	// ⎡1   2  3⎤
	// ⎢4  11  6⎥
	// ⎣7   8  9⎦
}
