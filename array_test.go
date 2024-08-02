package numgo

import (
	"fmt"
)

func ExampleArray_Cat_row() {
	a := NewArray(6, 3, []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		-1.0, 1.0, 0.0,
		0.0, -1.0, 1.0,
		-1.0, 0.0, 1.0,
	})
	b := NewArray(1, 3, []float64{
		1.0,
		2.0,
		3.0,
	})
	fmt.Printf("%v", a.Cat(b, 0))
	// Output:
	// ⎡ 1   0  0⎤
	// ⎢ 0   1  0⎥
	// ⎢ 0   0  1⎥
	// ⎢-1   1  0⎥
	// ⎢ 0  -1  1⎥
	// ⎢-1   0  1⎥
	// ⎣ 1   2  3⎦
}

func ExampleArray_Cat_col() {
	a := NewArray(3, 3, []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
	})
	b := NewArray(3, 1, []float64{
		1.0,
		2.0,
		3.0,
	})
	fmt.Printf("%v", a.Cat(b, 1))
	// Output:
	// ⎡1  0  0  1⎤
	// ⎢0  1  0  2⎥
	// ⎣0  0  1  3⎦
}

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

func ExampleArray_Swap_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Swap(0, 1, 0))
	// Output:
	// ⎡4  5  6⎤
	// ⎢1  2  3⎥
	// ⎣7  8  9⎦
}

func ExampleArray_Swap_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Swap(0, 1, 1))
	// Output:
	// ⎡2  1  3⎤
	// ⎢5  4  6⎥
	// ⎣8  7  9⎦
}

func ExampleArray_Reshape_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Reshape(2, 5))
	// Output:
	// ⎡1  2  3  4  5⎤
	// ⎣6  7  8  9  0⎦
}

func ExampleArray_Reshape_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Reshape(5, 2))
	// Output:
	// ⎡1  2⎤
	// ⎢3  4⎥
	// ⎢5  6⎥
	// ⎢7  8⎥
	// ⎣9  0⎦
}

func ExampleArray_Reshape_reduce() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Reshape(2, 2))
	// Output:
	// ⎡1  2⎤
	// ⎣3  4⎦
}

func ExampleArray_T() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.T())
	// Output:
	// ⎡1  4  7⎤
	// ⎢2  5  8⎥
	// ⎣3  6  9⎦
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

func ExampleArray_Delete_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Delete(1, 0))
	// Output:
	// ⎡1  2  3⎤
	// ⎣7  8  9⎦
}

func ExampleArray_Delete_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Delete(1, 1))
	// Output:
	// ⎡1  3⎤
	// ⎢4  6⎥
	// ⎣7  9⎦
}

func ExampleArray_RangeDelete_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.RangeDelete(0, 1, 0))
	// Output:
	// ⎡4  5  6⎤
	// ⎣7  8  9⎦
}

func ExampleArray_RangeDelete_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.RangeDelete(0, 1, 1))
	// Output:
	// ⎡2  3⎤
	// ⎢5  6⎥
	// ⎣8  9⎦
}
