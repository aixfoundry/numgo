package numgo

import "fmt"

func ExampleCat_row() {
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
	fmt.Printf("%v", Cat(a, b, 0))
	// Output:
	// ⎡ 1   0  0⎤
	// ⎢ 0   1  0⎥
	// ⎢ 0   0  1⎥
	// ⎢-1   1  0⎥
	// ⎢ 0  -1  1⎥
	// ⎢-1   0  1⎥
	// ⎣ 1   2  3⎦
}

func ExampleCat_col() {
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
	fmt.Printf("%v", Cat(a, b, 1))
	// Output:
	// ⎡1  0  0  1⎤
	// ⎢0  1  0  2⎥
	// ⎣0  0  1  3⎦
}

func ExampleT() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", T(a))
	// Output:
	// ⎡1  4  7⎤
	// ⎢2  5  8⎥
	// ⎣3  6  9⎦
}

func ExampleReshape_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", Reshape(a, 2, 5))
	// Output:
	// ⎡1  2  3  4  5⎤
	// ⎣6  7  8  9  0⎦
}

func ExampleReshape_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", Reshape(a, 5, 2))
	// Output:
	// ⎡1  2⎤
	// ⎢3  4⎥
	// ⎢5  6⎥
	// ⎢7  8⎥
	// ⎣9  0⎦
}

func ExampleReshape_reduce() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", Reshape(a, 2, 2))
	// Output:
	// ⎡1  2⎤
	// ⎣3  4⎦
}

func ExampleSwap_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", Swap(a, 0, 1, 0))
	// Output:
	// ⎡4  5  6⎤
	// ⎢1  2  3⎥
	// ⎣7  8  9⎦
}

func ExampleSwap_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", Swap(a, 0, 1, 1))
	// Output:
	// ⎡2  1  3⎤
	// ⎢5  4  6⎥
	// ⎣8  7  9⎦
}

func ExampleDelete_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", Delete(a, 1, 0))
	// Output:
	// ⎡1  2  3⎤
	// ⎣7  8  9⎦
}

func ExampleDelete_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", Delete(a, 1, 1))
	// Output:
	// ⎡1  3⎤
	// ⎢4  6⎥
	// ⎣7  9⎦
}

func ExampleRangeDelete_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", RangeDelete(a, 0, 1, 0))
	// Output:
	// ⎡4  5  6⎤
	// ⎣7  8  9⎦
}

func ExampleRangeDelete_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", RangeDelete(a, 0, 1, 1))
	// Output:
	// ⎡2  3⎤
	// ⎢5  6⎥
	// ⎣8  9⎦
}

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

func ExampleArray_Swap_row() {
	a := NewArray(3, 4, []float64{
		1, 2, 3, 3,
		4, 5, 6, 2,
		7, 8, 9, 2,
	})
	fmt.Printf("%v", a.Swap(0, 1, 0))
	// Output:
	// ⎡4  5  6  2⎤
	// ⎢1  2  3  3⎥
	// ⎣7  8  9  2⎦
}

func ExampleArray_Swap_col() {
	a := NewArray(3, 4, []float64{
		1, 2, 3, 3,
		4, 5, 6, 2,
		7, 8, 9, 2,
	})
	fmt.Printf("%v", a.Swap(0, 1, 1))
	// Output:
	// ⎡2  1  3  3⎤
	// ⎢5  4  6  2⎥
	// ⎣8  7  9  2⎦
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
