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
