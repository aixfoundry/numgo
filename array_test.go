package numgo

import (
	"fmt"
	"testing"
)

func TestArray_Cat(t *testing.T) {
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
	t.Logf("\n%v\n", a)
	t.Logf("\n%v\n", b)

	t.Logf("\n%v\n", Cat(a, b, 0))
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
	a.Swap(0, 1, 0)
	fmt.Printf("%v", a)
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
	a.Swap(0, 1, 1)
	fmt.Printf("%v", a)
	// Output:
	// ⎡2  1  3⎤
	// ⎢5  4  6⎥
	// ⎣8  7  9⎦
}
