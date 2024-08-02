package numgo

import (
	"fmt"
	"testing"
)

func TestArray_Max(t *testing.T) {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := a.Max()
	if sol != 13 || r != 0 || c != 4 {
		t.Log(sol, r, c)
		t.FailNow()
	}
}

func ExampleArray_Max() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := a.Max()
	fmt.Println(sol, r, c)
	// Output:
	// 13 0 4
}

func ExampleMax() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := Max(a)
	fmt.Println(sol, r, c)
	// Output:
	// 13 0 4
}

func ExampleArray_MaxAbs() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := a.MaxAbs()
	fmt.Println(sol, r, c)
	// Output:
	// 13 0 4
}

func ExampleMaxAbs() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := MaxAbs(a)
	fmt.Println(sol, r, c)
	// Output:
	// 13 0 4
}

func ExampleArray_MinAbs() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := a.MinAbs()
	fmt.Println(sol, r, c)
	// Output:
	// 0 0 0
}

func ExampleMinAbs() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := MinAbs(a)
	fmt.Println(sol, r, c)
	// Output:
	// 0 0 0
}

func ExampleArray_Min() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := a.Min()
	fmt.Println(sol, r, c)
	// Output:
	// 0 0 0
}

func ExampleMin() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := Min(a)
	fmt.Println(sol, r, c)
	// Output:
	// 0 0 0
}

func ExampleArray_Dot() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	b := NewArray(9, 1, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	c := a.Dot(b)
	fmt.Println(c)
	// Output:
	// [374]
}

func ExampleDot() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	b := NewArray(9, 1, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	c := Dot(a, b)
	fmt.Println(c)
	// Output:
	// [374]
}
