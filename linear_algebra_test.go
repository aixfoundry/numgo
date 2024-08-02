package numgo

import (
	"fmt"
)

func ExampleEMCP() {
	a := NewArray(3, 3, []float64{
		2, 1, -1,
		-3, -1, 2,
		-2, 1, 2,
	})
	b := NewArray(3, 1, []float64{
		8,
		-11,
		-3,
	})
	ab := Cat(a, b, 1)
	sol := EMCP(ab)
	fmt.Println(sol)
	// Output:
	// [3.230769230769231  8.846153846153847  -2.692307692307692]
}

func ExampleLstsq() {
	a := NewArray(6, 3, []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		-1.0, 1.0, 0.0,
		0.0, -1.0, 1.0,
		-1.0, 0.0, 1.0,
	})
	b := NewArray(6, 1, []float64{
		1.0,
		2.0,
		3.0,
		1.0,
		2.0,
		1.0,
	})
	sol := Lstsq(a, b)
	fmt.Println(sol)
	// Output:
	// [1.25  1.7500000000000002  3]
}

func ExampleLsqFit() {
	a := NewArray(8, 2, []float64{
		-1.0, 10.0,
		0.0, 9.0,
		1.0, 7.0,
		2.0, 5.0,
		3.0, 4.0,
		4.0, 3.0,
		5.0, 0.0,
		6.0, -1.0})
	sol := LsqFit(a)
	fmt.Println(sol)
	// Output:
	// [-2.2692307692307705  8.798076923076925]
}

func ExamplePolyFit_lsq() {
	a := NewArray(8, 2, []float64{
		-1.0, 10.0,
		0.0, 9.0,
		1.0, 7.0,
		2.0, 5.0,
		3.0, 4.0,
		4.0, 3.0,
		5.0, 0.0,
		6.0, -1.0})
	sol := PolyFit(a, 1)
	fmt.Println(sol)
	// Output:
	// [-2.2692307692307705  8.798076923076925]
}
