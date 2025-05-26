package numgo

import (
	"fmt"
)

func ExampleEMCP() {
	a := NewArray(3, 3, []float64{
		1.0, 4.0, -5.0,
		1.0, 3.0, -2.0,
		6.0, -1.0, 18.0,
	})
	b := NewArray(3, 1, []float64{
		3,
		2,
		2,
	})
	ab := Cat(a, b, 1)
	sol := EMCP(ab)
	fmt.Println(sol)
	// Output:
	// [1.333333333333333  1.0658141036401502e-16  -0.33333333333333326]
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
	// [8.642857142857142  -1.6071428571428568]
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
	// [8.642857142857142  -1.6071428571428568]
}

func ExampleFrobeniusNorm() {
	a := New1DArray([]float64{-4, -3, -2, -1, 0, 1, 2, 3, 4})
	b := FrobeniusNorm(a, -1)
	fmt.Println(b)
	// Output:
	// [7.745966692414834]
}

func ExampleFrobeniusNorm_col() {
	a := New2DArray([][]float64{{1, 2, 3}, {-1, 1, 4}})
	b := FrobeniusNorm(a, 0)
	fmt.Println(b)
	// Output:
	// [1.4142135623730951  2.23606797749979  5]
}

func ExampleFrobeniusNorm_row() {
	a := New2DArray([][]float64{{1, 2, 3}, {-1, 1, 4}})
	b := FrobeniusNorm(a, 1)
	fmt.Println(b)
	// Output:
	// ⎡3.7416573867739413⎤
	// ⎣ 4.242640687119285⎦
}
