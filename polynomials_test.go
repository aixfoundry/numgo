package numgo

import "fmt"

func ExamplePolyFit() {
	a := NewArray(7, 2, []float64{
		1.0, 10.0,
		3.0, 5.0,
		4.0, 4.0,
		5.0, 2.0,
		6.0, 1.0,
		7.0, 1.0,
		8.0, 2.0,
	})
	sol := PolyFit(a, 2)
	fmt.Println(sol)
	// Output:
	// [13.445100354191204  -3.585005903187692  0.26387249114521544]
}

func ExamplePoly1D() {
	a := NewArray(7, 2, []float64{
		1.0, 2.0,
		3.0, 6.0,
		4.0, 8.0,
		5.0, 10.0,
		6.0, 12.0,
		7.0, 14.0,
		8.0, 16.0,
	})
	sol := PolyFit(a, 1)
	fmt.Println(sol)
	y := Poly1D(sol, 1.0)
	fmt.Println(y)
	// Output:
	// [0  2]
	// 2
}

func ExamplePolyVal() {
	a := NewArray(7, 2, []float64{
		1.0, 2.0,
		3.0, 6.0,
		4.0, 8.0,
		5.0, 10.0,
		6.0, 12.0,
		7.0, 14.0,
		8.0, 16.0,
	})
	sol := PolyFit(a, 1)
	fmt.Println(sol)
	b := NewArray(7, 1, []float64{
		9,
		10,
		11,
		12,
		13,
		14,
		15,
	})
	y := PolyVal(sol, b)
	fmt.Println(y)
	// Output:
	// [0  2]
	// ⎡18⎤
	// ⎢20⎥
	// ⎢22⎥
	// ⎢24⎥
	// ⎢26⎥
	// ⎢28⎥
	// ⎣30⎦
}
