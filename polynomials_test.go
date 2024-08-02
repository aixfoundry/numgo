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
		8.0, 2.0})
	sol := PolyFit(a, 2)
	fmt.Println(sol)
	// Output:
	// [0.9967721352741029  -0.00018136278480192529  0.020859993948697978]
}
