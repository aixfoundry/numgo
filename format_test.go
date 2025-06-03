package numgo

import "fmt"

func ExampleArray_Format() {
	a := NewArray(3, 3, []float64{
		0, 0.2, 0.3,
		0.4001, 0.5, 0.6,
		0.7, 0.8, 0.9001,
	})
	fmt.Printf("%f", a)
	// Output:
	// ⎡     0  0.2     0.3⎤
	// ⎢0.4001  0.5     0.6⎥
	// ⎣   0.7  0.8  0.9001⎦
}

func ExampleArray_Format_left() {
	a := NewArray(3, 3, []float64{
		0, 0.2, 0.3,
		0.4001, 0.5, 0.6,
		0.7, 0.8, 0.9001,
	})
	fmt.Printf("%-f", a)
	// Output:
	// ⎡0       0.2  0.3   ⎤
	// ⎢0.4001  0.5  0.6   ⎥
	// ⎣0.7     0.8  0.9001⎦
}

func ExampleArray_Format_v() {
	a := NewArray(3, 3, []float64{
		0, 0.2, 0.3,
		0.4001, 0.5, 0.6,
		0.7, 0.8, 0.9001,
	})
	fmt.Printf("%v", a)
	// Output:
	// ⎡     0  0.2     0.3⎤
	// ⎢0.4001  0.5     0.6⎥
	// ⎣   0.7  0.8  0.9001⎦
}

func ExampleArray_Format_p2() {
	a := NewArray(3, 3, []float64{
		0, 0.2, 0.3,
		0.4001, 0.5, 0.6,
		0.7, 0.8, 0.9001,
	})
	fmt.Printf("%02.2f", a)
	// Output:
	// ⎡0.00  0.20  0.30⎤
	// ⎢0.40  0.50  0.60⎥
	// ⎣0.70  0.80  0.90⎦
}

func ExampleArray_Format_w10p2() {
	a := NewArray(3, 3, []float64{
		0, 0.2, 0.3,
		0.4001, 0.5, 0.6,
		0.7, 0.8, 0.9001,
	})
	fmt.Printf("%010.2g", a)
	// Output:
	// ⎡         0         0.2         0.3⎤
	// ⎢       0.4         0.5         0.6⎥
	// ⎣       0.7         0.8         0.9⎦
}
