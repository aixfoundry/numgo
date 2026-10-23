package numgo

import (
	"fmt"
	"math"
	"strings"
	"testing"
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

func ExampleNorm_row() {
	a := New1DArray([]float64{-4, -3, -2, -1, 0, 1, 2, 3, 4})
	b := Norm(a, 1, 1)
	fmt.Println(b)
	// Output:
	// [20]
}

func ExampleNorm_col() {
	a := New1DArray([]float64{-4, -3, -2, -1, 0, 1, 2, 3, 4})
	b := Norm(a, math.Inf(1), 1)
	fmt.Println(b)
	// Output:
	// [4]
}

func ExampleNormInf() {
	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
	norm := NormInf(a, -1) // norm will be a 1D array with value [15]
	fmt.Println(norm)
	// Output:
	// [15]
}

func ExampleNormInf_row() {
	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
	norm := NormInf(a, 1)
	fmt.Println(norm)
	// Output:
	// ⎡3⎤
	// ⎣6⎦
}

func ExampleNormInf_col() {
	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
	norm := NormInf(a, 0)
	fmt.Println(norm)
	// Output:
	// [4  5  6]
}

func ExampleNorm1() {
	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
	norm := Norm1(a, -1) // norm will be a 1D array with value [9]
	fmt.Println(norm)
	// Output:
	// [9]
}
func ExampleNorm1_row() {
	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
	norm := Norm1(a, 1) // norm will be a 1D array with value [9]
	fmt.Println(norm)
	// Output:
	// ⎡ 6⎤
	// ⎣15⎦
}

func ExampleNorm1_col() {
	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
	norm := Norm1(a, 0) // norm will be a 1D array with value [9]
	fmt.Println(norm)
	// Output:
	// [5  7  9]
}

func ExampleNorm2() {
	a := New2DArray([][]float64{{1, -2, 3}, {-4, 5, -6}})
	norm := Norm2(a, -1)
	fmt.Println(norm)
	// Output:
	// [9.539392014169456]
}

func ExampleHistogram2d() {
	x := New1DArray([]float64{1, 2, 2.5, 3, 3.5, 4})
	y := New1DArray([]float64{2, 3, 3.5, 4, 4.5, 5})
	bins := []int{3, 3}
	hist, xEdges, yEdges := Histogram2d(x, y, bins, [][]float64{})
	fmt.Println(hist)
	fmt.Println(xEdges)
	fmt.Println(yEdges)
	// Output:
	// ⎡1  0  0⎤
	// ⎢0  2  0⎥
	// ⎣0  0  3⎦
	// [1  2  3  4]
	// [2  3  4  5]
}

func TestLinearAlgebraPanics(t *testing.T) {
	cases := []struct {
		name string
		fn   func()
		want string
	}{
		{
			name: "EMCPSingular",
			fn: func() {
				// [[1, 2], [2, 4]] x = [3, 7]: dependent columns
				EMCP(NewArray(2, 3, []float64{1, 2, 3, 2, 4, 7}))
			},
			want: "Singular matrix",
		},
		{
			name: "EMCPShape",
			fn:   func() { EMCP(NewArray(2, 2, []float64{1, 2, 3, 4})) },
			want: "shape mismatch",
		},
		{
			name: "LstsqShape",
			fn: func() {
				Lstsq(NewArray(2, 1, []float64{1, 2}), NewArray(3, 1, []float64{1, 2, 3}))
			},
			want: "shape mismatch",
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				msg := fmt.Sprint(recover())
				if !strings.Contains(msg, tt.want) {
					t.Errorf("panic = %q, want it to contain %q", msg, tt.want)
				}
			}()
			tt.fn()
		})
	}
}
