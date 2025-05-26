package numgo

import (
	"fmt"
	"reflect"
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

func ExampleMinAbs_first() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := MinAbs(a)
	fmt.Println(sol, r, c)
	// Output:
	// 0 0 0
}

func ExampleMinAbs_same() {
	a := NewArray(1, 9, []float64{2, 2, 1, 1, 13, 5, 6, 7, 9})
	sol, r, c := MinAbs(a)
	fmt.Println(sol, r, c)
	// Output:
	// 1 0 2
}

func ExampleMinAbs_second() {
	a := NewArray(1, 9, []float64{2, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := MinAbs(a)
	fmt.Println(sol, r, c)
	// Output:
	// 1 0 1
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

func ExampleDot_x2d() {
	a := New2DArray([][]float64{{1, 2}, {3, 4}})
	b := New2DArray([][]float64{{2, 0}, {1, 2}})
	result := a.Dot(b)
	fmt.Println(result)
	// Output:
	// ⎡ 4  4⎤
	// ⎣10  8⎦
}

func ExampleSum() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	fmt.Println(Sum(a))
	// Output:
	// 46
}

func ExampleAvg() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	fmt.Println(Avg(a))
	// Output:
	// 5.111111111111111
}

func TestWindowAvg(t *testing.T) {
	type args struct {
		a      *Array
		window int
		step   int
		axios  int
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Full Window",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 9,
				step:   1,
				axios:  0,
			},
			wantRet: New1DArray([]float64{4}),
		},
		{
			name: "Test Full Window Step 2",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 9,
				step:   2,
				axios:  0,
			},
			wantRet: New1DArray([]float64{4}),
		},
		{
			name: "Test Window 3 Step 1",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 3,
				step:   1,
				axios:  0,
			},
			wantRet: New1DArray([]float64{1, 2, 3, 4, 5, 6, 7}),
		},
		{
			name: "Test Window 3 Step 2",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 3,
				step:   2,
				axios:  0,
			},
			wantRet: New1DArray([]float64{1, 3, 5, 7}),
		},
		{
			name: "Test Window 3 Step 100",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 3,
				step:   100,
				axios:  0,
			},
			wantRet: New1DArray([]float64{1}),
		},
		{
			name: "Test Window 0 Step 1",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 0,
				step:   1,
				axios:  0,
			},
			wantRet: New1DArray([]float64{4}),
		},
		{
			name: "Test Window 4 Step 1",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 4,
				step:   1,
				axios:  0,
			},
			wantRet: New1DArray([]float64{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}),
		},
		{
			name: "Test Window 4 Step 2",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 4,
				step:   2,
				axios:  0,
			},
			wantRet: New1DArray([]float64{1.5, 3.5, 5.5, 7}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := MovingAvg(tt.args.a, tt.args.window, tt.args.step, tt.args.axios); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("WindowAvg() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}
