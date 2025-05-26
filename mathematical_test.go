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
	a := NewArray(1, 9, []float64{0, 1, 2, 3, -13, 5, 6, 7, 9})
	sol, r, c := a.MaxAbs()
	fmt.Println(sol, r, c)
	// Output:
	// 13 0 4
}

func ExampleArray_Abs() {
	a := NewArray(1, 4, []float64{-1.0, 2.0, -3.0, 4.0})
	sol := Abs(a)
	fmt.Println(sol)
	// Output:
	// [1  2  3  4]
}

func ExampleMaxAbs() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, -13, 5, 6, 7, 9})
	sol, r, c := MaxAbs(a)
	fmt.Println(sol, r, c)
	// Output:
	// 13 0 4
}

func ExampleArray_MinAbs() {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, -13, 5, 6, 7, 9})
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
	fmt.Println(Sum(a, -1))
	// Output:
	// [46]
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
		axis   int
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
				axis:   0,
			},
			wantRet: New1DArray([]float64{4}),
		},
		{
			name: "Test Full Window Step 2",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 9,
				step:   2,
				axis:   0,
			},
			wantRet: New1DArray([]float64{4}),
		},
		{
			name: "Test Window 3 Step 1",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 3,
				step:   1,
				axis:   0,
			},
			wantRet: New1DArray([]float64{1, 2, 3, 4, 5, 6, 7}),
		},
		{
			name: "Test Window 3 Step 2",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 3,
				step:   2,
				axis:   0,
			},
			wantRet: New1DArray([]float64{1, 3, 5, 7}),
		},
		{
			name: "Test Window 3 Step 100",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 3,
				step:   100,
				axis:   0,
			},
			wantRet: New1DArray([]float64{1}),
		},
		{
			name: "Test Window 0 Step 1",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 0,
				step:   1,
				axis:   0,
			},
			wantRet: New1DArray([]float64{4}),
		},
		{
			name: "Test Window 4 Step 1",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 4,
				step:   1,
				axis:   0,
			},
			wantRet: New1DArray([]float64{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}),
		},
		{
			name: "Test Window 4 Step 2",
			args: args{
				a:      New1DArray([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}),
				window: 4,
				step:   2,
				axis:   0,
			},
			wantRet: New1DArray([]float64{1.5, 3.5, 5.5, 7}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := MovingAvg(tt.args.a, tt.args.window, tt.args.step, tt.args.axis); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("WindowAvg() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func TestMean(t *testing.T) {
	type args struct {
		a    *Array
		axis int
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Mean Axis 0",
			args: args{
				a:    New2DArray([][]float64{{1, 2}, {3, 4}}),
				axis: 0,
			},
			wantRet: New1DArray([]float64{2, 3}),
		},
		{
			name: "Test Mean Axis 1",
			args: args{
				a:    New2DArray([][]float64{{1, 2}, {3, 4}}),
				axis: 1,
			},
			wantRet: New1DArray([]float64{1.5, 3.5}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := Mean(tt.args.a, tt.args.axis); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("Mean() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func TestDiv(t *testing.T) {
	type args struct {
		a *Array
		b *Array
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Div 2x2 2x2",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New2DArray([][]float64{{1, 1}, {1, 1}}),
			},
			wantRet: New2DArray([][]float64{{1, 2}, {3, 4}}),
		},
		{
			name: "Test Div 2x2 1x2",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New1DArray([]float64{1, 1}),
			},
			wantRet: New2DArray([][]float64{{1, 2}, {3, 4}}),
		},
		{
			name: "Test Div 2x2 2x1",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: NewArray(2, 1, []float64{1, 1}),
			},
			wantRet: New2DArray([][]float64{{1, 2}, {3, 4}}),
		},
		{
			name: "Test Div 2x2 1x1",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New1DArray([]float64{1}),
			},
			wantRet: New2DArray([][]float64{{1, 2}, {3, 4}}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := Div(tt.args.a, tt.args.b); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("Div() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func TestMul(t *testing.T) {
	type args struct {
		a *Array
		b *Array
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Mul 2x2 2x2",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New2DArray([][]float64{{1, 1}, {1, 1}}),
			},
			wantRet: New2DArray([][]float64{{1, 2}, {3, 4}}),
		},
		{
			name: "Test Mul 2x2 1x2",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New1DArray([]float64{1, 1}),
			},
			wantRet: New2DArray([][]float64{{1, 2}, {3, 4}}),
		},
		{
			name: "Test Mul 2x2 2x1",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: NewArray(2, 1, []float64{1, 1}),
			},
			wantRet: New2DArray([][]float64{{1, 2}, {3, 4}}),
		},
		{
			name: "Test Mul 2x2 1x1",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New1DArray([]float64{1}),
			},
			wantRet: New2DArray([][]float64{{1, 2}, {3, 4}}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := Mul(tt.args.a, tt.args.b); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("Mul() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func TestAdd(t *testing.T) {
	type args struct {
		a *Array
		b *Array
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Add 2x2 2x2",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New2DArray([][]float64{{1, 1}, {1, 1}}),
			},
			wantRet: New2DArray([][]float64{{2, 3}, {4, 5}}),
		},
		{
			name: "Test Add 2x2 1x2",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New1DArray([]float64{1, 1}),
			},
			wantRet: New2DArray([][]float64{{2, 3}, {4, 5}}),
		},
		{
			name: "Test Add 2x2 2x1",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: NewArray(2, 1, []float64{1, 1}),
			},
			wantRet: New2DArray([][]float64{{2, 3}, {4, 5}}),
		},
		{
			name: "Test Add 2x2 1x1",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New1DArray([]float64{1}),
			},
			wantRet: New2DArray([][]float64{{2, 3}, {4, 5}}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := Add(tt.args.a, tt.args.b); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("Add() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func TestSub(t *testing.T) {
	type args struct {
		a *Array
		b *Array
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Sub 2x2 2x2",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New2DArray([][]float64{{1, 1}, {1, 1}}),
			},
			wantRet: New2DArray([][]float64{{0, 1}, {2, 3}}),
		},
		{
			name: "Test Sub 2x2 1x2",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New1DArray([]float64{1, 1}),
			},
			wantRet: New2DArray([][]float64{{0, 1}, {2, 3}}),
		},
		{
			name: "Test Sub 2x2 2x1",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: NewArray(2, 1, []float64{1, 1}),
			},
			wantRet: New2DArray([][]float64{{0, 1}, {2, 3}}),
		},
		{
			name: "Test Sub 2x2 1x1",
			args: args{
				a: New2DArray([][]float64{{1, 2}, {3, 4}}),
				b: New1DArray([]float64{1}),
			},
			wantRet: New2DArray([][]float64{{0, 1}, {2, 3}}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := Sub(tt.args.a, tt.args.b); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("Sub() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func TestAddValue(t *testing.T) {
	type args struct {
		a     *Array
		value float64
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Add 2x2",
			args: args{
				a:     New2DArray([][]float64{{1, 2}, {3, 4}}),
				value: 2,
			},
			wantRet: New2DArray([][]float64{{3, 4}, {5, 6}}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := AddValue(tt.args.a, tt.args.value); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("AddValue() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func TestSubValue(t *testing.T) {
	type args struct {
		a     *Array
		value float64
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Sub 2x2",
			args: args{
				a:     New2DArray([][]float64{{1, 2}, {3, 4}}),
				value: 2,
			},
			wantRet: New2DArray([][]float64{{-1, 0}, {1, 2}}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := SubValue(tt.args.a, tt.args.value); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("SubValue() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func TestMulValue(t *testing.T) {
	type args struct {
		a     *Array
		value float64
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Mul 2x2",
			args: args{
				a:     New2DArray([][]float64{{1, 2}, {3, 4}}),
				value: 2,
			},
			wantRet: New2DArray([][]float64{{2, 4}, {6, 8}}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := MulValue(tt.args.a, tt.args.value); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("MulValue() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func TestDivValue(t *testing.T) {
	type args struct {
		a     *Array
		value float64
	}
	tests := []struct {
		name    string
		args    args
		wantRet *Array
	}{
		{
			name: "Test Div 2x2",
			args: args{
				a:     New2DArray([][]float64{{1, 2}, {3, 4}}),
				value: 2,
			},
			wantRet: New2DArray([][]float64{{0.5, 1}, {1.5, 2}}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := DivValue(tt.args.a, tt.args.value); !reflect.DeepEqual(gotRet, tt.wantRet) {
				t.Errorf("DivValue() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func ExampleAtan2() {
	y := New2DArray([][]float64{{1, 2}, {3, 4}})
	x := New2DArray([][]float64{{1, 1}, {1, 1}})
	result := Atan2(y, x)
	fmt.Println(result)
	// Output:
	// ⎡0.7853981633974483  1.1071487177940904⎤
	// ⎣1.2490457723982544  1.3258176636680323⎦
}

func ExampleMedian_col() {
	a := New2DArray([][]float64{{1, 3, 5}, {2, 4, 6}})
	sorted := Median(a, 0)
	fmt.Println(sorted)
	// Output:
	// [1.5  3.5  5.5]
}

func ExampleMedian_row() {
	a := New2DArray([][]float64{{1, 3, 5}, {2, 4, 6}})
	sorted := Median(a, 1)
	fmt.Println(sorted)
	// Output:
	// ⎡3⎤
	// ⎣4⎦
}

func ExampleMedian_all() {
	a := New2DArray([][]float64{{1, 3, 5}, {2, 4, 6}})
	sorted := Median(a, -1)
	fmt.Println(sorted)
	// Output:
	// [3.5]
}

func ExampleSort_all() {
	a := New2DArray([][]float64{{3, 1, 5}, {6, 4, 2}})
	sorted := Sort(a, -1)
	fmt.Println(sorted)
	// Output:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
}

func ExampleSort_row() {
	a := New2DArray([][]float64{{3, 1, 5}, {6, 4, 2}})
	sorted := Sort(a, 1)
	fmt.Println(sorted)
	// Output:
	// ⎡1  3  5⎤
	// ⎣2  4  6⎦
}

func ExampleSort_col() {
	a := New2DArray([][]float64{{6, 1, 2}, {3, 4, 5}})
	sorted := Sort(a, 1)
	fmt.Println(sorted)
	// Output:
	// ⎡1  2  6⎤
	// ⎣3  4  5⎦
}
