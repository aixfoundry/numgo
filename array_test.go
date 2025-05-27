package numgo

import (
	"fmt"
	"reflect"
	"slices"
	"testing"
)

func TestNewValueArray(t *testing.T) {
	data := []float64{
		12, 12, 12,
		12, 12, 12,
		12, 12, 12,
	}
	a := NewValueArray(3, 3, 12)
	b := NewArray(3, 3, data)
	if !reflect.DeepEqual(a, b) {
		t.Error(a)
		t.Error(b)
		t.FailNow()
	}
}

func TestArray_Data(t *testing.T) {
	data := []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	a := NewArray(3, 3, data)
	if !slices.Equal(data, a.Data()) {
		t.Error(data)
		t.Error(a.Data())
		t.FailNow()
	}
}

func TestNew2DArray(t *testing.T) {
	adata := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	bdata := []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	a := New2DArray(adata)
	b := NewArray(3, 3, bdata)
	if !slices.Equal(a.Data(), b.Data()) {
		t.Error(a.Data())
		t.Error(b.Data())
		t.FailNow()
	}
}

func ExampleArray_Of_col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Of(0, 0))
	// Output:
	// ⎡1⎤
	// ⎢4⎥
	// ⎣7⎦
}

func ExampleArray_Col() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Col(0))
	// Output:
	// ⎡1⎤
	// ⎢4⎥
	// ⎣7⎦
}

func ExampleArray_Col_d2() {
	a := New2DArray([][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})
	fmt.Printf("%v", a.Col(0))
	// Output:
	// ⎡1⎤
	// ⎢4⎥
	// ⎣7⎦
}

func ExampleArray_Of_row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Of(0, 1))
	// Output:
	// [1  2  3]
}

func ExampleArray_Row() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Row(0))
	// Output:
	// [1  2  3]
}

func ExampleArray_Row_d2() {
	a := New2DArray([][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})
	fmt.Printf("%v", a.Row(0))
	// Output:
	// [1  2  3]
}

func ExampleArray_Row_row1() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Row(1))
	// Output:
	// [4  5  6]
}

func ExampleArray_Dims() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	r, c := a.Dims()
	fmt.Printf("%v %v", r, c)
	// Output:
	// 3 3
}

func ExampleArray_At() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.At(1, 1))
	// Output:
	// 5
}

func ExampleArray_Set() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	fmt.Printf("%v", a.Set(1, 1, 11))
	// Output:
	// ⎡1   2  3⎤
	// ⎢4  11  6⎥
	// ⎣7   8  9⎦
}

func ExampleAll() {
	a := NewArray(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	for k, v := range All(a) {
		fmt.Printf("%v %v\n", k, v)
	}
	// Output:
	// [0 0] 1
	// [0 1] 2
	// [0 2] 3
	// [1 0] 4
	// [1 1] 5
	// [1 2] 6
	// [2 0] 7
	// [2 1] 8
	// [2 2] 9
}
