package nditer

import (
	"reflect"
	"testing"
)

func TestCOrderStrides(t *testing.T) {
	cases := []struct {
		shape, want []int
	}{
		{[]int{2, 3}, []int{3, 1}},
		{[]int{2, 3, 4}, []int{12, 4, 1}},
		{[]int{5}, []int{1}},
	}
	for _, c := range cases {
		if got := COrderStrides(c.shape); !reflect.DeepEqual(got, c.want) {
			t.Errorf("COrderStrides(%v) = %v, want %v", c.shape, got, c.want)
		}
	}
}

func TestSize(t *testing.T) {
	if got := Size([]int{2, 3, 4}); got != 24 {
		t.Errorf("Size(2,3,4) = %d, want 24", got)
	}
	if got := Size(nil); got != 0 {
		t.Errorf("Size(nil) = %d, want 0", got)
	}
}

func TestBroadcast(t *testing.T) {
	cases := []struct {
		shapes  [][]int
		want    []int
		strides [][]int
	}{
		{
			shapes:  [][]int{{2, 3}, {3}},
			want:    []int{2, 3},
			strides: [][]int{{3, 1}, {0, 1}},
		},
		{
			shapes:  [][]int{{2, 1}, {1, 3}},
			want:    []int{2, 3},
			strides: [][]int{{1, 0}, {0, 1}},
		},
		{
			shapes:  [][]int{{2, 1, 4}, {3, 1}},
			want:    []int{2, 3, 4},
			strides: [][]int{{4, 0, 1}, {0, 1, 0}},
		},
	}
	for _, c := range cases {
		gotShape, gotStrides, err := Broadcast(c.shapes...)
		if err != nil {
			t.Fatalf("Broadcast(%v): %v", c.shapes, err)
		}
		if !reflect.DeepEqual(gotShape, c.want) {
			t.Errorf("Broadcast(%v) shape = %v, want %v", c.shapes, gotShape, c.want)
		}
		if !reflect.DeepEqual(gotStrides, c.strides) {
			t.Errorf("Broadcast(%v) strides = %v, want %v", c.shapes, gotStrides, c.strides)
		}
	}
}

func TestBroadcastMismatch(t *testing.T) {
	_, _, err := Broadcast([]int{5, 4, 3}, []int{2, 3})
	want := "numgo: operands could not be broadcast together with shapes (5,4,3) (2,3)"
	if err == nil || err.Error() != want {
		t.Fatalf("err = %v, want %q", err, want)
	}
}

func TestZip2D(t *testing.T) {
	dst := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	out := make([]float64, 6)
	a := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	b := &Layout{Shape: []int{3}, Strides: []int{0, 1}}
	Zip(dst, out, []*Layout{a, b}, [][]float64{{0, 1, 2, 3, 4, 5}, {10, 20, 30}},
		func(d []float64, src [][]float64) { d[0] = src[0][0] + src[1][0] })
	want := []float64{10, 21, 32, 13, 24, 35}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("got %v, want %v", out, want)
	}
}

func TestZip3D(t *testing.T) {
	dst := &Layout{Shape: []int{2, 3, 4}, Strides: []int{12, 4, 1}}
	out := make([]int, 24)
	a := &Layout{Shape: []int{2, 3, 4}, Strides: []int{12, 4, 1}}
	b := &Layout{Shape: []int{4}, Strides: []int{0, 0, 1}}
	adata := make([]int, 24)
	for i := range adata {
		adata[i] = i
	}
	Zip(dst, out, []*Layout{a, b}, [][]int{adata, {10, 20, 30, 40}},
		func(d []int, src [][]int) { d[0] = src[0][0] + src[1][0] })
	want := []int{
		10, 21, 32, 43,
		14, 25, 36, 47,
		18, 29, 40, 51,
		22, 33, 44, 55,
		26, 37, 48, 59,
		30, 41, 52, 63,
	}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("got %v, want %v", out, want)
	}
}

func TestZipStridedSrc(t *testing.T) {
	// 2x3 data viewed transposed (3x2) with element strides [1,3].
	dst := &Layout{Shape: []int{3, 2}, Strides: []int{2, 1}}
	out := make([]int, 6)
	a := &Layout{Shape: []int{3, 2}, Strides: []int{1, 3}}
	b := &Layout{Shape: []int{3, 2}, Strides: []int{2, 1}}
	Zip(dst, out, []*Layout{a, b}, [][]int{{1, 2, 3, 4, 5, 6}, {10, 20, 30, 40, 50, 60}},
		func(d []int, src [][]int) { d[0] = src[0][0] + src[1][0] })
	want := []int{11, 24, 32, 45, 53, 66}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("got %v, want %v", out, want)
	}
}

func TestReduce(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	l := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	sum := func(acc *float64, v float64) { *acc += v }

	cases := []struct {
		name string
		axes []int
		want []float64
	}{
		{"axis0", []int{0}, []float64{5, 7, 9}},
		{"axis1", []int{1}, []float64{6, 15}},
		{"axis-1", []int{-1}, []float64{6, 15}},
		{"full", []int{0, 1}, []float64{21}},
	}
	for _, c := range cases {
		out := make([]float64, len(c.want))
		Reduce(l, data, c.axes, out, sum)
		if !reflect.DeepEqual(out, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, out, c.want)
		}
	}
}
