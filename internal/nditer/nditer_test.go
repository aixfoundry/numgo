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
		func(d []float64, src [][]float64) {
			for i := range d {
				d[i] = src[0][i] + src[1][i]
			}
		})
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
		func(d []int, src [][]int) {
			for i := range d {
				d[i] = src[0][i] + src[1][i]
			}
		})
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

func TestAccumulateAxis0(t *testing.T) {
	// [[1 2 3]      [[1 2 3]
	//  [4 5 6]]  ->  [5 7 9]]
	data := []float64{1, 2, 3, 4, 5, 6}
	l := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	out := make([]float64, 6)
	Accumulate(l, data, []int{0}, out, func(acc *float64, v float64) { *acc += v })
	want := []float64{1, 2, 3, 5, 7, 9}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("got %v, want %v", out, want)
	}
}

func TestAccumulateAxis1(t *testing.T) {
	// [[1 2 3]      [[1 3 6]
	//  [4 5 6]]  ->  [4 9 15]]
	data := []float64{1, 2, 3, 4, 5, 6}
	l := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	out := make([]float64, 6)
	Accumulate(l, data, []int{1}, out, func(acc *float64, v float64) { *acc += v })
	want := []float64{1, 3, 6, 4, 9, 15}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("got %v, want %v", out, want)
	}
}

func TestAccumulateAxisNegative(t *testing.T) {
	// Axis -1 on a 2x3 must match axis 1.
	data := []float64{1, 2, 3, 4, 5, 6}
	l := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	out := make([]float64, 6)
	Accumulate(l, data, []int{-1}, out, func(acc *float64, v float64) { *acc += v })
	want := []float64{1, 3, 6, 4, 9, 15}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("got %v, want %v", out, want)
	}
}

func TestAccumulateAllAxes(t *testing.T) {
	// Folding over both axes prefixes the raveled data.
	data := []float64{1, 2, 3, 4, 5, 6}
	l := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	out := make([]float64, 6)
	Accumulate(l, data, []int{0, 1}, out, func(acc *float64, v float64) { *acc += v })
	want := []float64{1, 3, 6, 10, 15, 21}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("got %v, want %v", out, want)
	}
}

// --- contiguous fast-path tests ---

func TestZipFastPathCallbackBlocks(t *testing.T) {
	dst := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	out := make([]float64, 6)
	a := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	var dstLens, srcLens []int
	Zip(dst, out, []*Layout{a}, [][]float64{{1, 2, 3, 4, 5, 6}},
		func(d []float64, src [][]float64) {
			dstLens = append(dstLens, len(d))
			srcLens = append(srcLens, len(src[0]))
			for i := range d {
				d[i] = src[0][i] * 2
			}
		})
	// Fast path contract: a whole block (>= 2 elements) in ONE call.
	if len(dstLens) != 1 || dstLens[0] < 2 || srcLens[0] < 2 {
		t.Fatalf("fast path: want one whole-block call, got dstLens=%v srcLens=%v", dstLens, srcLens)
	}
	if !reflect.DeepEqual(out, []float64{2, 4, 6, 8, 10, 12}) {
		t.Fatalf("fast path result = %v", out)
	}

	// Strided src forces the per-element path: every block has length 1.
	out2 := make([]int, 6)
	aStrided := &Layout{Shape: []int{3, 2}, Strides: []int{1, 3}}
	dLens, sLens := 0, map[int]bool{}
	Zip(&Layout{Shape: []int{3, 2}, Strides: []int{2, 1}}, out2,
		[]*Layout{aStrided}, [][]int{{1, 2, 3, 4, 5, 6}},
		func(d []int, src [][]int) {
			dLens++
			sLens[len(src[0])] = true
			d[0] = src[0][0]
		})
	if dLens != 6 || !sLens[1] {
		t.Fatalf("strided path: want 6 per-element calls, got %d calls lens=%v", dLens, sLens)
	}
}

func TestZipFastPathBroadcastAndOffset(t *testing.T) {
	// Broadcast src (stride 0 on axis 0) must keep per-element semantics.
	dst := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	out := make([]int, 6)
	a := &Layout{Shape: []int{1, 3}, Strides: []int{0, 1}}
	Zip(dst, out, []*Layout{a}, [][]int{{1, 2, 3}},
		func(d []int, src [][]int) { d[0] = src[0][0] })
	if !reflect.DeepEqual(out, []int{1, 2, 3, 1, 2, 3}) {
		t.Fatalf("broadcast zip = %v", out)
	}

	// Offset contiguous data must read/write from the right base.
	off := &Layout{Shape: []int{2, 2}, Strides: []int{2, 1}, Offset: 2}
	outOff := make([]int, 6)
	dstOff := &Layout{Shape: []int{2, 2}, Strides: []int{2, 1}, Offset: 2}
	Zip(dstOff, outOff, []*Layout{off}, [][]int{{9, 9, 1, 2, 3, 4}},
		func(d []int, src [][]int) {
			for i := range d {
				d[i] = src[0][i]
			}
		})
	if !reflect.DeepEqual(outOff, []int{0, 0, 1, 2, 3, 4}) {
		t.Fatalf("offset zip = %v", outOff)
	}
}

func TestReduceFastPaths(t *testing.T) {
	// Whole-array reduce on a contiguous 2x3: linear fold into out[0].
	data := []float64{1, 2, 3, 4, 5, 6}
	l := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	out := []float64{0}
	calls := 0
	Reduce(l, data, []int{0, 1}, out, func(acc *float64, v float64) {
		*acc += v
		calls++
	})
	if out[0] != 21 || calls != 6 {
		t.Fatalf("whole reduce: out=%v calls=%d", out, calls)
	}

	// Whole-array reduce via negative axes.
	out2 := []float64{1}
	Reduce(l, data, []int{-2, -1}, out2, func(acc *float64, v float64) { *acc *= v })
	if out2[0] != 720 {
		t.Fatalf("whole reduce product = %v", out2)
	}

	// Last-axis reduce on contiguous input: per-row folds, slot = row.
	out3 := make([]float64, 2)
	Reduce(l, data, []int{-1}, out3, func(acc *float64, v float64) { *acc += v })
	if !reflect.DeepEqual(out3, []float64{6, 15}) {
		t.Fatalf("last-axis reduce = %v", out3)
	}

	// Same data on a strided layout (slow path) must still agree.
	lt := &Layout{Shape: []int{3, 2}, Strides: []int{1, 3}}
	out4 := make([]float64, 3)
	Reduce(lt, data, []int{1}, out4, func(acc *float64, v float64) { *acc += v })
	if !reflect.DeepEqual(out4, []float64{5, 7, 9}) {
		t.Fatalf("strided reduce = %v", out4)
	}

	// 1-D whole reduce.
	out5 := make([]float64, 1)
	Reduce(&Layout{Shape: []int{6}, Strides: []int{1}}, data, []int{0}, out5,
		func(acc *float64, v float64) { *acc += v })
	if out5[0] != 21 {
		t.Fatalf("1d reduce = %v", out5)
	}
}

func TestAccumulateFastPath(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	l := &Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}
	out := make([]float64, 6)
	Accumulate(l, data, []int{-1}, out, func(acc *float64, v float64) { *acc += v })
	if !reflect.DeepEqual(out, []float64{1, 3, 6, 4, 9, 15}) {
		t.Fatalf("last-axis accumulate = %v", out)
	}

	// Offset data: out is indexed by the same linear position as data.
	lo := &Layout{Shape: []int{2, 2}, Strides: []int{2, 1}, Offset: 2}
	do := []int{9, 9, 1, 2, 3, 4}
	outO := make([]int, 6)
	Accumulate(lo, do, []int{1}, outO, func(acc *int, v int) { *acc += v })
	if !reflect.DeepEqual(outO[2:], []int{1, 3, 3, 7}) {
		t.Fatalf("offset accumulate = %v", outO)
	}

	// Axis 0 on contiguous input stays on the general path and matches.
	out2 := make([]float64, 6)
	Accumulate(l, data, []int{0}, out2, func(acc *float64, v float64) { *acc += v })
	if !reflect.DeepEqual(out2, []float64{1, 2, 3, 5, 7, 9}) {
		t.Fatalf("axis0 accumulate = %v", out2)
	}
}

func TestContiguousSame(t *testing.T) {
	if !contiguousSame(&Layout{Shape: []int{2, 3}, Strides: []int{3, 1}}, []int{2, 3}) {
		t.Fatal("contiguous same shape should be true")
	}
	if contiguousSame(&Layout{Shape: []int{2, 3}, Strides: []int{1, 2}}, []int{2, 3}) {
		t.Fatal("wrong strides should be false")
	}
	if contiguousSame(&Layout{Shape: []int{3}, Strides: []int{1}}, []int{2, 3}) {
		t.Fatal("shape mismatch should be false")
	}
}
