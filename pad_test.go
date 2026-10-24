package numgo

import (
	"math"
	"slices"
	"testing"
)

// pad1 is [1 2 3]; pad2x3 is [[1 2 3] [4 5 6]]. Goldens come from
// numpy.pad on numpy 2.5.1 with asymmetric widths.
func pad1() *Array[float64]   { return New1DArray([]float64{1, 2, 3}) }
func pad2x3() *Array[float64] { return FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3) }

func TestPadConstantEmpty(t *testing.T) {
	got := Pad(pad1(), [][2]int{{2, 3}}, PadConstant, 9)
	wantNear(t, got, []int{8}, []float64{9, 9, 1, 2, 3, 9, 9, 9})
	got2 := Pad(pad2x3(), [][2]int{{1, 2}, {2, 1}}, PadConstant, 0)
	wantNear(t, got2, []int{5, 6}, []float64{
		0, 0, 0, 0, 0, 0,
		0, 0, 1, 2, 3, 0,
		0, 0, 4, 5, 6, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0})
	// PadEmpty fills Go zeros (numpy's uninitialized memory can't be
	// reproduced).
	wantNear(t, Pad(pad1(), [][2]int{{2, 3}}, PadEmpty, 0), []int{8},
		[]float64{0, 0, 1, 2, 3, 0, 0, 0})
	// width-count mismatches and negatives panic.
	assertPanic(t, func() { Pad(pad1(), [][2]int{{1, 1}, {1, 1}}, PadConstant, 0) })
	assertPanic(t, func() { Pad(pad1(), [][2]int{{-1, 1}}, PadConstant, 0) })
}

func TestPadEdge(t *testing.T) {
	wantNear(t, Pad(pad1(), [][2]int{{2, 3}}, PadEdge, 0), []int{8},
		[]float64{1, 1, 1, 2, 3, 3, 3, 3})
	wantNear(t, Pad(pad2x3(), [][2]int{{1, 2}, {2, 1}}, PadEdge, 0), []int{5, 6}, []float64{
		1, 1, 1, 2, 3, 3,
		1, 1, 1, 2, 3, 3,
		4, 4, 4, 5, 6, 6,
		4, 4, 4, 5, 6, 6,
		4, 4, 4, 5, 6, 6})
}

func TestPadReflect(t *testing.T) {
	// np.pad([1,2,3], (2,3), 'reflect') = [3 2 1 2 3 2 1 2]
	wantNear(t, Pad(pad1(), [][2]int{{2, 3}}, PadReflect, 0), []int{8},
		[]float64{3, 2, 1, 2, 3, 2, 1, 2})
	// np.pad([1,2,3], (4,4), 'reflect') = [1 2 3 2 1 2 3 2 1 2 3]
	wantNear(t, Pad(pad1(), [][2]int{{4, 4}}, PadReflect, 0), []int{11},
		[]float64{1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3})
	// np.pad([1], (2,3), 'reflect') = all ones (period collapses).
	wantNear(t, Pad(New1DArray([]float64{1}), [][2]int{{2, 3}}, PadReflect, 0), []int{6},
		[]float64{1, 1, 1, 1, 1, 1})
	// numpy 2x3 asymmetric golden.
	wantNear(t, Pad(pad2x3(), [][2]int{{1, 2}, {2, 1}}, PadReflect, 0), []int{5, 6}, []float64{
		6, 5, 4, 5, 6, 5,
		3, 2, 1, 2, 3, 2,
		6, 5, 4, 5, 6, 5,
		3, 2, 1, 2, 3, 2,
		6, 5, 4, 5, 6, 5})
}

func TestPadSymmetric(t *testing.T) {
	// np.pad([1,2,3], (2,3), 'symmetric') = [2 1 1 2 3 4 5 5]
	wantNear(t, Pad(pad1(), [][2]int{{2, 3}}, PadSymmetric, 0), []int{8},
		[]float64{2, 1, 1, 2, 3, 3, 2, 1})
	// np.pad([1,2,3], (4,4), 'symmetric')
	wantNear(t, Pad(pad1(), [][2]int{{4, 4}}, PadSymmetric, 0), []int{11},
		[]float64{3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1})
	// numpy 2x3 asymmetric golden.
	wantNear(t, Pad(pad2x3(), [][2]int{{1, 2}, {2, 1}}, PadSymmetric, 0), []int{5, 6}, []float64{
		2, 1, 1, 2, 3, 3,
		2, 1, 1, 2, 3, 3,
		5, 4, 4, 5, 6, 6,
		5, 4, 4, 5, 6, 6,
		2, 1, 1, 2, 3, 3})
}

func TestPadWrap(t *testing.T) {
	// np.pad([1,2,3], (2,3), 'wrap') = [2 3 1 2 3 1 2 3]
	wantNear(t, Pad(pad1(), [][2]int{{2, 3}}, PadWrap, 0), []int{8},
		[]float64{2, 3, 1, 2, 3, 1, 2, 3})
	// np.pad([1,2,3], (4,0), 'wrap') = [3 1 2 3 1 2 3]
	wantNear(t, Pad(pad1(), [][2]int{{4, 0}}, PadWrap, 0), []int{7},
		[]float64{3, 1, 2, 3, 1, 2, 3})
	// numpy 2x3 asymmetric golden.
	wantNear(t, Pad(pad2x3(), [][2]int{{1, 2}, {2, 1}}, PadWrap, 0), []int{5, 6}, []float64{
		5, 6, 4, 5, 6, 4,
		2, 3, 1, 2, 3, 1,
		5, 6, 4, 5, 6, 4,
		2, 3, 1, 2, 3, 1,
		5, 6, 4, 5, 6, 4})
}

func TestPadFloat32(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3}, 3)
	got := Pad(a, [][2]int{{1, 2}}, PadReflect, 0)
	if !slices.Equal(got.Shape(), []int{6}) {
		t.Fatalf("shape = %v", got.Shape())
	}
	want := []float64{2, 1, 2, 3, 2, 1}
	for i, w := range want {
		if math.Abs(float64(got.Data()[i])-w) > 0 {
			t.Fatalf("data[%d] = %v, want %v", i, got.Data()[i], w)
		}
	}
}
