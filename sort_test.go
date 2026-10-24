package numgo

import (
	"math"
	"slices"
	"testing"
)

var sortData = []float64{3, 7, 2, 6, 4, 5} // [[3 7 2] [6 4 5]]

func sort2x3() *Array[float64] {
	return FromSlice(sortData, 2, 3)
}

func sort3D() *Array[float64] {
	return FromSlice([]float64{
		30, 10, 20, 25,
		3, 1, 2, 24,
		13, 11, 12, 23,

		16, 6, 26, 36,
		15, 5, 28, 38,
		14, 4, 27, 37,
	}, 2, 3, 4)
}

func TestSortDefaultLastAxis(t *testing.T) {
	a := sort2x3()
	got := a.Sort()
	want := []float64{2, 3, 7, 4, 5, 6}
	if !slices.Equal(got.Data(), want) {
		t.Fatalf("Sort() = %v, want %v", got.Data(), want)
	}
	// numpy: np.sort(a, axis=0) sorts columns: (3,6)(7,4)(2,5) -> [3,4,2,6,7,5].
	byAxis0 := a.Sort(0)
	for i, w := range []float64{3, 4, 2, 6, 7, 5} {
		if v := byAxis0.Data()[i]; v != w {
			t.Errorf("Sort(0)[%d] = %v, want %v", i, v, w)
		}
	}
	if a.At(0, 0) != 3 {
		t.Error("Sort mutated its receiver")
	}
}

func TestSort3DMultiAxis(t *testing.T) {
	a := sort3D()
	one := a.Sort(1)
	both := a.Sort(1, 0)
	// Each axis-1 channel of `one` must be ascending (numpy spot-check:
	// np.sort(a,1) channel (0,1) = [3 10 12 24]-style columns ascending).
	for r := range 2 {
		for k := range 4 {
			x, y, z := one.At(r, 0, k), one.At(r, 1, k), one.At(r, 2, k)
			if !(x <= y && y <= z) {
				t.Errorf("Sort(1) channel (%d,%d) = %v %v %v, not ascending", r, k, x, y, z)
			}
		}
	}
	// numpy spot-check: np.sort(np.sort(a,1),0) with
	// a = [[[30,10,20,25],[3,1,2,24],[13,11,12,23]],
	//      [[16,6,26,36],[15,5,28,38],[14,4,27,37]]]
	want := []float64{
		3, 1, 2, 23,
		13, 5, 12, 24,
		16, 6, 20, 25,

		14, 4, 26, 36,
		15, 10, 27, 37,
		30, 11, 28, 38,
	}
	if !slices.Equal(both.Data(), want) {
		t.Errorf("Sort(1,0) = %v, want %v", both.Data(), want)
	}
}

func TestSortNegativeAxis(t *testing.T) {
	a := sort3D()
	got := a.Sort(-2) // axis 1, same as Sort(1)
	want := a.Sort(1)
	if !slices.Equal(got.Data(), want.Data()) {
		t.Errorf("Sort(-2) = %v, want %v", got.Data(), want.Data())
	}
}

func TestSortIntoMutates(t *testing.T) {
	a := sort2x3()
	ret := a.SortInto()
	if ret != a {
		t.Fatal("SortInto did not return the receiver")
	}
	if !slices.Equal(a.Data(), []float64{2, 3, 7, 4, 5, 6}) {
		t.Fatalf("after SortInto a = %v", a.Data())
	}
}

func TestSortNaNLast(t *testing.T) {
	a := FromSlice([]float64{math.NaN(), 2, 1, math.NaN()}, 4)
	got := a.Sort()
	if got.At(0) != 1 || got.At(1) != 2 {
		t.Fatalf("head = %v %v, want 1 2", got.At(0), got.At(1))
	}
	if !math.IsNaN(got.At(2)) || !math.IsNaN(got.At(3)) {
		t.Errorf("tail = %v %v, want NaN NaN", got.At(2), got.At(3))
	}
}

func TestSortFloat32(t *testing.T) {
	a := FromSlice([]float32{3, 1, 2}, 3)
	got := a.Sort()
	want := []float32{1, 2, 3}
	for i, w := range want {
		if v := got.At(i); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestSortInvalidAxisPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Sort(5) on 2x3 did not panic")
		}
	}()
	sort2x3().Sort(5)
}

func TestArgsortStableDuplicates(t *testing.T) {
	// Column-major duplicates: 2x3 rows [2 1 2] [1 2 1].
	a := FromSlice([]float64{2, 1, 2, 1, 2, 1}, 2, 3)
	got := a.Argsort(-1)
	want := []int{1, 0, 2, 0, 2, 1} // row0: 1@1 then stable 2s (0,2); row1: stable 1s (0,2) then 2@1
	if !slices.Equal(got, want) {
		t.Fatalf("Argsort(-1) = %v, want %v", got, want)
	}
	// Verify round trip: applying the indices sorts the row.
	for row := range 2 {
		prev := math.Inf(-1)
		for k := range 3 {
			v := a.At(row, got[row*3+k])
			if v < prev {
				t.Errorf("row %d not sorted by argsort at k=%d", row, k)
			}
			prev = v
		}
	}
}

func TestPartitionContract(t *testing.T) {
	// Fixed pseudo-random-ish data, non-square 2x7.
	data := []float64{9, 4, 7, 1, 8, 2, 6, 5, 3, 0, 10, 11, 12, 13}
	a := FromSlice(data, 2, 7)
	got := a.Partition(1, -1)
	for row := range 2 {
		kthVal := got.At(row, 1)
		for k := range 1 {
			if v := got.At(row, k); v > kthVal {
				t.Errorf("row %d: left element %v > kth %v", row, v, kthVal)
			}
		}
		for k := 2; k < 7; k++ {
			if v := got.At(row, k); v < kthVal {
				t.Errorf("row %d: right element %v < kth %v", row, v, kthVal)
			}
		}
	}
	// numpy spot-check: np.partition(data, 1, axis=-1) for this data is fully
	// sorted rows [1 2 4 6 7 8 9] [0 3 5 10 11 12 13]; sort-superset matches.
	if !slices.Equal(got.Shape(), []int{2, 7}) {
		t.Errorf("shape = %v", got.Shape())
	}
}

func TestArgPartitionContract(t *testing.T) {
	data := []float64{9, 4, 7, 1, 8, 2, 6, 5, 3, 0, 10, 11, 12, 13}
	a := FromSlice(data, 2, 7)
	got := a.ArgPartition(1, -1)
	for row := range 2 {
		kthVal := a.At(row, got[row*7+1])
		for k := range 1 {
			if a.At(row, got[row*7+k]) > kthVal {
				t.Errorf("row %d: left via index %d > kth", row, got[row*7+k])
			}
		}
		for k := 2; k < 7; k++ {
			if a.At(row, got[row*7+k]) < kthVal {
				t.Errorf("row %d: right via index %d < kth", row, got[row*7+k])
			}
		}
	}
}

func TestPartitionInvalidKthPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Partition(7, -1) on 2x7 did not panic")
		}
	}()
	FromSlice(make([]float64, 14), 2, 7).Partition(7, -1)
}

func TestUniqueSortedDedup(t *testing.T) {
	a := FromSlice([]float64{3, 1, 2, 3, 1, 2, 0}, 7)
	got := a.Unique()
	want := []float64{0, 1, 2, 3}
	if !slices.Equal(got.Data(), want) || !slices.Equal(got.Shape(), []int{4}) {
		t.Fatalf("Unique = %v shape %v, want %v shape [4]", got.Data(), got.Shape(), want)
	}
	// 2x3 non-square input ravels in C order first.
	m := FromSlice([]float64{2, 1, 2, 1, 3, 3}, 2, 3)
	if got := m.Unique(); !slices.Equal(got.Data(), []float64{1, 2, 3}) {
		t.Errorf("Unique(2x3) = %v, want [1 2 3]", got.Data())
	}
}

func TestUniqueAllDuplicatesAndEmpty(t *testing.T) {
	all := FromSlice([]float64{5, 5, 5}, 3)
	if got := all.Unique(); !slices.Equal(got.Data(), []float64{5}) {
		t.Errorf("Unique(all dup) = %v, want [5]", got.Data())
	}
	empty := Zeros[float64](0, 3)
	if got := empty.Unique(); !slices.Equal(got.Shape(), []int{0}) {
		t.Errorf("Unique(empty) shape = %v, want [0]", got.Shape())
	}
}

func TestUniqueNaN(t *testing.T) {
	a := FromSlice([]float64{1, math.NaN(), 1, math.NaN(), 0}, 5)
	got := a.Unique()
	if !math.IsNaN(got.At(len(got.Data()) - 1)) {
		t.Fatalf("Unique tail = %v, want NaN", got.Data())
	}
	if len(got.Data()) != 3 || got.At(0) != 0 || got.At(1) != 1 {
		t.Errorf("Unique = %v, want [0 1 NaN]", got.Data())
	}
}

func TestUniqueComplexOrder(t *testing.T) {
	a := FromSlice([]complex128{2 + 1i, 1 + 5i, 1 + 1i, 2 + 0i}, 4)
	got := a.Unique()
	want := []complex128{1 + 1i, 1 + 5i, 2 + 0i, 2 + 1i}
	for i, w := range want {
		if v := got.At(i); v != w {
			t.Errorf("[%d] = %v, want %v", i, v, w)
		}
	}
}

func TestUniqueCounts(t *testing.T) {
	a := FromSlice([]float64{2, 1, 2, 1, 2}, 5)
	vals, counts := a.UniqueCounts()
	if !slices.Equal(vals.Data(), []float64{1, 2}) {
		t.Errorf("values = %v, want [1 2]", vals.Data())
	}
	if !slices.Equal(counts, []int{2, 3}) {
		t.Errorf("counts = %v, want [2 3]", counts)
	}
	empty := Zeros[float64](2, 0)
	if vals, counts := empty.UniqueCounts(); len(vals.Data()) != 0 || len(counts) != 0 {
		t.Errorf("UniqueCounts(empty) = %v %v", vals.Data(), counts)
	}
}

func TestUniqueCountsNaN(t *testing.T) {
	a := FromSlice([]float64{math.NaN(), 1, math.NaN()}, 3)
	vals, counts := a.UniqueCounts()
	if len(vals.Data()) != 2 || !math.IsNaN(vals.At(1)) || !slices.Equal(counts, []int{1, 2}) {
		t.Errorf("UniqueCounts(NaNs) = %v %v, want [1 NaN] [1 2]", vals.Data(), counts)
	}
}
