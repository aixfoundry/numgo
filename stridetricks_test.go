package numgo

import (
	"slices"
	"testing"
)

// assertPanics fails unless fn panics; name labels the case (assertPanic is
// already taken by io_test.go).
func assertPanics(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected panic", name)
		}
	}()
	fn()
}

// numpy: sliding_window_view(arange(1,6), 3) -> shape (3,3), rows [1 2 3][2 3 4][3 4 5]
func TestSlidingWindow1D(t *testing.T) {
	a := Arange(1.0, 6.0, 1.0)
	v := a.SlidingWindowView(3)
	if !slices.Equal(v.Shape(), []int{3, 3}) {
		t.Fatalf("shape = %v, want [3 3]", v.Shape())
	}
	want := [][]float64{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}
	for i := range want {
		for j := range want[i] {
			if got := v.At(i, j); got != want[i][j] {
				t.Fatalf("At(%d,%d) = %v, want %v", i, j, got, want[i][j])
			}
		}
	}
	// View semantics: mutating the view mutates the original. strides are
	// (1,1), so v[1,1] = a[1+1] = a[2] — the same element aliases at several
	// window slots, exactly numpy's aliasing behavior.
	v.Set([]int{1, 1}, 99)
	if a.At(2) != 99 {
		t.Fatalf("view write did not reach original: At(2) = %v", a.At(2))
	}
	a.Set([]int{2}, 3)
}

// numpy: sliding_window_view(arange(6).reshape(2,3), (2,2)) -> shape (1,2,2,2)
// with v[0,1] = [[1 2],[4 5]].
func TestSlidingWindow2D(t *testing.T) {
	a := New2DArray([][]float64{{0, 1, 2}, {3, 4, 5}})
	v := a.SlidingWindowView(2, 2)
	if !slices.Equal(v.Shape(), []int{1, 2, 2, 2}) {
		t.Fatalf("shape = %v, want [1 2 2 2]", v.Shape())
	}
	want := []float64{0, 1, 3, 4, 1, 2, 4, 5}
	got := v.Ravel().Data()
	if !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
}

// Window 1 leaves the shape with appended unit axes; equal windows are the
// identity view modulo the extra axes.
func TestSlidingWindowIdentity(t *testing.T) {
	a := Arange(1.0, 6.0, 1.0)
	v := a.SlidingWindowView(1)
	if !slices.Equal(v.Shape(), []int{5, 1}) {
		t.Fatalf("shape = %v, want [5 1]", v.Shape())
	}
	for i := range 5 {
		if v.At(i, 0) != float64(i+1) {
			t.Fatalf("At(%d,0) = %v", i, v.At(i, 0))
		}
	}
	// window == axis length: one window slot.
	if got := a.SlidingWindowView(5); !slices.Equal(got.Shape(), []int{1, 5}) {
		t.Fatalf("full-window shape = %v", got.Shape())
	}
}

func TestSlidingWindowPanics(t *testing.T) {
	a := Arange(1.0, 6.0, 1.0)
	assertPanics(t, "negative", func() { a.SlidingWindowView(-1) })
	assertPanics(t, "too large", func() { a.SlidingWindowView(6) })
	assertPanics(t, "count", func() { New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}}).SlidingWindowView(2) })
}

// AsStrided with a's own shape/strides is the identity view.
func TestAsStridedIdentity(t *testing.T) {
	a := New2DArray([][]float64{{1, 2, 3}, {4, 5, 6}})
	v := a.AsStrided([]int{2, 3}, []int{3, 1})
	if !slices.Equal(v.Shape(), []int{2, 3}) {
		t.Fatalf("shape = %v", v.Shape())
	}
	for i := range 2 {
		for j := range 3 {
			if v.At(i, j) != float64(3*i+j+1) {
				t.Fatalf("At(%d,%d) = %v", i, j, v.At(i, j))
			}
		}
	}
}

// AsStrided with a negative stride reverses the axis (numpy:
// as_strided(x, (3,), (-1,))).
func TestAsStridedReverse(t *testing.T) {
	// A negative stride starts at a's offset (the LAST element is reachable
	// first): with a starting at offset 2 of the same data, positions 0..2
	// are covered. Mirrors numpy as_strided(x[len-1:], (3,), (-1,)).
	all := New1DArray([]float64{1, 2, 3})
	tail := all.Slice(From(2)) // offset 2, one element
	v := tail.AsStrided([]int{3}, []int{-1})
	got := v.Ravel().Data()
	if !slices.Equal(got, []float64{3, 2, 1}) {
		t.Fatalf("values = %v, want [3 2 1]", got)
	}
	// A negative stride reaching before the data start panics (bounds check).
	assertPanics(t, "before start", func() { all.AsStrided([]int{4}, []int{-1}) })
}

// AsStrided windows built by hand match SlidingWindowView.
func TestAsStridedWindow(t *testing.T) {
	a := Arange(1.0, 6.0, 1.0)
	v := a.AsStrided([]int{3, 3}, []int{1, 1})
	if !slices.Equal(v.Shape(), []int{3, 3}) {
		t.Fatalf("shape = %v", v.Shape())
	}
	if v.At(2, 2) != 5 {
		t.Fatalf("At(2,2) = %v, want 5", v.At(2, 2))
	}
	// Zero-size views skip reachability and are allowed.
	if got := a.AsStrided([]int{0}, []int{100}); got.Size() != 0 {
		t.Fatalf("empty view size = %d", got.Size())
	}
}

func TestAsStridedPanics(t *testing.T) {
	a := Arange(1.0, 4.0, 1.0)
	assertPanics(t, "bounds", func() { a.AsStrided([]int{4}, []int{1}) })
	assertPanics(t, "bounds neg", func() { a.AsStrided([]int{3}, []int{-2}) })
	assertPanics(t, "rank", func() { a.AsStrided([]int{2, 2}, []int{1}) })
	assertPanics(t, "neg dim", func() { a.AsStrided([]int{-1}, []int{1}) })
}

// numpy: broadcast_arrays(arange(6).reshape(2,3), [10,20,30]) -> both (2,3),
// the second is a stride-0 view sharing its data.
func TestBroadcastArrays(t *testing.T) {
	a := New2DArray([][]float64{{0, 1, 2}, {3, 4, 5}})
	b := New1DArray([]float64{10, 20, 30})
	out := BroadcastArrays(a, b)
	if !slices.Equal(out[0].Shape(), []int{2, 3}) || !slices.Equal(out[1].Shape(), []int{2, 3}) {
		t.Fatalf("shapes = %v %v", out[0].Shape(), out[1].Shape())
	}
	if out[0].At(1, 2) != 5 {
		t.Fatalf("out[0].At(1,2) = %v", out[0].At(1, 2))
	}
	if out[1].At(0, 1) != 20 || out[1].At(1, 1) != 20 {
		t.Fatalf("out[1] column = %v %v", out[1].At(0, 1), out[1].At(1, 1))
	}
	// Views share data: mutating a changes out[0]; b changes out[1].
	a.Set([]int{0, 0}, 42)
	if out[0].At(0, 0) != 42 {
		t.Fatal("out[0] does not share a's data")
	}
	b.Set([]int{0}, 77)
	if out[1].At(1, 0) != 77 {
		t.Fatal("out[1] does not share b's data")
	}
}

func TestBroadcastArraysPanic(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
	}()
	BroadcastArrays(New1DArray([]float64{1, 2}), New1DArray([]float64{1, 2, 3}))
}
