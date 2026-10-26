package numgo

import (
	"slices"
	"testing"
)

func TestSearchSorted(t *testing.T) {
	a := FromSlice([]float64{10, 20, 30, 50})
	v := FromSlice([]float64{20})
	if got := a.SearchSorted(v, "left"); !slices.Equal(got, []int{1}) {
		t.Errorf("left 20 = %v, want [1]", got)
	}
	if got := a.SearchSorted(v, "right"); !slices.Equal(got, []int{2}) {
		t.Errorf("right 20 = %v, want [2]", got)
	}
	// out-of-range values
	out := FromSlice([]float64{5, 60, 30.5})
	if got := a.SearchSorted(out, "left"); !slices.Equal(got, []int{0, 4, 3}) {
		t.Errorf("left out-of-range = %v, want [0 4 3]", got)
	}
	// vector v
	vv := FromSlice([]float64{20, 30})
	if got := a.SearchSorted(vv, "right"); !slices.Equal(got, []int{2, 3}) {
		t.Errorf("right [20 30] = %v, want [2 3]", got)
	}
	mustPanic(t, `numgo: SearchSorted: side must be "left" or "right"`, func() {
		a.SearchSorted(v, "middle")
	})
}

func TestDigitize(t *testing.T) {
	bins := FromSlice([]float64{10, 20, 30, 50})
	x := FromSlice([]float64{5, 10, 15, 20, 25, 30, 35, 50, 60})
	// left (right=false): index of first bin >= x
	if got := bins.Digitize(x, false); !slices.Equal(got, bins.SearchSorted(x, "left")) {
		t.Errorf("Digitize(right=false) = %v, want SearchSorted left %v", got, bins.SearchSorted(x, "left"))
	}
	// right (right=true): index of first bin > x
	if got := bins.Digitize(x, true); !slices.Equal(got, bins.SearchSorted(x, "right")) {
		t.Errorf("Digitize(right=true) = %v, want SearchSorted right %v", got, bins.SearchSorted(x, "right"))
	}
	if got, want := bins.Digitize(x, false), []int{0, 0, 1, 1, 2, 2, 3, 3, 4}; !slices.Equal(got, want) {
		t.Errorf("Digitize(left) = %v, want %v", got, want)
	}
}
