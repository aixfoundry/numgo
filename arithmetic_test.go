package numgo

import "testing"

func TestArray_Max(t *testing.T) {
	a := NewArray(1, 9, []float64{0, 1, 2, 3, 13, 5, 6, 7, 9})
	sol, r, c := a.Max()
	if sol != 13 || r != 0 || c != 4 {
		t.Log(sol, r, c)
		t.FailNow()
	}
}
