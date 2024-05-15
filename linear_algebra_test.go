package numgo

import "testing"

func TestArray_EMCP(t *testing.T) {
	a := NewArray(3, 4, []float64{
		2, 1, -1, 8,
		-3, -1, 2, -11,
		-2, 1, 2, -3,
	})
	sol := a.EMCP()
	t.Log(sol)
}
