package numgo

import "testing"

func TestArray_Cat(t *testing.T) {
	a := NewArray(6, 3, []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		-1.0, 1.0, 0.0,
		0.0, -1.0, 1.0,
		-1.0, 0.0, 1.0,
	})
	b := NewArray(1, 3, []float64{
		1.0,
		2.0,
		3.0,
	})
	t.Logf("\n%v\n", a)
	t.Logf("\n%v\n", b)

	t.Logf("\n%v\n", a.Cat(b, 0))
}
