package numgo

import "testing"

func TestEMCP(t *testing.T) {
	a := NewArray(3, 4, []float64{
		2, 1, -1, 8,
		-3, -1, 2, -11,
		-2, 1, 2, -3,
	})
	sol := EMCP(a)
	t.Log(sol)
}

func TestLSQ(t *testing.T) {
	a := NewArray(6, 3, []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		-1.0, 1.0, 0.0,
		0.0, -1.0, 1.0,
		-1.0, 0.0, 1.0,
	})
	b := NewArray(6, 1, []float64{
		1.0,
		2.0,
		3.0,
		1.0,
		2.0,
		1.0,
	})
	sol := CELSQ(a, b)
	t.Log(sol)

	t.Logf("\n%v\n", sol)
}

func TestLSQFitting(t *testing.T) {
	a := NewArray(8, 2, []float64{
		-1.0, 10.0,
		0.0, 9.0,
		1.0, 7.0,
		2.0, 5.0,
		3.0, 4.0,
		4.0, 3.0,
		5.0, 0.0,
		6.0, -1.0})
	sol := LSQFitting(a)
	t.Log(sol)

	t.Logf("\n%v\n", sol)
}

func TestPolyFitting(t *testing.T) {
	a := NewArray(7, 2, []float64{
		1.0, 10.0,
		3.0, 5.0,
		4.0, 4.0,
		5.0, 2.0,
		6.0, 1.0,
		7.0, 1.0,
		8.0, 2.0})
	sol := PolyFitting(a, 2)
	t.Log(sol)

	t.Logf("\n%v\n", sol)
}
