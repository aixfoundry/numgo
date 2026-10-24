package random

// arangeI builds the int range [0, n) as an arrayI (numpy arange for
// permutation); int64 is not a numgo.Number so the root Arange is unusable.
func arangeI(n int) *arrayI {
	data := make([]int64, n)
	for i := range data {
		data[i] = int64(i)
	}
	return &arrayI{data: data}
}
