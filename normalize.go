package numgo

// MinMaxScaler scales the elements of the array to a range of [0, 1] using the min-max normalization technique.
// It calculates the minimum and maximum values of the array, then scales each element by subtracting the minimum
// and dividing by the range (max - min).
//
// Example:
//
//	a := New1DArray([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
//	scaled := a.MinMaxScaler()
//	fmt.Println(scaled.Data) // Output: [0 0.25 0.5 0.75 1]
func (a *Array) MinMaxScaler() (ret *Array) {
	min, _, _ := a.Min()
	max, _, _ := a.Max()
	seq := max - min
	ret = a.SubValue(min).DivValue(seq)
	return
}

// MinMaxScaler scales the elements of the array to a range of [0, 1] using the min-max normalization technique.
// It calculates the minimum and maximum values of the array, then scales each element by subtracting the minimum
// and dividing by the range (max - min).
//
// Example:
//
//	a := New1DArray([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
//	scaled := MinMaxScaler(a)
//	fmt.Println(scaled.Data) // Output: [0 0.25 0.5 0.75 1]
func MinMaxScaler(a *Array) *Array {
	return a.MinMaxScaler()
}
