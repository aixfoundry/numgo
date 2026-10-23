package numgo

// MinMaxScaler scales the elements of the array to a range of [0, 1] using the min-max normalization technique.
// It calculates the minimum and maximum values of the array, then scales each element by subtracting the minimum
// and dividing by the range (max - min).
// A constant array has a zero range, so its elements divide by zero (IEEE semantics: Inf/NaN).
//
// Example:
//
//	a := New1DArray([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
//	scaled := a.MinMaxScaler()
//	fmt.Println(scaled.Data()) // Output: [0 0.25 0.5 0.75 1]
func (a *Array[T]) MinMaxScaler() (ret *Array[T]) {
	min := a.Min(-1).At(0, 0)
	max := a.Max(-1).At(0, 0)
	ret = a.SubValue(min).DivValue(max - min)
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
//	fmt.Println(scaled.Data()) // Output: [0 0.25 0.5 0.75 1]
func MinMaxScaler[T Float](a *Array[T]) *Array[T] {
	return a.MinMaxScaler()
}
