package fft

import "fmt"

// Panic messages live here, worded after numpy.

// panicAxis reports an axis outside [0, ndim) with the original, un-normalized
// value (numpy normalize_axis_index wording).
func panicAxis(axis, ndim int) {
	panic(fmt.Sprintf("numgo/fft: axis %d is out of bounds for %d-d array", axis, ndim))
}

// panicInvalidN reports a non-positive resolved transform length (numpy
// _raw_fft wording).
func panicInvalidN(n int) {
	panic(fmt.Sprintf("numgo/fft: Invalid number of FFT data points (%d) specified.", n))
}

// panicShapeAxes reports an s/axes length mismatch (numpy wording).
func panicShapeAxes() {
	panic("numgo/fft: Shape and axes have different lengths.")
}

// panicSingleAxis reports more than one axis passed to a 1-D transform.
func panicSingleAxis() {
	panic("numgo/fft: 1-d transforms take at most one axis")
}
