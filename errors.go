package numgo

import (
	"fmt"
	"strings"
)

// shapeString renders a shape as "(2,3)"; the empty shape is "()".
func shapeString(shape []int) string {
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = fmt.Sprint(d)
	}
	return "(" + strings.Join(parts, ",") + ")"
}

// panicBroadcast reports incompatible operand shapes.
func panicBroadcast(a, b []int) {
	panic(fmt.Sprintf("numgo: operands could not be broadcast together with shapes %s %s",
		shapeString(a), shapeString(b)))
}

// panicIndex reports a bad index count or an out-of-bounds index.
func panicIndex(coords, shape []int) {
	if len(coords) != len(shape) {
		panic(fmt.Sprintf("numgo: expected %d indices for a %d-d array, got %d",
			len(shape), len(shape), len(coords)))
	}
	panic(fmt.Sprintf("numgo: index %s is out of bounds for shape %s",
		shapeString(coords), shapeString(shape)))
}

// panicAxis reports an axis outside [0, ndim).
func panicAxis(axis, ndim int) {
	panic(fmt.Sprintf("numgo: axis %d is out of bounds for %d-d array", axis, ndim))
}

// panicAxes reports a malformed axis permutation.
func panicAxes() {
	panic("numgo: axes don't match array")
}

// panicReshape reports a reshape whose element counts do not match.
func panicReshape(size int, shape []int) {
	panic(fmt.Sprintf("numgo: cannot reshape array of size %d into shape %s",
		size, shapeString(shape)))
}

// panicReshapeUnknown reports more than one inferred (-1) reshape dimension.
func panicReshapeUnknown() {
	panic("numgo: can only specify one unknown dimension (-1)")
}

// panicRanges reports a Slice whose range count differs from the array's ndim.
func panicRanges(got, ndim int) {
	panic(fmt.Sprintf("numgo: expected %d ranges for a %d-d array, got %d",
		ndim, ndim, got))
}

// panicInto reports an Into destination whose shape differs from the result shape.
func panicInto(got, want []int) {
	panic(fmt.Sprintf("numgo: output shape %s does not match result shape %s",
		shapeString(got), shapeString(want)))
}

// panicJagged reports a non-rectangular [][]T input.
func panicJagged(rowLen, want int) {
	panic(fmt.Sprintf("numgo: New2DArray: jagged input: row has %d elements, want %d",
		rowLen, want))
}

// panicZeroStep reports a zero Arange step.
func panicZeroStep() {
	panic("numgo: Arange: step must not be zero")
}

// panicSamples reports a negative Linspace sample count.
func panicSamples(num int) {
	panic(fmt.Sprintf("numgo: number of samples, %d, must be non-negative", num))
}
