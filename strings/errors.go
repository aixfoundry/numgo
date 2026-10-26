package strings

import (
	"fmt"
	"strings"
)

// Panic messages live here, reusing numgo's root message text
// patterns with the subpackage prefix. The root helpers are unexported and a
// root import would not provide them, so they are mirrored locally.

// shapeString renders a shape as "(2,3)"; the empty shape is "()".
func shapeString(shape []int) string {
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = fmt.Sprint(d)
	}
	return "(" + strings.Join(parts, ",") + ")"
}

// panicIndex reports a bad index count or an out-of-bounds index.
func panicIndex(coords, shape []int) {
	if len(coords) != len(shape) {
		panic(fmt.Sprintf("numgo/strings: expected %d indices for a %d-d array, got %d",
			len(shape), len(shape), len(coords)))
	}
	panic(fmt.Sprintf("numgo/strings: index %s is out of bounds for shape %s",
		shapeString(coords), shapeString(shape)))
}

// panicAxis reports an axis outside [0, ndim).
func panicAxis(axis, ndim int) {
	panic(fmt.Sprintf("numgo/strings: axis %d is out of bounds for %d-d array", axis, ndim))
}

// panicAxes reports a malformed axis permutation.
func panicAxes() {
	panic("numgo/strings: axes don't match array")
}

// panicReshape reports a reshape whose element counts do not match.
func panicReshape(size int, shape []int) {
	panic(fmt.Sprintf("numgo/strings: cannot reshape array of size %d into shape %s",
		size, shapeString(shape)))
}

// panicReshapeUnknown reports more than one inferred (-1) reshape dimension.
func panicReshapeUnknown() {
	panic("numgo/strings: can only specify one unknown dimension (-1)")
}

// panicRanges reports a Slice whose range count differs from the array's ndim.
func panicRanges(got, ndim int) {
	panic(fmt.Sprintf("numgo/strings: expected %d ranges for a %d-d array, got %d",
		ndim, ndim, got))
}

// panicNoArrays reports an operation that received no input arrays.
func panicNoArrays() {
	panic("numgo/strings: need at least one array to concatenate")
}

// panicConcatenateShapes reports Concatenate inputs whose non-axis dimensions differ.
func panicConcatenateShapes() {
	panic("numgo/strings: all the input array dimensions except for the concatenation axis must match exactly")
}

// panicNotFound reports Index/RIndex on an element lacking the substring
// (numpy ValueError wording).
func panicNotFound() {
	panic("numgo/strings: substring not found")
}

// panicNegCount reports a negative Repeat count (numpy ValueError wording).
func panicNegCount() {
	panic("numgo/strings: multiply() argument must be non-negative")
}
