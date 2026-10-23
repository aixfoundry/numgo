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

// panicNoArrays reports an operation that received no input arrays.
func panicNoArrays() {
	panic("numgo: need at least one array to concatenate")
}

// panicConcatenateShapes reports Concatenate inputs whose non-axis dimensions differ.
func panicConcatenateShapes() {
	panic("numgo: all the input array dimensions except for the concatenation axis must match exactly")
}

// panicStackShapes reports Stack inputs of differing shapes.
func panicStackShapes() {
	panic("numgo: all input arrays must have the same shape")
}

// panicSqueeze reports squeezing an axis whose size is not one.
func panicSqueeze() {
	panic("numgo: cannot select an axis to squeeze out which has size not equal to one")
}

// panicSplitIndices reports out-of-order or out-of-bounds Split boundaries.
func panicSplitIndices() {
	panic("numgo: split indices must be ascending and within the axis bounds")
}

// panicRepeat reports a negative repeat count.
func panicRepeat(count int) {
	panic(fmt.Sprintf("numgo: repeat count %d must be non-negative", count))
}

// panicMeshGrid reports a MeshGrid input that is not 1-D.
func panicMeshGrid() {
	panic("numgo: MeshGrid inputs must be 1-D")
}

// panicComplexStat reports a statistical reduction that is undefined for
// complex element types (numpy raises TypeError for the same reason).
func panicComplexStat(op string) {
	panic(fmt.Sprintf("numgo: %s are not defined for complex arrays", op))
}

// panicKth reports a partition kth outside the axis bounds. numpy's wording
// for np.partition with a bad kth is a ValueError on index range.
func panicKth(kth, n int) {
	panic(fmt.Sprintf("numgo: kth(=%d) out of bounds (%d)", kth, n))
}

// panicTakeIndex reports an advanced-indexing entry outside the axis dimension
// (numpy wording, reporting the original index value).
func panicTakeIndex(idx, axis, size int) {
	panic(fmt.Sprintf("numgo: index %d is out of bounds for axis %d with size %d",
		idx, axis, size))
}

// panicCompressMask reports a boolean mask whose length differs from the
// selection size (axis dimension or flattened element count).
func panicCompressMask(got, want int) {
	panic(fmt.Sprintf("numgo: mask length %d does not match selection size %d",
		got, want))
}

// panicComplexAbs reports Abs on a complex array, whose real-valued result
// Array[T] cannot represent.
func panicComplexAbs() {
	panic("numgo: Abs on complex arrays returns real values; use AbsToReal")
}

// panicMatmul1D reports matmul applied to two 1-D operands.
func panicMatmul1D() {
	panic("numgo: matmul: cannot multiply two 1-d arrays; use Vdot")
}

// panicMatmulShape reports incompatible matmul/Vdot inner dimensions.
func panicMatmulShape(op string, a, b []int) {
	panic(fmt.Sprintf("numgo: %s: shape mismatch %s x %s", op, shapeString(a), shapeString(b)))
}

// panicDotLead reports N-D dot operands whose leading dimensions differ and
// neither operand is 2-D.
func panicDotLead(a, b []int) {
	panic(fmt.Sprintf("numgo: dot: leading dimensions differ %s x %s", shapeString(a), shapeString(b)))
}
