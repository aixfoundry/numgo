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

// panicEinsumLabelCount reports an operand whose subscript label count differs
// from its dimension count.
func panicEinsumLabelCount(labels, ndim int) {
	panic(fmt.Sprintf("numgo: einsum: subscript label count %d does not match array dimension %d",
		labels, ndim))
}

// panicEinsumOperands reports a subscript/operand count mismatch.
func panicEinsumOperands(subs, operands int) {
	panic(fmt.Sprintf("numgo: einsum: %d subscripts for %d operands", subs, operands))
}

// panicEinsumOutputLabel reports an output label absent from every input.
func panicEinsumOutputLabel(l rune) {
	panic(fmt.Sprintf("numgo: einsum: output label %c does not appear in any input", l))
}

// panicEinsumOutputRepeat reports an output label written more than once.
func panicEinsumOutputRepeat(l rune) {
	panic(fmt.Sprintf("numgo: einsum: output subscript %c appears multiple times", l))
}

// panicEinsumOutputEllipsis reports "..." in the output while no operand has
// an ellipsis (numpy: output has more dimensions than subscripts).
func panicEinsumOutputEllipsis() {
	panic("numgo: einsum: output has an ellipsis but no operand provides one")
}

// panicEinsumBadChar reports an invalid subscript character.
func panicEinsumBadChar(r rune, where string) {
	panic(fmt.Sprintf("numgo: einsum: invalid subscript %q in %s, subscripts must be letters", r, where))
}

// panicAllNaN reports an all-NaN slice passed to an operation numpy raises on
// (nanargmax/nanargmin raise ValueError "All-NaN slice encountered").
func panicAllNaN() {
	panic("numgo: All-NaN slice encountered")
}

// panicQRange reports a percentile/quantile outside its range (numpy
// ValueError "Percentiles must be in the range [0, 100]").
func panicQRange(kind string, hi float64) {
	panic(fmt.Sprintf("numgo: %s must be in the range [0, %g]", kind, hi))
}

// panicGradient reports an axis too short for numpy's default gradient
// (edge_order 1 needs two points per axis).
func panicGradient() {
	panic("numgo: shape of array too small to calculate a numerical gradient, at least 2 elements are required")
}

// panicPadWidth reports a negative pad width (numpy ValueError "index can't
// contain negative values").
func panicPadWidth() {
	panic("numgo: pad widths cannot be negative")
}

// panicHistBins reports a non-positive histogram bin count (numpy ValueError
// "`bins` must be positive, when an integer").
func panicHistBins() {
	panic("numgo: `bins` must be positive, when an integer")
}

// panicSampleLength reports histogram2d x/y lengths (or interp xp/fp lengths)
// that differ (numpy: "x and y must have the same length." / "xp and fp are
// not of same length").
func panicSampleLength() {
	panic("numgo: sample point and value lengths differ")
}

// panicEmptyXP reports interp called without sample points (numpy ValueError
// "array of sample points is empty").
func panicEmptyXP() {
	panic("numgo: array of sample points is empty")
}

// panicWindowNegative reports a negative sliding-window size (numpy ValueError
// "`window_shape` cannot contain negative values").
func panicWindowNegative() {
	panic("numgo: `window_shape` cannot contain negative values")
}

// panicWindowCount reports a window_shape length that does not match the
// array's dimension count (numpy ValueError "Since axis is `None`, must
// provide window_shape for all dimensions of `x`; got N window_shape
// elements and `x.ndim` is M.").
func panicWindowCount(got, ndim int) {
	panic(fmt.Sprintf("numgo: since axis is `None`, must provide window_shape for all dimensions of `x`; got %d window_shape elements and `x.ndim` is %d", got, ndim))
}

// panicWindowTooLarge reports a window larger than its axis (numpy ValueError
// "window shape cannot be larger than input array shape").
func panicWindowTooLarge() {
	panic("numgo: window shape cannot be larger than input array shape")
}

// panicAsStridedDims reports AsStrided shape/strides of different lengths.
func panicAsStridedDims() {
	panic("numgo: as_strided: shape and strides must have the same length")
}

// panicAsStridedNegDim reports a negative AsStrided dimension.
func panicAsStridedNegDim() {
	panic("numgo: as_strided: negative dimensions are not allowed")
}

// panicAsStridedBounds reports an AsStrided view that can reach outside the
// backing data (a deliberate safety check numpy's as_strided lacks).
func panicAsStridedBounds() {
	panic("numgo: as_strided: view exceeds data bounds")
}

// panicBroadcastArrays reports BroadcastArrays inputs that cannot broadcast
// together (numpy wording listing every operand shape).
func panicBroadcastArrays(shapes [][]int) {
	var b strings.Builder
	b.WriteString("numgo: operands could not be broadcast together with shapes")
	for _, s := range shapes {
		b.WriteString(" " + shapeString(s))
	}
	panic(b.String())
}

// panicRealIfClose reports RealIfClose on an array with significant imaginary
// parts (deviation from numpy, which returns the complex array).
func panicRealIfClose() {
	panic("numgo: RealIfClose: array has significant imaginary parts")
}
