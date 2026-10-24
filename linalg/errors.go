package linalg

import (
	"fmt"
	"math"

	"github.com/aixfoundry/numgo"
)

// NumberFromFloat converts a float64 to any real Number element type.
// Exported for callers that need the same float64→T bridge as the package
// internals.
func NumberFromFloat[T numgo.Number](v float64) T { return scalarOf[T](v) }

// Panic messages live here (design.md P0-4), worded after numpy's
// numpy.linalg exception strings.

// panicComplex reports a complex element type reaching a real-only routine.
func panicComplex() {
	panic("numgo/linalg: complex dtypes not supported yet")
}

// panicSquare reports non-square trailing matrix dimensions.
func panicSquare(op string, shape []int) {
	panic(fmt.Sprintf("numgo/linalg: %s: Last 2 dimensions of the array must be square, got %s",
		op, shapeString(shape)))
}

// panicSingular reports an exactly singular matrix.
func panicSingular() { panic("numgo/linalg: Singular matrix") }

// panicDims reports incompatible operand dimensions.
func panicDims(op, msg string) {
	panic(fmt.Sprintf("numgo/linalg: %s: %s", op, msg))
}

// panicConvergence reports a failed iterative decomposition.
func panicConvergence(op string) {
	panic(fmt.Sprintf("numgo/linalg: %s: algorithm failed to converge", op))
}

// panicAxis reports an axis outside [0, ndim) (numpy normalize_axis_index).
func panicAxis(axis, ndim int) {
	panic(fmt.Sprintf("numgo/linalg: axis %d is out of bounds for %d-d array", axis, ndim))
}

// panicNormOrder reports an invalid norm order for the operand kind.
func panicNormOrder(ord string) {
	panic(fmt.Sprintf("numgo/linalg: Invalid norm order '%s' for vectors", ord))
}

// panicNormOrderMatrix reports an invalid matrix norm order.
func panicNormOrderMatrix() {
	panic("numgo/linalg: Invalid norm order for matrices.")
}

// panicNormAxis reports an improper axis count for Norm.
func panicNormAxis() {
	panic("numgo/linalg: Improper number of dimensions to norm.")
}

// panicMatrixNormAxis reports a 1-D operand passed to MatrixNorm.
func panicMatrixNormAxis() {
	panic("numgo/linalg: matrix_norm requires at least 2 dimensions")
}

// shapeString renders a shape as "(2,3)"; the empty shape is "()".
func shapeString(shape []int) string {
	parts := ""
	for i, d := range shape {
		if i > 0 {
			parts += ","
		}
		parts += fmt.Sprint(d)
	}
	return "(" + parts + ")"
}

// lapackNorm is the linalg-side norm selector for the internal kernels.
type lapackNorm int

const (
	normFro lapackNorm = iota
	normOne
	normInf
	normMax
)

// matrixNorm computes a row-major matrix norm with the own kernels.
func matrixNorm(norm lapackNorm, m, n int, a []float64, _ int) float64 {
	switch norm {
	case normMax:
		best := 0.0
		for _, v := range a {
			best = math.Max(best, math.Abs(v))
		}
		return best
	default:
		return 0
	}
}
