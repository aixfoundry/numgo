package ma

import (
	"github.com/kaisawind/numgo"
	"github.com/kaisawind/numgo/internal/nditer"
)

// Small index/shape helpers shared by the operations.

// broadcastShape right-aligns shapes, panicking on incompatibility (numpy
// message wording).
func broadcastShape(a, b []int) []int {
	shape, _, err := nditer.Broadcast(a, b)
	if err != nil {
		panic("numgo: operands could not be broadcast together with shapes " +
			shapeString(a) + " " + shapeString(b))
	}
	return shape
}

func shapeString(shape []int) string {
	s := "("
	for i, d := range shape {
		if i > 0 {
			s += ","
		}
		s += itoa(d)
	}
	if len(shape) == 1 {
		s += ","
	}
	return s + ")"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

func sizeOf(shape []int) int { return nditer.Size(shape) }

// flat computes the C-order linear index of coords within shape. coords must
// be full-rank for shape.
func flat(shape, coords []int) int {
	idx := 0
	for i, c := range coords {
		idx = idx*shape[i] + c
	}
	return idx
}

// maskAtB reads a mask against a broadcast operand shape at full result
// coords (size-1 dims read slot 0).
func maskAtB(mask []bool, opShape, coords []int) bool {
	lead := len(coords) - len(opShape)
	sub := make([]int, len(opShape))
	copy(sub, coords[lead:])
	for i, d := range opShape {
		if d == 1 {
			sub[i] = 0
		}
	}
	return mask[flat(opShape, sub)]
}

// valueAt reads an operand element for broadcast coords (size-1 dims read
// slot 0).
func valueAt[T numgo.Number](a *numgo.Array[T], coords []int) T {
	shape := a.Shape()
	lead := len(coords) - len(shape)
	sub := make([]int, len(shape))
	copy(sub, coords[lead:])
	for i, d := range shape {
		if d == 1 {
			sub[i] = 0
		}
	}
	return a.At(sub...)
}

// normAxes resolves negative axes and rejects invalid ones (root convention).
func normAxes(axes []int, ndim int) []int {
	out := make([]int, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax += ndim
		}
		if ax < 0 || ax >= ndim {
			panicAxis(ax, ndim)
		}
		out[i] = ax
	}
	return out
}
