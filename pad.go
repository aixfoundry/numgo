package numgo

// This file holds array padding (numpy pad's mode subset). Pad builds the
// output with per-axis index mapping, which is exactly equivalent to numpy's
// axis-by-axis padding for these modes: every virtual coordinate resolves to
// a source coordinate per axis independently.

// PadMode selects how Pad fills the padded area.
type PadMode int

const (
	PadConstant  PadMode = iota // fill with the given constant value
	PadEdge                     // repeat the edge value
	PadReflect                  // reflect about the edge, edge value NOT repeated
	PadSymmetric                // reflect including the edge value
	PadWrap                     // circular shift (numpy wrap)
	PadEmpty                    // leave Go zero values: numpy's uninitialized
	// memory cannot be reproduced in Go, so PadEmpty behaves like
	// PadConstant(0).
)

// Pure: Pad returns a new array grown by pads[ax] = {before, after} along
// each axis, filled per mode. pads must have one entry per axis and no
// negative widths.
func Pad[T Number](a *Array[T], pads [][2]int, mode PadMode, constantValue T) *Array[T] {
	if len(pads) != a.Ndim() {
		panicRanges(len(pads), a.Ndim())
	}
	for _, p := range pads {
		if p[0] < 0 || p[1] < 0 {
			panicPadWidth()
		}
	}
	shape := make([]int, a.Ndim())
	for i, p := range pads {
		shape[i] = p[0] + a.shape[i] + p[1]
	}
	switch mode {
	case PadConstant:
		// Seed with the constant, then copy the interior from a.
		out := Full(constantValue, shape...)
		CopyInto(out.Slice(padRanges(pads, a.shape)...), a)
		return out
	case PadEmpty:
		// Go zero values: numpy's uninitialized memory cannot be
		// reproduced in Go (documented deviation).
		out := Zeros[T](shape...)
		CopyInto(out.Slice(padRanges(pads, a.shape)...), a)
		return out
	}
	return FromFunction(shape, func(coords ...int) T {
		src := make([]int, len(coords))
		for i, c := range coords {
			src[i] = padIndex(mode, c, pads[i][0], a.shape[i])
		}
		return a.At(src...)
	})
}

// padRanges builds the interior Slice ranges: `before` zeros then whole-axis
// ranges (a zero Range selects the entire axis).
func padRanges(pads [][2]int, aShape []int) []Range {
	rs := make([]Range, len(aShape))
	for i, p := range pads {
		rs[i] = Range{Start: p[0], Stop: p[0] + aShape[i], Step: 1}
	}
	return rs
}

// padIndex maps one output coordinate c on an axis padded by before, with
// source length n, back to the source coordinate for the mode.
func padIndex(mode PadMode, c, before, n int) int {
	v := c - before
	if v >= 0 && v < n {
		return v
	}
	switch mode {
	case PadEdge:
		if v < 0 {
			return 0
		}
		return n - 1
	case PadReflect:
		if n == 1 {
			return 0
		}
		m := 2 * (n - 1)
		r := mod(v, m)
		if r >= n {
			r = m - r
		}
		return r
	case PadSymmetric:
		m := 2 * n
		r := mod(v, m)
		if r >= n {
			r = m - 1 - r
		}
		return r
	case PadWrap:
		return mod(v, n)
	}
	return v // unreachable: constant/empty never reach the mapping
}

// mod is the non-negative modulo.
func mod(v, m int) int {
	r := v % m
	if r < 0 {
		r += m
	}
	return r
}
