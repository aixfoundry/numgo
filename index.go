package numgo

import (
	"math"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// Range is one slice axis. A zero Range means the whole axis. Step 0 is treated
// as 1. Negative Start/Stop count from the end. With a negative Step and no
// bounds, the whole axis is reversed (Range{Step: -1}).
type Range struct {
	Start, Stop, Step int
}

// Pure: R builds a Range with explicit start, stop and step.
func R(start, stop, step int) Range {
	return Range{Start: start, Stop: stop, Step: step}
}

// Pure: To builds a Range from 0 up to stop (exclusive) with step 1.
func To(stop int) Range {
	return Range{Stop: stop}
}

// Pure: From builds a Range from start to the end of the axis with step 1. Stop uses
// an explicit sentinel because a zero Stop would be ambiguous with an explicit 0.
func From(start int) Range {
	return Range{Start: start, Stop: math.MaxInt}
}

// normalizeRange resolves a Range against a dimension, returning the start,
// element count and effective step.
func normalizeRange(r Range, dim int) (start, n, step int) {
	step = r.Step
	if step == 0 {
		step = 1
	}
	if start, stop := r.Start, r.Stop; start == 0 && stop == 0 {
		if step > 0 {
			return 0, dim, step
		}
		return dim - 1, dim, step
	}
	start, stop := r.Start, r.Stop
	if start < 0 {
		start += dim
	}
	if stop < 0 {
		stop += dim
	}
	if step > 0 {
		start = min(max(start, 0), dim)
		stop = min(max(stop, 0), dim)
		if stop <= start {
			return start, 0, step
		}
		return start, (stop - start + step - 1) / step, step
	}
	start = min(max(start, -1), dim-1)
	stop = min(max(stop, -1), dim-1)
	if stop >= start {
		return start, 0, step
	}
	return start, (start - stop + (-step) - 1) / (-step), step
}

// Pure: NormalizeRange exposes range normalization for subpackages that build
// views from numgo.Range values (numgo/strings).
func NormalizeRange(r Range, dim int) (start, n, step int) {
	return normalizeRange(r, dim)
}

// Pure: At returns the element at coords. It panics if the rank or index is invalid.
func (a *Array[T]) At(coords ...int) T {
	if err := nditer.CheckCoords(a.layout(), coords); err != nil {
		panicIndex(coords, a.shape)
	}
	return a.data[nditer.Index(a.layout(), coords)]
}

// In-place: Set writes v at coords in place and returns a for chaining.
func (a *Array[T]) Set(coords []int, v T) *Array[T] {
	if err := nditer.CheckCoords(a.layout(), coords); err != nil {
		panicIndex(coords, a.shape)
	}
	a.data[nditer.Index(a.layout(), coords)] = v
	return a
}

// Pure: Idx returns the linear position of coords in the backing data.
func (a *Array[T]) Idx(coords ...int) int {
	if err := nditer.CheckCoords(a.layout(), coords); err != nil {
		panicIndex(coords, a.shape)
	}
	return nditer.Index(a.layout(), coords)
}
