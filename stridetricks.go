package numgo

import (
	"slices"

	"github.com/kaisawind/numgo/internal/nditer"
)

// This file ports numpy.lib.stride_tricks: sliding_window_view, as_strided and
// broadcast_arrays. All results are VIEWS sharing the receiver's data; writes
// through them alias across slots exactly as in numpy.

// View: SlidingWindowView appends len(windowShape) window axes to a trimmed
// copy of a's shape: view.shape = trimmed + window, where trimmed[k] =
// shape[k] - (window[k] - 1). The window axes run over ALL of a's axes in
// order, so len(windowShape) must equal a.Ndim() (numpy's axis=None branch;
// there is no last-k-axes default in numpy either). The result shares data and
// the same element is reachable from many window slots, so writes alias —
// treat the result as read-only like numpy's default writeable=False.
func (a *Array[T]) SlidingWindowView(windowShape ...int) *Array[T] {
	for _, w := range windowShape {
		if w < 0 {
			panicWindowNegative()
		}
	}
	if len(windowShape) != a.Ndim() {
		panicWindowCount(len(windowShape), a.Ndim())
	}
	shape := make([]int, 0, 2*a.Ndim())
	for k, d := range a.shape {
		if d < windowShape[k] {
			panicWindowTooLarge()
		}
		shape = append(shape, d-(windowShape[k]-1))
	}
	shape = append(shape, windowShape...)
	strides := append(slices.Clone(a.strides), a.strides...)
	return newArray(shape, strides, a.offset, a.data)
}

// View: AsStrided builds a low-level view with explicit shape and strides over
// a's data starting at a's offset (numpy as_strided). SAFER than numpy: every
// reachable element position is validated against [0, len(a.data)) — numpy
// silently reads out of bounds, this panics. Negative strides are allowed;
// shape entries must be non-negative. The result shares data with no aliasing
// protection.
func (a *Array[T]) AsStrided(shape, strides []int) *Array[T] {
	if len(shape) != len(strides) {
		panicAsStridedDims()
	}
	for _, d := range shape {
		if d < 0 {
			panicAsStridedNegDim()
		}
	}
	// Reachability over offset + Σ iₖ·stridesₖ: each axis contributes
	// independently, so the extreme positions add per-axis extremes.
	minPos, maxPos := 0, 0
	for k := range shape {
		if shape[k] == 0 {
			continue
		}
		span := (shape[k] - 1) * strides[k]
		minPos += min(0, span)
		maxPos += max(0, span)
	}
	if a.offset+minPos < 0 || a.offset+maxPos >= len(a.data) {
		panicAsStridedBounds()
	}
	return newArray(slices.Clone(shape), slices.Clone(strides), a.offset, a.data)
}

// Pure-broadcast: BroadcastArrays returns views of all inputs expanded to
// their common broadcast shape (stride-0 dimensions, numpy broadcast_arrays).
// Incompatible shapes panic. The views share each input's data.
func BroadcastArrays[T Number](as ...*Array[T]) []*Array[T] {
	shapes := make([][]int, len(as))
	for i, a := range as {
		shapes[i] = a.shape
	}
	shape, strides, err := nditer.Broadcast(shapes...)
	if err != nil {
		panicBroadcastArrays(shapes)
	}
	out := make([]*Array[T], len(as))
	for i, a := range as {
		out[i] = newArray(slices.Clone(shape), strides[i], a.offset, a.data)
	}
	return out
}
