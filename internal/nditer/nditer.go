// Package nditer is the single place where strided index arithmetic lives.
// Every array operation routes through it so stride bugs cannot spread.
package nditer

import (
	"fmt"
	"slices"
	"strings"
)

// Layout is the dtype-independent projection of an array: shape, element-unit
// strides and a base offset into the data slice.
type Layout struct {
	Shape, Strides []int
	Offset         int
}

// COrderStrides returns row-major strides in element units.
func COrderStrides(shape []int) []int {
	strides := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = acc
		acc *= shape[i]
	}
	return strides
}

// Size returns the product of shape. The empty shape has size 0 (no scalar
// arrays exist in this package yet).
func Size(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// IsContiguous reports whether l's strides are exactly C-order.
func IsContiguous(l *Layout) bool {
	if len(l.Shape) != len(l.Strides) {
		return false
	}
	want := COrderStrides(l.Shape)
	for i := range want {
		if l.Strides[i] != want[i] {
			return false
		}
	}
	return true
}

// contiguousSame reports whether l addresses exactly shape in C order with no
// broadcasting: equal shape and C-order strides.
func contiguousSame(l *Layout, shape []int) bool {
	return slices.Equal(l.Shape, shape) && IsContiguous(l)
}

// CheckCoords validates index rank and bounds. Negative indices are not
// supported (numpy semantics, but the Go API keeps it explicit).
func CheckCoords(l *Layout, coords []int) error {
	if len(coords) != len(l.Shape) {
		return fmt.Errorf("numgo: expected %d indices for a %d-d array, got %d",
			len(l.Shape), len(l.Shape), len(coords))
	}
	for i, c := range coords {
		if c < 0 || c >= l.Shape[i] {
			return fmt.Errorf("numgo: index %s is out of bounds for shape %s",
				shapeString(coords), shapeString(l.Shape))
		}
	}
	return nil
}

// Index returns the linear data position of coords. Callers must validate
// coords with CheckCoords first.
func Index(l *Layout, coords []int) int {
	pos := l.Offset
	for i, c := range coords {
		pos += c * l.Strides[i]
	}
	return pos
}

// Broadcast applies full numpy right-alignment. It returns the broadcast shape
// and one stride row per input, with 0 strides on broadcast dimensions. Unequal
// non-1 dimensions return an error whose text the root package re-panics.
func Broadcast(shapes ...[]int) (shape []int, strides [][]int, err error) {
	ndim := 0
	for _, s := range shapes {
		ndim = max(ndim, len(s))
	}
	shape = make([]int, ndim)
	for i := range shape {
		shape[i] = 1
	}
	// First pass: resolve each output dimension.
	for _, s := range shapes {
		lead := ndim - len(s)
		for i, d := range s {
			k := lead + i
			if d == 1 {
				continue
			}
			if shape[k] != 1 && shape[k] != d {
				var b strings.Builder
				b.WriteString("numgo: operands could not be broadcast together with shapes")
				for _, other := range shapes {
					b.WriteString(" " + shapeString(other))
				}
				return nil, nil, fmt.Errorf("%s", b.String())
			}
			shape[k] = d
		}
	}
	// Second pass: per-input strides, 0 where the input dim is absent or 1.
	strides = make([][]int, len(shapes))
	for j, s := range shapes {
		own := COrderStrides(s)
		row := make([]int, ndim)
		lead := ndim - len(s)
		for i, d := range s {
			k := lead + i
			if d != 1 {
				row[k] = own[i]
			}
		}
		strides[j] = row
	}
	return shape, strides, nil
}

// Slice derives the layout of a strided sub-view. starts, steps and shape are
// per-axis, already normalized by the caller.
func Slice(l *Layout, starts, steps, shape []int) *Layout {
	out := &Layout{
		Shape:   append([]int(nil), shape...),
		Strides: make([]int, len(steps)),
		Offset:  l.Offset,
	}
	for i, step := range steps {
		out.Strides[i] = l.Strides[i] * step
		out.Offset += starts[i] * l.Strides[i]
	}
	return out
}

// Permute reorders axes. axes must be a valid permutation of [0, ndim).
func Permute(l *Layout, axes []int) *Layout {
	out := &Layout{
		Shape:   make([]int, len(axes)),
		Strides: make([]int, len(axes)),
		Offset:  l.Offset,
	}
	for i, ax := range axes {
		out.Shape[i] = l.Shape[ax]
		out.Strides[i] = l.Strides[ax]
	}
	return out
}

// Zip iterates dst in C order, feeding fn successive slices of each operand.
// The block size fn receives per call is UNSPECIFIED: one element on strided
// paths, a whole contiguous run on the fast path. Callbacks must therefore
// iterate the passed slices rather than assume length 1. dst must already be
// broadcast-aligned; each src Layout must be aligned to dst.Shape.
func Zip[T any](dst *Layout, dstData []T, srcs []*Layout, srcData [][]T,
	fn func(dst []T, src [][]T)) {
	if Size(dst.Shape) == 0 {
		return
	}
	// Fast path: dst and every src fully C-contiguous over the same shape —
	// one callback covers everything with zero-copy slices.
	n := Size(dst.Shape)
	if contiguousSame(dst, dst.Shape) && len(dstData) >= n+dst.Offset {
		allContig := true
		for j, s := range srcs {
			if !contiguousSame(s, dst.Shape) || len(srcData[j]) < n+s.Offset {
				allContig = false
				break
			}
		}
		if allContig {
			rows := make([][]T, len(srcs))
			for j, s := range srcs {
				rows[j] = srcData[j][s.Offset : s.Offset+n]
			}
			fn(dstData[dst.Offset:dst.Offset+n], rows)
			return
		}
	}
	idx := make([]int, len(dst.Shape))
	src := make([][]T, len(srcs))
	for {
		dstPos := Index(dst, idx)
		for j, s := range srcs {
			p := Index(s, idx)
			src[j] = srcData[j][p : p+1]
		}
		fn(dstData[dstPos:dstPos+1], src)
		// Increment idx in C order.
		k := len(idx) - 1
		for k >= 0 {
			idx[k]++
			if idx[k] < dst.Shape[k] {
				break
			}
			idx[k] = 0
			k--
		}
		if k < 0 {
			return
		}
	}
}

// Reduce folds every element of data along axes into the kept-coordinate slot
// of out. Axes are normalized (negative allowed) then deduplicated and sorted.
// The caller prefills out with the identity value. fn may receive a whole
// contiguous run per call — the block size is UNSPECIFIED, like Zip.
func Reduce[T any](l *Layout, data []T, axes []int, out []T,
	fn func(acc *T, v T)) {
	if Size(l.Shape) == 0 {
		return
	}
	ndim := len(l.Shape)
	reduced := make([]bool, ndim)
	for _, ax := range axes {
		if ax < 0 {
			ax += ndim
		}
		if ax >= 0 && ax < ndim {
			reduced[ax] = true
		}
	}
	keptShape := make([]int, 0, ndim)
	for i := range l.Shape {
		if !reduced[i] {
			keptShape = append(keptShape, l.Shape[i])
		}
	}
	if Size(keptShape) == 0 && len(keptShape) > 0 {
		return
	}
	// Fast path: whole-array reduce — every axis folded and the layout
	// C-contiguous means data maps linearly onto out[0].
	if contiguousSame(l, l.Shape) && len(out) > 0 && Size(l.Shape) <= len(data)-l.Offset {
		whole := true
		for i := range reduced {
			if !reduced[i] {
				whole = false
				break
			}
		}
		if whole {
			n := Size(l.Shape)
			for i := l.Offset; i < l.Offset+n; i++ {
				fn(&out[0], data[i])
			}
			return
		}
	}
	// Fast path: last-axis reduce on contiguous input — each kept slot folds
	// the contiguous row data[row*last : (row+1)*last].
	if ndim > 1 && reduced[ndim-1] && contiguousSame(l, l.Shape) &&
		Size(l.Shape) <= len(data)-l.Offset {
		last := true
		for i := 0; i < ndim-1; i++ {
			if reduced[i] {
				last = false
				break
			}
		}
		if last {
			n := Size(l.Shape[:ndim-1])
			w := l.Shape[ndim-1]
			for row := 0; row < n && row < len(out); row++ {
				base := l.Offset + row*w
				for i := base; i < base+w; i++ {
					fn(&out[row], data[i])
				}
			}
			return
		}
	}
	idx := make([]int, ndim)
	for {
		slot := 0
		for i := range idx {
			if !reduced[i] {
				slot = slot*l.Shape[i] + idx[i]
			}
		}
		if slot < len(out) {
			fn(&out[slot], data[Index(l, idx)])
		}
		k := ndim - 1
		for k >= 0 {
			idx[k]++
			if idx[k] < l.Shape[k] {
				break
			}
			idx[k] = 0
			k--
		}
		if k < 0 {
			return
		}
	}
}

func shapeString(shape []int) string {
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = fmt.Sprint(d)
	}
	return "(" + strings.Join(parts, ",") + ")"
}

// ReduceAxis folds along a single axis, handing fn the C-order kept slot, the
// coordinate along axis and the element. It exists so argmax-style operations
// can report positions without doing index arithmetic themselves.
func ReduceAxis[T any](l *Layout, data []T, axis int, fn func(slot, coord int, v T)) {
	if Size(l.Shape) == 0 {
		return
	}
	ndim := len(l.Shape)
	if axis < 0 {
		axis += ndim
	}
	if axis < 0 || axis >= ndim {
		return
	}
	idx := make([]int, ndim)
	for {
		slot := 0
		for i := range idx {
			if i != axis {
				slot = slot*l.Shape[i] + idx[i]
			}
		}
		fn(slot, idx[axis], data[Index(l, idx)])
		k := ndim - 1
		for k >= 0 {
			idx[k]++
			if idx[k] < l.Shape[k] {
				break
			}
			idx[k] = 0
			k--
		}
		if k < 0 {
			return
		}
	}
}

// Accumulate folds along axes in C-order prefix style (cumsum semantics): out
// has the same element count as input; within each channel (all non-axes
// coords fixed), out[.., i_k, ..] holds the fold of data[.., 0..i_k, ..]. Axes
// are normalized (negative allowed) like Reduce. fn is called per element with
// a pointer to the folded-so-far accumulator.
func Accumulate[T any](l *Layout, data []T, axes []int, out []T,
	fn func(acc *T, v T)) {
	if Size(l.Shape) == 0 {
		return
	}
	ndim := len(l.Shape)
	folded := make([]bool, ndim)
	for _, ax := range axes {
		if ax < 0 {
			ax += ndim
		}
		if ax >= 0 && ax < ndim {
			folded[ax] = true
		}
	}
	// Fast path: fold along the last axis of contiguous input — each channel is
	// a contiguous row, so one accumulator per row walks it linearly. out is
	// indexed by the same linear position as data.
	if ndim > 1 && folded[ndim-1] && contiguousSame(l, l.Shape) {
		onlyLast := true
		for i := 0; i < ndim-1; i++ {
			if folded[i] {
				onlyLast = false
				break
			}
		}
		if onlyLast {
			n := Size(l.Shape[:ndim-1])
			w := l.Shape[ndim-1]
			if l.Offset+n*w <= len(out) {
				// acc lives outside the row loop so &acc does not escape per row.
				var acc T
				for row := 0; row < n; row++ {
					base := l.Offset + row*w
					acc = data[base]
					for i := base; i < base+w; i++ {
						if i > base {
							fn(&acc, data[i])
						}
						out[i] = acc
					}
				}
				return
			}
		}
	}
	idx := make([]int, ndim)
	prev := make([]int, ndim)
	// Odometer weights over the folded axes only: the prefix chain runs in
	// C-order restricted to the folded coordinates, so a channel's first
	// element is the one whose folded counter is zero.
	foldStride := make([]int, ndim)
	s := 1
	for i := ndim - 1; i >= 0; i-- {
		foldStride[i] = s
		if folded[i] {
			s *= l.Shape[i]
		}
	}
	for {
		// acc carries the fold from out at the predecessor coordinate (folded
		// counter minus one); counter zero means empty prefix.
		foldPos := 0
		for i := range idx {
			if folded[i] {
				foldPos += idx[i] * foldStride[i]
			}
			prev[i] = idx[i]
		}
		acc := data[Index(l, idx)]
		if foldPos > 0 {
			foldPos--
			for i := ndim - 1; i >= 0; i-- {
				if !folded[i] {
					continue
				}
				prev[i] = foldPos / foldStride[i]
				foldPos %= foldStride[i]
			}
			acc = out[Index(l, prev)]
			fn(&acc, data[Index(l, idx)])
		}
		out[Index(l, idx)] = acc
		// Advance idx as a full C-order odometer over every coordinate.
		k := len(idx) - 1
		for k >= 0 {
			idx[k]++
			if idx[k] < l.Shape[k] {
				break
			}
			idx[k] = 0
			k--
		}
		if k < 0 {
			return
		}
	}
}
