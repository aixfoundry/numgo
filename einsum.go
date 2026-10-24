package numgo

import (
	"fmt"
	"slices"
	"strings"
	"unicode"

	"github.com/kaisawind/numgo/internal/nditer"
)

// Einsum evaluates the Einstein summation convention on one or more arrays.
//
// Grammar (numpy np.einsum subset):
//
//		subscript := operand ("," operand)* ("->" output)?
//		operand   := (letter | "...")*
//		output    := (letter | "...")*
//
//	  - Explicit output after "->"; or implicit (no "->"): labels occurring
//	    exactly once across all operands, sorted alphabetically, with "..."
//	    placed before the free labels when any operand has one ("ij,jk" is
//	    "ik"; "i...j" yields shape [ellipsis, i, j] like numpy).
//	  - "..." stands for the operand's remaining dimensions (explicit labels
//	    take axes from the front and back around it); the operands' ellipsis
//	    dimensions broadcast together, so "...ij,...jk->...ik" batches. An
//	    explicit output lacking "..." sums the broadcast tail away.
//	  - A label repeated within ONE operand takes the diagonal of its axes
//	    ("ii->i" main diagonal, bare "ii" the trace); "...ii" traces the last
//	    two axes per batch. A non-square diagonal takes the min dimension.
//	  - Output labels must each appear once and be present in some input.
//
// Unsupported relative to numpy: integer dtypes (Number excludes them); the
// grammar above is otherwise complete. Deliberate deviations: non-square
// diagonals take the min dimension (numpy einsum errors; np.diagonal agrees
// with us), an explicit output without "..." sums the ellipsis tail (numpy
// errors), and scalar results are one-element arrays (the package has no
// 0-d arrays).
//
// Pure: returns new data; operands are never written.
//
// With two or more operands the contraction runs as a left fold of dense
// matmuls (einsumMatmul in einsum_matmul.go, on dot.go's matmulKernel): each
// step flattens the pair's shared labels to K in sorted order — the same
// ascending order the generic path sums in, so results are bit-identical —
// with the left's remaining output labels as M and the right's as N.
// Anything the pattern cannot express falls back to the generic
// transpose-broadcast-multiply-sum path below.
func Einsum[T Number](subscript string, as ...*Array[T]) *Array[T] {
	inPart, outPart, explicit := strings.Cut(subscript, "->")
	specs := strings.Split(inPart, ",")
	if len(specs) != len(as) {
		panicEinsumOperands(len(specs), len(as))
	}

	ops := make([]*einOperand[T], len(as))
	occurs := map[rune]int{}
	anyEll := false
	for i, spec := range specs {
		pre, post, ell := parseSubscripts(spec, fmt.Sprintf("operand %d", i))
		ops[i] = newEinOperand(as[i], pre, post, ell)
		for _, l := range ops[i].seq {
			occurs[l]++
		}
		anyEll = anyEll || ell
	}

	// Pure-contraction fast path: rewrite the label graph as a chain of
	// matmuls; fall back to the generic path when a pair does not fit.
	if acc, ok := einsumMatmul(ops, occurs, outPart, explicit, anyEll); ok {
		return acc
	}

	// numpy's implicit output places the broadcast dims FIRST, then the free
	// labels sorted: parse output as [ellipsis, sorted free labels].
	var outPre, outPost []rune
	outEll := false
	if explicit {
		outPre, outPost, outEll = parseSubscripts(outPart, "output")
	} else {
		free := []rune{}
		for l, c := range occurs {
			if c == 1 {
				free = append(free, l)
			}
		}
		slices.Sort(free)
		outEll = anyEll
		outPost = free
	}
	outSet := map[rune]bool{}
	for _, l := range append(append([]rune{}, outPre...), outPost...) {
		if outSet[l] {
			panicEinsumOutputRepeat(l)
		}
		if occurs[l] == 0 {
			panicEinsumOutputLabel(l)
		}
		outSet[l] = true
	}
	if outEll && !anyEll {
		panicEinsumOutputEllipsis()
	}

	// Union of labels with broadcast dims: max across operands, requiring
	// equality or 1 per label (numpy rejects other conflicts).
	var union []rune
	dims := map[rune]int{}
	firstShape := map[rune][]int{}
	for _, op := range ops {
		for _, l := range op.labels {
			d, seen := dims[l]
			if !seen {
				union = append(union, l)
				dims[l] = op.dims[l]
				firstShape[l] = op.arr.Shape()
				continue
			}
			if op.dims[l] != d && op.dims[l] != 1 && d != 1 {
				panicBroadcast(firstShape[l], op.arr.Shape())
			}
			dims[l] = max(d, op.dims[l])
		}
	}
	slices.Sort(union)

	ellShapes := make([][]int, len(ops))
	for i, op := range ops {
		ellShapes[i] = op.ellShape
	}
	ellShape, _, err := nditer.Broadcast(ellShapes...)
	if err != nil {
		panic(err)
	}
	unionShape := make([]int, 0, len(union)+len(ellShape))
	for _, l := range union {
		unionShape = append(unionShape, dims[l])
	}
	unionShape = append(unionShape, ellShape...)

	// Build each operand's view over [union labels..., ellipsis dims...]:
	// missing labels become size-1 axes, and the operand's ellipsis axes sit
	// last so the whole view right-aligns against unionShape for a single
	// BroadcastTo.
	views := make([]*Array[T], len(ops))
	for i, op := range ops {
		pos := map[rune]int{}
		for k, l := range op.labels {
			pos[l] = k
		}
		// Target layout: union labels not in the operand first (size 1),
		// then the operand's labels in union order, then its ellipsis. The
		// trailing ellipsis axes stay rightmost so broadcasting the copies
		// right-aligns them exactly as the union requires.
		shape := make([]int, 0, len(union)+len(ellShape))
		st := make([]int, 0, len(union)+len(ellShape))
		for _, l := range union {
			if p, ok := pos[l]; ok {
				shape = append(shape, op.view.shape[p])
				st = append(st, op.view.strides[p])
			} else {
				shape = append(shape, 1)
				st = append(st, 0)
			}
		}
		for range len(ellShape) - len(op.ellShape) {
			shape = append(shape, 1)
			st = append(st, 0)
		}
		shape = append(shape, op.ellShape...)
		st = append(st, op.view.strides[len(op.labels):]...)
		views[i] = newArray(shape, st, op.view.offset, op.view.data).Copy()
	}

	// Mul's internal re-broadcast assumes C-order operand strides, so bake
	// each aligned view into contiguous storage before the multiply chain.
	acc := views[0].Copy()
	for _, v := range views[1:] {
		acc = acc.Mul(v.Copy())
	}

	// Sum away every label that is not in the output (and the ellipsis tail
	// when an explicit output omits "...").
	var sumAxes []int
	for k, l := range union {
		if !outSet[l] {
			sumAxes = append(sumAxes, k)
		}
	}
	if explicit && !outEll && len(ellShape) > 0 {
		sumAxes = append(sumAxes, len(union))
	}
	if len(sumAxes) > 0 {
		acc = acc.Sum(sumAxes...)
	}

	// Transpose the kept axes [sorted output labels..., ell] into the written
	// order: explicit outputs may interleave the ellipsis anywhere, implicit
	// outputs put it first (numpy).
	outSorted := append(append([]rune{}, outPre...), outPost...)
	slices.Sort(outSorted)
	posOf := map[rune]int{}
	for k, l := range outSorted {
		posOf[l] = k
	}
	perm := make([]int, 0, len(outSorted)+1)
	for _, l := range outPre {
		perm = append(perm, posOf[l])
	}
	if outEll && len(ellShape) > 0 {
		perm = append(perm, len(outSorted))
	}
	for _, l := range outPost {
		perm = append(perm, posOf[l])
	}
	identity := true
	for k, p := range perm {
		if p != k {
			identity = false
			break
		}
	}
	if !identity {
		acc = acc.Transpose(perm...)
	}
	return acc
}

// einOperand is one parsed operand: its labels in first-occurrence order,
// per-label dims after diagonal extraction, and a canonical view with axes
// ordered [labels..., ellipsis dims...].
type einOperand[T Number] struct {
	arr      *Array[T]
	seq      []rune // labels with repeats, in subscript order
	labels   []rune // distinct, first-occurrence order
	dims     map[rune]int
	view     *Array[T]
	ellShape []int
}

// newEinOperand builds the operand's canonical view: one axis per distinct
// label (repeated labels collapse to the min-dim diagonal using the first
// axis's stride) plus the ellipsis dimensions last. The layout is assembled
// directly via newArray — stride configuration only; the index formula stays
// in nditer.
func newEinOperand[T Number](a *Array[T], pre, post []rune, ell bool) *einOperand[T] {
	n := a.Ndim()
	seq := append(append([]rune{}, pre...), post...)
	if len(seq) > n || (!ell && len(seq) != n) {
		panicEinsumLabelCount(len(seq), n)
	}
	// Labels before "..." take axes from the front, labels after it from the
	// back; the ellipsis covers the middle axes in ascending order.
	axis := make([]int, len(seq))
	for k := range pre {
		axis[k] = k
	}
	for k := range post {
		axis[len(pre)+k] = n - len(post) + k
	}
	// A repeated label keeps its first occurrence's stride, so the view walks
	// the diagonal: element k of the axis reads a[first_k, other_k, ...].
	op := &einOperand[T]{arr: a, seq: seq, dims: map[rune]int{}}
	strides := map[rune]int{}
	for k, l := range seq {
		ax := axis[k]
		if prev, seen := op.dims[l]; seen {
			op.dims[l] = min(prev, a.shape[ax])
			strides[l] += a.strides[ax]
		} else {
			op.labels = append(op.labels, l)
			op.dims[l] = a.shape[ax]
			strides[l] = a.strides[ax]
		}
	}
	for ax := len(pre); ax < n-len(post); ax++ {
		op.ellShape = append(op.ellShape, a.shape[ax])
	}
	shape := make([]int, 0, len(op.labels)+len(op.ellShape))
	st := make([]int, 0, len(op.labels)+len(op.ellShape))
	for _, l := range op.labels {
		shape = append(shape, op.dims[l])
		st = append(st, strides[l])
	}
	shape = append(shape, op.ellShape...)
	for ax := len(pre); ax < n-len(post); ax++ {
		st = append(st, a.strides[ax])
	}
	op.view = newArray(shape, st, a.offset, a.data)
	return op
}

// parseSubscripts splits a subscript group into labels before and after its
// (single, optional) ellipsis. Whitespace is skipped; any other non-letter,
// a stray '.', or a second "..." panics.
func parseSubscripts(s, where string) (pre, post []rune, ell bool) {
	runes := []rune(s)
	for i := 0; i < len(runes); {
		r := runes[i]
		switch {
		case unicode.IsSpace(r):
			i++
		case r == '.':
			if ell || i+3 > len(runes) || runes[i+1] != '.' || runes[i+2] != '.' {
				panicEinsumBadChar('.', where)
			}
			ell = true
			i += 3
		case unicode.IsLetter(r):
			if ell {
				post = append(post, r)
			} else {
				pre = append(pre, r)
			}
			i++
		default:
			panicEinsumBadChar(r, where)
		}
	}
	return pre, post, ell
}
