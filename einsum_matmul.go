package numgo

import "slices"

// einsum-matmul.go: Einsum's matmul fast path. Einsum itself stays in
// einsum.go; this file rewrites the parsed operands as a left fold of dense
// matmuls (dot.go's matmulKernel) and falls back (ok=false) whenever a pair
// does not fit the pure-contraction pattern.

// einsumMatmul rewrites the einsum as a left fold of matmuls over the parsed
// operands. Every step must contract at least one label; outSet is the set of
// labels surviving to the output. Returns ok=false for anything the pattern
// cannot express (ellipsis in the output or implicit with ellipsis, an output
// label missing from a contracting pair, size-1 free-label broadcasts, pure
// outer products) — the generic path then produces the result or numpy's
// error.
func einsumMatmul[T Number](ops []*einOperand[T], occurs map[rune]int, outPart string, explicit, anyEll bool) (*Array[T], bool) {
	if len(ops) < 2 {
		return nil, false
	}
	// Output label set. Ellipsis-in-output and implicit-ellipsis orderings
	// need the generic path's transpose machinery.
	var outSet map[rune]bool
	if explicit {
		pre, post, ell := parseSubscripts(outPart, "output")
		if ell {
			return nil, false
		}
		outSet = map[rune]bool{}
		for _, l := range append(append([]rune{}, pre...), post...) {
			outSet[l] = true
		}
	} else {
		if anyEll {
			return nil, false
		}
		outSet = map[rune]bool{}
		for l, c := range occurs {
			if c == 1 {
				outSet[l] = true
			}
		}
	}
	fold := &einFold[T]{
		arr:      ops[0].view.Copy(),
		labels:   append([]rune{}, ops[0].labels...),
		dims:     ops[0].dims,
		ellShape: append([]int(nil), ops[0].ellShape...),
	}
	for _, op := range ops[1:] {
		next, nextEll, ok := einMatmulPair(fold, op, outSet)
		if !ok {
			return nil, false
		}
		fold = &einFold[T]{
			arr:      next,
			labels:   survivingLabels(fold.labels, op.labels, outSet),
			dims:     nextDims(fold, op, outSet),
			ellShape: nextEll,
		}
	}
	// fold.arr is [sorted resLabels..., ell...]. The matmul path only claims
	// outputs whose kept labels are emitted in sorted order with no ellipsis
	// (checked above) — exactly the shape it already has.
	return fold.arr, true
}

// einFold is the matmul fast-path's running accumulator: the contracted array
// so far as a C-contiguous [sorted labels..., ell...] block plus its labels.
type einFold[T Number] struct {
	arr      *Array[T]
	labels   []rune // sorted, distinct
	dims     map[rune]int
	ellShape []int
}

// survivingLabels lists fold ∪ op labels kept by the output, sorted.
func survivingLabels(l1, l2 []rune, outSet map[rune]bool) []rune {
	seen := map[rune]bool{}
	var out []rune
	for _, l := range append(append([]rune{}, l1...), l2...) {
		if outSet[l] && !seen[l] {
			seen[l] = true
			out = append(out, l)
		}
	}
	slices.Sort(out)
	return out
}

// nextDims merges the pair's dims for surviving labels (max = broadcast win).
func nextDims[T Number](fold *einFold[T], op *einOperand[T], outSet map[rune]bool) map[rune]int {
	dims := map[rune]int{}
	for _, l := range append(append([]rune{}, fold.labels...), op.labels...) {
		if !outSet[l] {
			continue
		}
		d, has := fold.dims[l]
		if dOp, hasOp := op.dims[l]; hasOp && (!has || dOp > d) {
			d = dOp
		}
		dims[l] = d
	}
	return dims
}

// einMatmulPair contracts fold (left) with op (right) via matmulKernel:
// shared non-output labels flatten to K (sorted — the same ascending label
// order the generic path sums in, so results are bit-identical), left-only
// output labels to M, op-only output labels to N; the result is
// [sorted kept labels..., ell...].
func einMatmulPair[T Number](fold *einFold[T], op *einOperand[T], outSet map[rune]bool) (*Array[T], []int, bool) {
	// Every label must be output or shared: a non-output label would need a
	// post-matmul sum (generic path's job), a size-1 free-label broadcast an
	// extra expansion copy.
	var kLabels, mLabels, nLabels []rune
	for _, l := range fold.labels {
		if slices.Contains(op.labels, l) {
			// A shared label that survives to the output is an element-wise
			// (Hadamard) axis, not a contraction — the matmul pattern cannot
			// express that mix; fall back.
			if outSet[l] {
				return nil, nil, false
			}
			kLabels = append(kLabels, l)
		} else if outSet[l] {
			mLabels = append(mLabels, l)
		} else {
			return nil, nil, false
		}
	}
	for _, l := range op.labels {
		if !slices.Contains(fold.labels, l) {
			if outSet[l] {
				nLabels = append(nLabels, l)
			} else {
				return nil, nil, false
			}
		}
	}
	if len(kLabels) == 0 {
		return nil, nil, false // outer product: generic path
	}
	for _, l := range kLabels {
		if dl, dop := fold.dims[l], op.dims[l]; dl != dop && dl != 1 && dop != 1 {
			return nil, nil, false // broadcast error: generic path panics
		}
	}
	for _, l := range mLabels {
		if opD, ok := op.dims[l]; ok && opD != fold.dims[l] {
			return nil, nil, false
		}
	}
	for _, l := range nLabels {
		if fD, ok := fold.dims[l]; ok && fD != op.dims[l] {
			return nil, nil, false
		}
	}
	// Ellipsis dims broadcast right-aligned; mismatches beyond size-1 fall
	// back so the generic path raises numpy's error.
	ellRank := max(len(fold.ellShape), len(op.ellShape))
	resEll := make([]int, ellRank)
	for i := range ellRank {
		dl, dop := 1, 1
		if i < len(fold.ellShape) {
			dl = fold.ellShape[i]
		}
		if i < len(op.ellShape) {
			dop = op.ellShape[i]
		}
		if dl != dop && dl != 1 && dop != 1 {
			return nil, nil, false
		}
		resEll[i] = max(dl, dop)
	}
	K := 1
	for _, l := range kLabels {
		K *= max(fold.dims[l], op.dims[l])
	}
	M := 1
	for _, l := range mLabels {
		M *= max(fold.dims[l], op.dims[l])
	}
	N := 1
	for _, l := range nLabels {
		N *= max(fold.dims[l], op.dims[l])
	}
	// Dense operands: (resEll..., M, K) and (resEll..., K, N), C-contiguous;
	// empty nLabels yields an N=1 column operand.
	av := einDenseSide(fold.arr, fold.labels, fold.ellShape, mLabels, kLabels, fold.dims, op.dims, resEll)
	bv := einDenseSide(op.view, op.labels, op.ellShape, kLabels, nLabels, op.dims, fold.dims, resEll)
	out := Zeros[T](append(append([]int{}, resEll...), M, N)...)
	aData := av.data[av.offset:]
	bData := bv.data[bv.offset:]
	dData := out.data
	dstLeads := out.strides[:len(resEll)]
	odometer(resEll, func(bc []int) {
		abase, bbase, obase := 0, 0, 0
		for d, c := range bc {
			abase += c * av.strides[d]
			bbase += c * bv.strides[d]
			obase += c * dstLeads[d]
		}
		matmulKernel(aData[abase:], bData[bbase:], dData[obase:], M, K, N)
	})
	// out is [batch..., M, N]; un-flatten M/N to their label axes and
	// transpose [batch..., m..., n...] -> [sorted kept labels..., batch...].
	// N==1 with no nLabels is the dropped column vector axis: nLabels empty
	// means flat has no axis for it and Reshape drops the size-1 tail.
	pd := pairDims(fold, op)
	flat := append(append([]int{}, resEll...), labelSizes(mLabels, pd)...)
	flat = append(flat, labelSizes(nLabels, pd)...)
	if len(flat) == 0 {
		// No labels and no ellipsis: the result is a scalar in a 1-element
		// 1-D array (this package has no 0-d arrays).
		return newArray([]int{1}, []int{1}, 0, out.data), resEll, true
	}
	res := out.Reshape(flat...)
	kept := survivingLabels(mLabels, nLabels, unionSet(mLabels, nLabels))
	perm := make([]int, 0, len(flat))
	for _, l := range kept {
		perm = append(perm, axisOfLabel(l, mLabels, nLabels, len(resEll)))
	}
	for i := range len(resEll) {
		perm = append(perm, i)
	}
	if !isIdentityPerm(perm) {
		res = res.Transpose(perm...)
	}
	return newArray(res.shape, res.strides, res.offset, res.data), resEll, true
}

// einDenseSide builds a C-contiguous (resEll..., rows..., cols...) block from
// src's [labels..., ell] layout: permute to [ell..., rows..., cols...],
// left-pad the batch rank with size-1 axes, broadcast to resEll and Copy.
// rowDims/otherDims resolve each shared (K) label's broadcast-winner size.
func einDenseSide[T Number](src *Array[T], labels []rune, ellShape []int,
	rowLabels, colLabels []rune, ownDims, otherDims map[rune]int, resEll []int) *Array[T] {
	ellRank := len(resEll)
	perm := make([]int, 0, len(labels)+len(ellShape))
	for i := range len(ellShape) {
		perm = append(perm, len(labels)+i)
	}
	for _, l := range rowLabels {
		perm = append(perm, indexOfRune(labels, l))
	}
	for _, l := range colLabels {
		perm = append(perm, indexOfRune(labels, l))
	}
	v := src.Transpose(perm...)
	shape := make([]int, 0, ellRank+v.Ndim())
	for range ellRank - len(ellShape) {
		shape = append(shape, 1)
	}
	shape = append(shape, v.Shape()...)
	v = v.Reshape(shape...)
	v = v.BroadcastTo(append(append([]int{}, resEll...), v.Shape()[ellRank:]...)...).Copy()
	// Shared K labels keep the broadcast-winner size; reshape now that the
	// copy baked the expanded dims.
	cols := make([]int, len(colLabels))
	for i, l := range colLabels {
		cols[i] = max(ownDims[l], otherDims[l])
	}
	rows := make([]int, len(rowLabels))
	for i, l := range rowLabels {
		rows[i] = max(ownDims[l], otherDims[l])
	}
	return v.Reshape(append(append(append([]int{}, resEll...), rows...), cols...)...)
}

// pairDims merges the fold and op dim maps (max = broadcast winner).
func pairDims[T Number](fold *einFold[T], op *einOperand[T]) map[rune]int {
	dims := make(map[rune]int, len(fold.dims)+len(op.dims))
	for l, d := range fold.dims {
		dims[l] = d
	}
	for l, d := range op.dims {
		if d > dims[l] {
			dims[l] = d
		}
	}
	return dims
}

// labelSizes maps labels to their dims in order.
func labelSizes(labels []rune, dims map[rune]int) []int {
	out := make([]int, len(labels))
	for i, l := range labels {
		out[i] = dims[l]
	}
	return out
}

func unionSet(groups ...[]rune) map[rune]bool {
	s := map[rune]bool{}
	for _, g := range groups {
		for _, l := range g {
			s[l] = true
		}
	}
	return s
}

func axisOfLabel(l rune, mLabels, nLabels []rune, ellRank int) int {
	for i, c := range mLabels {
		if c == l {
			return ellRank + i
		}
	}
	for i, c := range nLabels {
		if c == l {
			return ellRank + len(mLabels) + i
		}
	}
	return -1
}

func isIdentityPerm(perm []int) bool {
	for i, p := range perm {
		if p != i {
			return false
		}
	}
	return true
}

func indexOfRune(s []rune, r rune) int {
	for i, c := range s {
		if c == r {
			return i
		}
	}
	return -1
}
