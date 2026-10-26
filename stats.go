package numgo

import (
	"math"
	"slices"
	"sort"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// This file holds the numpy lib function_base / histogram statistics:
// percentile-family quantiles, covariance, histograms, gradient and linear
// interpolation. Quantile results use numpy's "linear" method only (Hyndman &
// Fan type 7). NaN in a channel poisons that channel's quantile/median result
// (numpy masks slices containing NaN); use the Nan* variants to skip NaNs.

// channelValues collects the values of a along axes into one slice per kept
// slot: slots iterate in C order over the kept dimensions and each channel
// holds that slot's folded values in C order (the layout Sort and Argsort
// use). n is the per-channel element count.
func channelValues[T Number](a *Array[T], axes []int) (channels [][]T, n int) {
	norm := normalizeAxes(axes, a.Ndim())
	others := make([]int, 0, a.Ndim())
	for k := range a.shape {
		if !slices.Contains(norm, k) {
			others = append(others, k)
		}
	}
	perm := append(others, norm...)
	permShape := make([]int, len(perm))
	for i, k := range perm {
		permShape[i] = a.shape[k]
	}
	n = 1
	for _, ax := range norm {
		n *= a.shape[ax]
	}
	kept := make([]int, len(others))
	for i, k := range others {
		kept[i] = a.shape[k]
	}
	slots := nditer.Size(kept)
	if slots == 0 {
		// Reducing every axis collapses to one whole-array channel.
		slots = 1
	}
	channels = make([][]T, slots)
	if a.Size() == 0 {
		return channels, n
	}
	// Reading a C-order-raveled permuted view yields exactly slot-major,
	// coord-minor order.
	flat := make([]T, a.Size())
	src := a.Transpose(perm...)
	dst := &nditer.Layout{Shape: permShape, Strides: nditer.COrderStrides(permShape)}
	nditer.Zip(dst, flat, []*nditer.Layout{src.layout()}, [][]T{a.data},
		func(d []T, s [][]T) {
			for i := range d {
				d[i] = s[0][i]
			}
		})
	if n == 0 {
		return channels, 0
	}
	for i := range channels {
		channels[i] = flat[i*n : (i+1)*n]
	}
	return channels, n
}

// lerpParts mirrors numpy's _lerp per component: a + (b-a)*t, computed as
// b - (b-a)*(1-t) when t >= 0.5 for symmetric rounding.
func lerpParts(are, aim, bre, bim, t float64) (re, im float64) {
	if t >= 0.5 {
		return bre - (bre-are)*(1-t), bim - (bim-aim)*(1-t)
	}
	return are + (bre-are)*t, aim + (bim-aim)*t
}

// quantileSorted interpolates the q01-quantile (q01 in [0,1]) of a sorted
// slice with numpy's linear method. An empty slice yields NaN (numpy raises
// instead).
func quantileSorted[T Number](sorted []T, q01 float64) T {
	if len(sorted) == 0 {
		return nanScalar[T]()
	}
	pos := float64(len(sorted)-1) * q01
	lo := int(math.Floor(pos))
	if lo >= len(sorted)-1 {
		return sorted[len(sorted)-1]
	}
	gamma := pos - float64(lo)
	are, aim := parts(sorted[lo])
	bre, bim := parts(sorted[lo+1])
	re, im := lerpParts(are, aim, bre, bim, gamma)
	return scalarFromParts[T](re, im)
}

// quantileImpl computes the q01-quantile of each channel along the given
// axes; a channel containing NaN yields NaN.
func (a *Array[T]) quantileImpl(q01 float64, axes []int) []T {
	channels, n := channelValues(a, axes)
	out := make([]T, len(channels))
	for i, ch := range channels {
		sorted := slices.Clone(ch)
		slices.SortFunc(sorted, cmpNumber[T])
		if n == 0 || (n > 0 && isNaN(sorted[n-1])) {
			// NaN sorts last: a trailing NaN means the channel holds NaN.
			out[i] = nanScalar[T]()
			continue
		}
		out[i] = quantileSorted(sorted, q01)
	}
	return out
}

// Percentile computes the q-th percentile (q in [0, 100]) along axes
// with numpy's linear interpolation method — the only method implemented.
// The result is always float64: percentiles of complex values are not
// representable here (numpy raises TypeError). A channel containing NaN
// yields NaN for it.
func (a *Array[T]) Percentile(q float64, axes ...int) *Array[float64] {
	return a.quantileAsReal(q, 100, "Percentiles", axes)
}

// Quantile is Percentile with numpy's quantile scale: q in [0, 1].
func (a *Array[T]) Quantile(q float64, axes ...int) *Array[float64] {
	return a.quantileAsReal(q, 1, "Quantiles", axes)
}

// quantileAsReal runs the quantile at the given scale and returns float64.
func (a *Array[T]) quantileAsReal(q, scale float64, kind string, axes []int) *Array[float64] {
	if isComplexKind[T]() {
		panicComplexStat("Percentile/Quantile")
	}
	if math.IsNaN(q) || q < 0 || q > scale {
		panicQRange(kind, scale)
	}
	axes = a.axesOrAll(axes)
	norm := normalizeAxes(axes, a.Ndim())
	vals := a.quantileImpl(q/scale, norm)
	out := make([]float64, len(vals))
	for i, v := range vals {
		out[i] = mustFloat(v)
	}
	shape := reduceShape(a.shape, norm)
	return newArray(shape, nditer.COrderStrides(shape), 0, out)
}

// Median computes the median along axes (linear interpolation between
// the two middle values for even counts). A channel containing NaN yields NaN
// for it; an empty channel yields NaN (numpy raises).
func (a *Array[T]) Median(axes ...int) *Array[T] {
	axes = a.axesOrAll(axes)
	norm := normalizeAxes(axes, a.Ndim())
	vals := a.quantileImpl(0.5, norm)
	shape := reduceShape(a.shape, norm)
	return newArray(shape, nditer.COrderStrides(shape), 0, vals)
}

// toReal maps a flat slice to its real parts.
func toReal[T Number](vals []T) []float64 {
	out := make([]float64, len(vals))
	for i, v := range vals {
		out[i] = mustFloat(v)
	}
	return out
}

// covRows flattens each variable of a into a float64 slice: a 1-D array is a
// single variable with its elements as observations; a 2-D array is one
// variable per row (numpy cov rowvar=true).
func covRows[T Number](a *Array[T]) [][]float64 {
	if a.Ndim() <= 1 {
		return [][]float64{toReal(a.Ravel().Data())}
	}
	vals := toReal(a.Ravel().Data())
	cols := a.shape[1]
	rows := make([][]float64, 0, a.shape[0])
	for i := 0; i < a.shape[0]; i++ {
		rows = append(rows, vals[i*cols:(i+1)*cols])
	}
	return rows
}

// Cov returns the covariance matrix of the variables in a's rows, with b's
// rows appended as further variables when given — numpy cov with rowvar=true,
// bias=false (ddof=1). Every row must hold the same number of observations.
// Complex arrays panic: the float64 result cannot represent complex
// covariance (numpy's result is complex there).
func Cov[T Number](a, b *Array[T]) *Array[float64] {
	if isComplexKind[T]() {
		panicComplexStat("Cov/CorrCoef")
	}
	rows := covRows(a)
	if b != nil {
		rows = append(rows, covRows(b)...)
	}
	k := len(rows)
	n := 0
	if k > 0 {
		n = len(rows[0])
	}
	for _, r := range rows {
		if len(r) != n {
			panicSampleLength()
		}
	}
	means := make([]float64, k)
	for i, r := range rows {
		s := 0.0
		for _, v := range r {
			s += v
		}
		means[i] = s / float64(n)
	}
	fact := float64(n - 1)
	out := Zeros[float64](k, k)
	for i := range k {
		for j := range k {
			s := 0.0
			for t := 0; t < n; t++ {
				s += (rows[i][t] - means[i]) * (rows[j][t] - means[j])
			}
			out.Set([]int{i, j}, s/fact)
		}
	}
	return out
}

// CorrCoef returns the Pearson correlation coefficients of the variables in
// a's rows (b's rows appended, as with Cov): Cov normalized by the outer
// product of the per-variable standard deviations, clipped to [-1, 1].
func CorrCoef[T Number](a, b *Array[T]) *Array[float64] {
	c := Cov(a, b)
	k := c.Shape()[0]
	sd := make([]float64, k)
	for i := range sd {
		sd[i] = math.Sqrt(c.At(i, i))
	}
	out := Zeros[float64](k, k)
	for i := range k {
		for j := range k {
			v := c.At(i, j) / sd[i] / sd[j]
			out.Set([]int{i, j}, min(1, max(-1, v)))
		}
	}
	return out
}

// binEdges builds bins+1 equal-width edges over [min, max] of vals: the
// degenerate range is widened by 0.5 on both sides and an empty sample uses
// [0, 1] (numpy _get_outer_edges).
func binEdges(vals []float64, bins int) []float64 {
	first, last := 0.0, 1.0
	if len(vals) > 0 {
		first, last = math.Inf(1), math.Inf(-1)
		for _, v := range vals {
			first = min(first, v)
			last = max(last, v)
		}
		if first == last {
			first -= 0.5
			last += 0.5
		}
	}
	return linspaceEdges(first, last, bins)
}

// linspaceEdges spreads bins+1 edges over [first, last] inclusive.
func linspaceEdges(first, last float64, bins int) []float64 {
	edges := make([]float64, bins+1)
	for i := range edges {
		edges[i] = first + (last-first)*float64(i)/float64(bins)
	}
	return edges
}

// binIndex assigns v to a bin over edges (last edge inclusive, values outside
// dropped), porting numpy's histogram f_indices arithmetic. Returns -1 when v
// is out of range (NaN included).
func binIndex(v float64, edges []float64) int {
	bins := len(edges) - 1
	if !(v >= edges[0] && v <= edges[bins]) {
		return -1
	}
	f := (v - edges[0]) / (edges[bins] - edges[0]) * float64(bins)
	idx := int(f)
	if idx == bins {
		idx--
	}
	if v < edges[idx] {
		idx--
	}
	if idx != bins-1 && v >= edges[idx+1] {
		idx++
	}
	return idx
}

// HistogramBinEdges returns the bins+1 equal-width edges numpy's
// histogram would use: linspace over [min, max] of the raveled values,
// widened by 0.5 on both sides when min == max and [0, 1] when a is empty.
func (a *Array[T]) HistogramBinEdges(bins int) *Array[float64] {
	if isComplexKind[T]() {
		panicComplexStat("Histogram")
	}
	if bins < 1 {
		panicHistBins()
	}
	edges := binEdges(toReal(a.Ravel().Data()), bins)
	shape := []int{len(edges)}
	return newArray(shape, nditer.COrderStrides(shape), 0, edges)
}

// Histogram counts the raveled values of a into bins equal-width bins
// over [min, max] — numpy's histogram defaults: the last edge is inclusive
// and out-of-range values (NaN included) are dropped. Returns the counts then
// the edges.
func (a *Array[T]) Histogram(bins int) (*Array[float64], *Array[float64]) {
	edges := a.HistogramBinEdges(bins)
	counts := make([]float64, bins)
	for _, v := range a.Ravel().Data() {
		if idx := binIndex(mustFloat(v), edges.data); idx >= 0 {
			counts[idx]++
		}
	}
	shape := []int{bins}
	return newArray(shape, nditer.COrderStrides(shape), 0, counts), edges
}

// HistogramDD counts a's points into bins[d] equal-width bins per
// dimension: a's last axis holds the D coordinates of each point (numpy's
// sample of shape (N, D) with the points along the leading axes). Optional
// ranges override the autodetected [min, max] per dimension. Returns the
// counts array of shape bins and one edges array per dimension.
func (a *Array[T]) HistogramDD(bins []int, ranges ...[2]float64) (*Array[float64], []*Array[float64]) {
	if isComplexKind[T]() {
		panicComplexStat("Histogram")
	}
	d := 0
	if a.Ndim() > 0 {
		d = a.shape[a.Ndim()-1]
	}
	if len(bins) != d {
		panicRanges(len(bins), d)
	}
	if len(ranges) > 0 && len(ranges) != d {
		panicRanges(len(ranges), d)
	}
	for _, b := range bins {
		if b < 1 {
			panicHistBins()
		}
	}
	pts := a.Shape()[:max(a.Ndim()-1, 0)]
	n := nditer.Size(pts)
	coords := make([]int, a.Ndim())
	cols := make([][]float64, d)
	for j := range cols {
		col := make([]float64, n)
		for p := range n {
			coords[len(coords)-1] = j
			col[p] = mustFloat(a.At(coords...))
			for k := len(coords) - 2; k >= 0; k-- {
				coords[k]++
				if coords[k] < pts[k] {
					break
				}
				coords[k] = 0
			}
		}
		cols[j] = col
	}
	edges := make([][]float64, d)
	edgeArrs := make([]*Array[float64], d)
	for j := 0; j < d; j++ {
		if j < len(ranges) {
			// Supplied range, widened like the autodetected degenerate one.
			lo, hi := ranges[j][0], ranges[j][1]
			if lo == hi {
				lo -= 0.5
				hi += 0.5
			}
			edges[j] = linspaceEdges(lo, hi, bins[j])
		} else {
			edges[j] = binEdges(cols[j], bins[j])
		}
		shape := []int{len(edges[j])}
		edgeArrs[j] = newArray(shape, nditer.COrderStrides(shape), 0, edges[j])
	}
	hist := Zeros[float64](bins...)
	point := make([]int, d)
	for p := range n {
		ok := true
		for j := 0; j < d; j++ {
			idx := binIndex(cols[j][p], edges[j])
			if idx < 0 {
				ok = false
				break
			}
			point[j] = idx
		}
		if ok {
			hist.Set(point, hist.At(point...)+1)
		}
	}
	return hist, edgeArrs
}

// Histogram2D histograms paired samples: returns the counts H (x bins × y
// bins, H[i][j] counts x in bin i and y in bin j) then the x and y edges.
// x and y are read flattened and must have equal length; defaults follow
// numpy histogram2d (range min..max per axis, last edge inclusive).
func Histogram2D[T Number](x, y *Array[T], bins int) (*Array[float64], *Array[float64], *Array[float64]) {
	xv, yv := toReal(x.Ravel().Data()), toReal(y.Ravel().Data())
	if len(xv) != len(yv) {
		panicSampleLength()
	}
	hist, edges := xySample(xv, yv).HistogramDD([]int{bins, bins})
	return hist, edges[0], edges[1]
}

// xySample stacks two flat coordinate lists as an (n, 2) point array.
func xySample(xv, yv []float64) *Array[float64] {
	data := make([]float64, 2*len(xv))
	for i := range xv {
		data[2*i] = xv[i]
		data[2*i+1] = yv[i]
	}
	return FromSlice(data, len(xv), 2)
}

// Gradient returns one float64 array per axis with numpy's default
// gradient: second-order central differences in the interior and first-order
// one-sided differences at the edges, unit spacing. Every axis must have at
// least two elements (numpy raises for fewer).
func (a *Array[T]) Gradient() []*Array[float64] {
	if isComplexKind[T]() {
		panicComplexStat("Gradient")
	}
	outs := make([]*Array[float64], a.Ndim())
	for ax := range a.shape {
		if a.shape[ax] < 2 {
			panicGradient()
		}
		n := a.shape[ax]
		outs[ax] = FromFunction(a.Shape(), func(coords ...int) float64 {
			at := func(k int) float64 {
				src := slices.Clone(coords)
				src[ax] = k
				return mustFloat(a.At(src...))
			}
			i := coords[ax]
			switch {
			case i == 0:
				return at(1) - at(0)
			case i == n-1:
				return at(n-1) - at(n-2)
			default:
				return (at(i+1) - at(i-1)) / 2
			}
		})
	}
	return outs
}

// Interp linearly interpolates fp (sampled at the increasing abscissae xp) at
// each point of x: values below xp[0] clamp to fp[0] and above xp[-1] to
// fp[-1] (numpy interp with left/right defaulted to the endpoints). x keeps
// its shape; xp and fp are read flattened, must have equal length, and x/xp
// compare by real part; fp may be complex. xp with duplicate abscissae
// resolves to the last of them, as numpy's search does.
func Interp[T Number](x, xp, fp *Array[T]) *Array[T] {
	xpn, fpn := xp.Ravel().Data(), fp.Ravel().Data()
	if len(xpn) != len(fpn) {
		panicSampleLength()
	}
	if len(xpn) == 0 {
		panicEmptyXP()
	}
	out := make([]T, x.Size())
	for i, xv := range x.Ravel().Data() {
		xr := mustFloat(xv)
		switch {
		case math.IsNaN(xr):
			out[i] = nanScalar[T]()
		case xr < mustFloat(xpn[0]):
			out[i] = fpn[0]
		case xr >= mustFloat(xpn[len(xpn)-1]):
			out[i] = fpn[len(xpn)-1]
		default:
			// First abscissa strictly greater than xr: the segment left
			// of it is the interpolating one, and duplicates of xr sit
			// inside it, resolving to the last one.
			right := sort.Search(len(xpn), func(j int) bool { return xr < mustFloat(xpn[j]) })
			x0, x1 := mustFloat(xpn[right-1]), mustFloat(xpn[right])
			t := (xr - x0) / (x1 - x0)
			f0re, f0im := parts(fpn[right-1])
			f1re, f1im := parts(fpn[right])
			re, im := lerpParts(f0re, f0im, f1re, f1im, t)
			out[i] = scalarFromParts[T](re, im)
		}
	}
	shape := slices.Clone(x.shape)
	return newArray(shape, nditer.COrderStrides(shape), 0, out)
}
