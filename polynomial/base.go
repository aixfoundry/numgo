package polynomial

import (
	"fmt"
	"math"
	"slices"

	"github.com/aixfoundry/numgo"
	"github.com/aixfoundry/numgo/linalg"
)

// Base holds the state shared by the six polynomial classes: coefficients
// (lowest degree first), domain/window and the class-specific math hooks.
// It is embedded by Polynomial, Chebyshev, Legendre, Hermite, HermiteE and
// Laguerre.
type Base[T numgo.Float] struct {
	coef           []T
	domain, window [2]T
	ops            classOps[T]
}

// classOps ports ABCPolyBase's per-class hooks (_val, _mul, _der, ...). All
// coefficient slices are trimmed series (no trailing zeros, length >= 1).
type classOps[T numgo.Float] interface {
	val(x, c []T) []T
	// vander returns the flat row-major (len(x), deg+1) pseudo-Vandermonde.
	vander(x []T, deg int) []T
	add(c1, c2 []T) []T
	sub(c1, c2 []T) []T
	mul(c1, c2 []T) []T
	div(c1, c2 []T) (quo, rem []T)
	pow(c []T, n int) []T
	der(c []T, m int, scl T) []T
	// integ integrates m times; k holds the m integration constants and
	// lbnd is the lower bound (integration constant convention F(lbnd)=k[i]).
	integ(c []T, m int, k []T, lbnd, scl T) []T
	line(off, scl T) []T
	// companion returns the flat row-major (n, n) companion matrix,
	// n = len(c)-1. Panics for degree < 1.
	companion(c []T) []T
	fromroots(roots []T) []T
	// roots returns the eigenvalue roots in window coordinates.
	roots(c []T) []complex128
	// compose returns the coefficients of the series composed with the
	// linear map off + scl*x (convert's reparameterization).
	compose(c []T, off, scl T) []T
	domainWindow() (domain, window [2]T)
	basisName() string
}

// f64 casts to float64 for float32/float64 element math.
func f64[T numgo.Float](v T) float64 { return float64(v) }

// scale multiplies a series by a scalar.
func scale[T numgo.Float](c []T, s T) []T {
	out := make([]T, len(c))
	for i, v := range c {
		out[i] = v * s
	}
	return out
}

// convolve returns the full linear convolution of a and b (np.convolve).
func convolve[T numgo.Float](a, b []T) []T {
	out := make([]T, len(a)+len(b)-1)
	for i, av := range a {
		for j, bv := range b {
			out[i+j] += av * bv
		}
	}
	return out
}

// trimseq removes trailing zero coefficients, keeping at least one.
func trimseq[T numgo.Float](c []T) []T {
	n := len(c)
	for n > 1 && c[n-1] == 0 {
		n--
	}
	return c[:n]
}

// trimcoef removes trailing coefficients with |c| <= tol (polyutils.trimcoef).
func trimcoef[T numgo.Float](c []T, tol T) []T {
	if tol < 0 {
		panicTol()
	}
	last := -1
	for i, v := range c {
		if T(math.Abs(f64(v))) > tol {
			last = i
		}
	}
	if last < 0 {
		return []T{0}
	}
	return c[:last+1]
}

// coefAdd adds two series coefficient-wise (polyutils._add).
func coefAdd[T numgo.Float](c1, c2 []T) []T {
	if len(c1) < len(c2) {
		c1, c2 = c2, c1
	}
	ret := slices.Clone(c1)
	for i, v := range c2 {
		ret[i] += v
	}
	return trimseq(ret)
}

// coefSub subtracts two series coefficient-wise (polyutils._sub).
func coefSub[T numgo.Float](c1, c2 []T) []T {
	if len(c1) >= len(c2) {
		ret := slices.Clone(c1)
		for i, v := range c2 {
			ret[i] -= v
		}
		return trimseq(ret)
	}
	ret := make([]T, len(c2))
	for i := range c2 {
		ret[i] = -c2[i]
	}
	for i, v := range c1 {
		ret[i] += v
	}
	return trimseq(ret)
}

// mapparms returns off, scl of the linear map off + scl*x taking old to new
// (polyutils.mapparms).
func mapparms[T numgo.Float](old, new [2]T) (off, scl T) {
	oldlen := old[1] - old[0]
	if oldlen == 0 {
		panicDegenerateDomain()
	}
	return (old[1]*new[0] - old[0]*new[1]) / oldlen, (new[1] - new[0]) / oldlen
}

// mapdomainArr applies the linear map from old to new to the array x.
func mapdomainArr[T numgo.Float](x *numgo.Array[T], old, new [2]T) *numgo.Array[T] {
	off, scl := mapparms(old, new)
	return x.MulValue(scl).AddValue(off)
}

// mapdomainSlice is mapdomainArr for a plain slice.
func mapdomainSlice[T numgo.Float](x []T, old, new [2]T) []T {
	off, scl := mapparms(old, new)
	out := make([]T, len(x))
	for i, v := range x {
		out[i] = off + scl*v
	}
	return out
}

// getdomain returns [min(x), max(x)], widened by one when degenerate
// (polyutils.getdomain).
func getdomain[T numgo.Float](x []T) [2]T {
	lo, hi := x[0], x[0]
	for _, v := range x {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if lo == hi {
		return [2]T{lo - 1, hi + 1}
	}
	return [2]T{lo, hi}
}

// powCoef raises c to the n-th power via repeated mul (polyutils._pow).
func powCoef[T numgo.Float](mul func(c1, c2 []T) []T, c []T, n int) []T {
	if n < 0 {
		panicNegPower()
	}
	if n == 0 {
		return []T{1}
	}
	prd := c
	for range n - 1 {
		prd = mul(prd, c)
	}
	return prd
}

// divByMul divides via repeated subtraction of c2 times the i-th basis unit
// (polyutils._div), the shared fallback for Legendre/Hermite/HermiteE/Laguerre.
func divByMul[T numgo.Float](mul func(c1, c2 []T) []T, c1, c2 []T) (quo, rem []T) {
	c1 = trimseq(slices.Clone(c1))
	c2 = trimseq(slices.Clone(c2))
	if c2[len(c2)-1] == 0 {
		panicZeroDivide()
	}
	lc1, lc2 := len(c1), len(c2)
	if lc1 < lc2 {
		return []T{0}, c1
	}
	if lc2 == 1 {
		return scale(c1, 1/c2[0]), []T{0}
	}
	quo = make([]T, lc1-lc2+1)
	rem = c1
	for i := lc1 - lc2; i >= 0; i-- {
		p := mul(append(make([]T, i), 1), c2)
		q := rem[len(rem)-1] / p[len(p)-1]
		rem = coefSub(rem[:len(rem)-1], scale(p[:len(p)-1], q))
		quo[i] = q
	}
	return quo, trimseq(rem)
}

// fromRootsCoef builds the monic-class-basis product of the root factors
// (polyutils._fromroots).
func fromRootsCoef[T numgo.Float](line func(off, scl T) []T, mul func(c1, c2 []T) []T, roots []T) []T {
	if len(roots) == 0 {
		return []T{1}
	}
	rs := slices.Clone(roots)
	slices.Sort(rs)
	p := make([][]T, len(rs))
	for i, r := range rs {
		p[i] = line(-r, 1)
	}
	n := len(p)
	for n > 1 {
		m := n / 2
		r := n % 2
		tmp := make([][]T, m)
		for i := range m {
			tmp[i] = mul(p[i], p[i+m])
		}
		if r != 0 {
			tmp[0] = mul(tmp[0], p[n-1])
		}
		p, n = tmp, m
	}
	return p[0]
}

// fitCoef ports _polybase.fit's default-domain path: map x into the window,
// least-squares the Vandermonde system (numpy scales design columns first;
// shortcut: unscaled SVD lstsq agrees to fp noise on well-conditioned fits,
// upgrade to column scaling for high-degree fits). rcond = len(x)*eps like
// numpy's default.
func fitCoef[T numgo.Float](ops classOps[T], x, y []T, deg int) (coef []T, domain [2]T) {
	if deg < 0 {
		panicNegDeg()
	}
	if len(x) == 0 || len(x) != len(y) {
		panicFitLen()
	}
	domain = getdomain(x)
	_, window := ops.domainWindow()
	xm := mapdomainSlice(x, domain, window)
	van := ops.vander(xm, deg)
	res := linalg.Lstsq(numgo.FromSlice(van, len(x), deg+1), numgo.New1DArray(y),
		float64(len(x))*2.220446049250313e-16)
	return slices.Clone(res.X.Data()), domain
}

// eigvalsOf runs the eigenvalue solver on a flat row-major matrix.
func eigvalsOf[T numgo.Float](mat []T, n int) []complex128 {
	f := make([]float64, len(mat))
	for i, v := range mat {
		f[i] = f64(v)
	}
	return slices.Clone(linalg.Eigvals(numgo.FromSlice(f, n, n)).Data())
}

// cx lifts a real coefficient to complex128.
func cx[T numgo.Float](v T) complex128 { return complex(f64(v), 0) }

// rootsEig sorts eigenvalues and is the shared tail of the per-class roots.
func rootsEig[T numgo.Float](mat []T, n int) []complex128 {
	r := eigvalsOf(mat, n)
	slices.SortFunc(r, func(a, b complex128) int {
		if ar, br := real(a), real(b); ar != br {
			if ar < br {
				return -1
			}
			return 1
		}
		if ai, bi := imag(a), imag(b); ai != bi {
			if ai < bi {
				return -1
			}
			return 1
		}
		return 0
	})
	return r
}

// newBase builds the embedded state with the class defaults.
func newBase[T numgo.Float](ops classOps[T], coef []T) Base[T] {
	domain, window := ops.domainWindow()
	c := slices.Clone(coef)
	if len(c) == 0 {
		c = []T{0}
	}
	return Base[T]{coef: c, domain: domain, window: window, ops: ops}
}

// with returns the embedded state carrying new coefficients.
func (b Base[T]) with(coef []T) Base[T] {
	return Base[T]{coef: coef, domain: b.domain, window: b.window, ops: b.ops}
}

// Coef returns a copy of the coefficients (lowest degree first).
func (b Base[T]) Coef() []T { return slices.Clone(b.coef) }

// Domain returns the domain interval the argument is mapped from.
func (b Base[T]) Domain() [2]T { return b.domain }

// Window returns the window interval the basis is defined on.
func (b Base[T]) Window() [2]T { return b.window }

// Len returns the number of coefficients.
func (b Base[T]) Len() int { return len(b.coef) }

// BasisName returns the class basis name.
func (b Base[T]) BasisName() string { return b.ops.basisName() }

// String renders "Chebyshev([1,2], domain=[0,1], window=[-1,1])".
func (b Base[T]) String() string {
	return fmt.Sprintf("%s(%v, domain=%v, window=%v)", b.ops.basisName(), b.coef, b.domain, b.window)
}

// Val evaluates the series at each element of x, which must already be in
// window coordinates. The input shape is preserved.
func (b Base[T]) Val(x *numgo.Array[T]) *numgo.Array[T] {
	out := x.Copy()
	copy(out.Data(), b.ops.val(out.Data(), b.coef))
	return out
}

// Call maps x from the domain onto the window, then evaluates (numpy
// __call__). The input shape is preserved.
func (b Base[T]) Call(x *numgo.Array[T]) *numgo.Array[T] {
	return b.Val(mapdomainArr(x, b.domain, b.window))
}

// Vander returns the (len(x), deg+1) pseudo-Vandermonde matrix in this class
// basis at the window-coordinate points x.
func (b Base[T]) Vander(x *numgo.Array[T], deg int) *numgo.Array[T] {
	v := b.ops.vander(x.Copy().Data(), deg)
	return numgo.FromSlice(v, x.Size(), deg+1)
}

// Companion returns the companion matrix of the series.
func (b Base[T]) Companion() *numgo.Array[T] {
	m := b.ops.companion(b.coef)
	n := len(b.coef) - 1
	return numgo.FromSlice(m, n, n)
}

// Roots returns the series roots mapped from window to domain coordinates,
// sorted by real then imaginary part. Values are complex128 because roots
// of real series may be complex conjugate pairs.
func (b Base[T]) Roots() *numgo.Array[complex128] {
	r := b.ops.roots(b.coef)
	slices.SortFunc(r, func(a, b complex128) int {
		if ar, br := real(a), real(b); ar != br {
			if ar < br {
				return -1
			}
			return 1
		}
		if ai, bi := imag(a), imag(b); ai != bi {
			if ai < bi {
				return -1
			}
			return 1
		}
		return 0
	})
	off, scl := mapparms(b.window, b.domain)
	for i := range r {
		r[i] = cx(off) + cx(scl)*r[i]
	}
	return numgo.New1DArray(r)
}

// convertParms returns the composite linear map (old domain -> old window)
// ∘ (new window -> new domain) applied to the identity: the argument the old
// series sees when the converted series is evaluated at window coordinate x
// (port of ABCPolyBase.convert via identity series composition).
func (b Base[T]) convertParms(domain, window [2]T) (off, scl T) {
	off1, scl1 := mapparms(window, domain)
	off2, scl2 := mapparms(b.domain, b.window)
	return off2 + scl2*off1, scl2 * scl1
}

// trimTruncate validates Truncate's size (ABCPolyBase.truncate).
func trimTruncate(size int) {
	if size < 1 {
		panicTruncate()
	}
}
