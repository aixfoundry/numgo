package polynomial

import (
	"slices"

	"github.com/aixfoundry/numgo"
)

// Polynomial is the power-basis series c[0] + c[1]*x + ... (numpy
// polynomial.polynomial, class Polynomial). Default domain and window [-1,1].
type Polynomial[T numgo.Float] struct{ Base[T] }

// polyOps ports polynomial.polynomial's coefficient kernels.
type polyOps[T numgo.Float] struct{}

// NewPolynomial builds a power-basis series from coefficients, lowest degree
// first. The coefficients are copied.
func NewPolynomial[T numgo.Float](coef []T) *Polynomial[T] {
	return &Polynomial[T]{newBase(polyOps[T]{}, coef)}
}

// NewPolynomialDomain is NewPolynomial with an explicit domain and window.
func NewPolynomialDomain[T numgo.Float](coef []T, domain, window [2]T) *Polynomial[T] {
	p := NewPolynomial(coef)
	p.domain, p.window = domain, window
	return p
}

// NewPolynomialFromRoots builds the monic polynomial with the given roots
// (numpy Polynomial.fromroots with default domain).
func NewPolynomialFromRoots[T numgo.Float](roots []T) *Polynomial[T] {
	return &Polynomial[T]{newBaseFromRoots(polyOps[T]{}, roots)}
}

// newBaseFromRoots shares the fromroots constructor path of all classes.
// numpy's classmethod maps the roots from the (default) domain onto the
// window and divides by scl^deg before building the raw series.
func newBaseFromRoots[T numgo.Float](ops classOps[T], roots []T) Base[T] {
	domain, window := ops.domainWindow()
	off, scl := mapparms(domain, window)
	r := make([]T, len(roots))
	for i, v := range roots {
		r[i] = off + scl*v
	}
	return newBase(ops, scale(fromRootsCoef(ops.line, ops.mul, r), 1/powAll(scl, len(roots))))
}

// powAll returns s^n (s^0 = 1).
func powAll[T numgo.Float](s T, n int) T {
	p := T(1)
	for range n {
		p *= s
	}
	return p
}

func (p *Polynomial[T]) derived(coef []T) *Polynomial[T] { return &Polynomial[T]{p.Base.with(coef)} }

func (polyOps[T]) domainWindow() ([2]T, [2]T) { return [2]T{-1, 1}, [2]T{-1, 1} }
func (polyOps[T]) basisName() string          { return "Polynomial" }

// val is polyval: Horner's rule.
func (polyOps[T]) val(x, c []T) []T {
	out := make([]T, len(x))
	for i, xv := range x {
		c0 := c[len(c)-1]
		for j := len(c) - 2; j >= 0; j-- {
			c0 = c[j] + c0*xv
		}
		out[i] = c0
	}
	return out
}

// vander is polyvander: columns 1, x, x^2, ...
func (polyOps[T]) vander(x []T, deg int) []T {
	if deg < 0 {
		panicNegDeg()
	}
	dof := deg + 1
	v := make([]T, len(x)*dof)
	for i, xv := range x {
		v[i*dof] = 1
		for d := 1; d <= deg; d++ {
			v[i*dof+d] = v[i*dof+d-1] * xv
		}
	}
	return v
}

func (polyOps[T]) add(c1, c2 []T) []T { return coefAdd(c1, c2) }
func (polyOps[T]) sub(c1, c2 []T) []T { return coefSub(c1, c2) }

// mul is polymul: coefficient convolution.
func (z polyOps[T]) mul(c1, c2 []T) []T { return trimseq(convolve(c1, c2)) }

// div is polydiv: in-place synthetic division on the high coefficients.
func (z polyOps[T]) div(c1, c2 []T) (quo, rem []T) {
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
	dlen := lc1 - lc2
	scl := c2[lc2-1]
	c2 = scale(c2[:lc2-1], 1/scl)
	dc1 := slices.Clone(c1)
	i, j := dlen, lc1-1
	for i >= 0 {
		for k := range c2 {
			dc1[i+k] -= c2[k] * dc1[j]
		}
		i--
		j--
	}
	return scale(dc1[j+1:], 1/scl), trimseq(dc1[:j+1])
}

func (z polyOps[T]) pow(c []T, n int) []T { return powCoef(z.mul, c, n) }

// der is polyder: c'[j] = (j+1)*c[j+1].
func (polyOps[T]) der(c []T, m int, scl T) []T {
	c = slices.Clone(c)
	n := len(c)
	if m == 0 {
		return c
	}
	if m >= n {
		return []T{0}
	}
	for range m {
		n--
		for j := 1; j <= n; j++ {
			c[j] *= scl
		}
		dst := make([]T, n)
		for j := n; j >= 1; j-- {
			dst[j-1] = T(j) * c[j]
		}
		c = dst
	}
	return c
}

// integ is polyint with lbnd folded into the constant term.
func (z polyOps[T]) integ(c []T, m int, k []T, lbnd, scl T) []T {
	c = slices.Clone(c)
	for i := range m {
		n := len(c)
		for j := range c {
			c[j] *= scl
		}
		if n == 1 && c[0] == 0 {
			c[0] += k[i]
			continue
		}
		tmp := make([]T, n+1)
		tmp[1] = c[0]
		for j := 1; j < n; j++ {
			tmp[j+1] = c[j] / T(j+1)
		}
		tmp[0] += k[i] - z.val([]T{lbnd}, tmp)[0]
		c = tmp
	}
	return c
}

// line is polyline: off + scl*x, [off] when scl == 0.
func (polyOps[T]) line(off, scl T) []T {
	if scl != 0 {
		return []T{off, scl}
	}
	return []T{off}
}

// companion is polycompanion: ones on the subdiagonal, last column
// -c[:-1]/c[-1].
func (polyOps[T]) companion(c []T) []T {
	c = trimseq(slices.Clone(c))
	n := len(c) - 1
	if n < 1 {
		panicDegAtLeast1()
	}
	if n == 1 {
		return []T{-c[0] / c[1]}
	}
	mat := make([]T, n*n)
	for k := range n - 1 {
		mat[(k+1)*n+k] = 1
	}
	for i := range n {
		mat[i*n+n-1] -= c[i] / c[n]
	}
	return mat
}

func (z polyOps[T]) fromroots(roots []T) []T { return fromRootsCoef(z.line, z.mul, roots) }

// roots is polyroots: eigenvalues of the companion matrix.
func (z polyOps[T]) roots(c []T) []complex128 {
	c = trimseq(slices.Clone(c))
	switch len(c) {
	case 0, 1:
		return nil
	case 2:
		return []complex128{-cx(c[0] / c[1])}
	}
	return rootsEig(z.companion(c), len(c)-1)
}

// compose substitutes off + scl*x into the power-basis series by Horner.
func (z polyOps[T]) compose(c []T, off, scl T) []T {
	L := z.line(off, scl)
	acc := []T{c[len(c)-1]}
	for i := len(c) - 2; i >= 0; i-- {
		acc = coefAdd(convolve(acc, L), []T{c[i]})
	}
	return trimseq(acc)
}

// Add returns p + o in the power basis.
func (p *Polynomial[T]) Add(o *Polynomial[T]) *Polynomial[T] {
	return p.derived(coefAdd(p.coef, o.coef))
}

// Sub returns p - o in the power basis.
func (p *Polynomial[T]) Sub(o *Polynomial[T]) *Polynomial[T] {
	return p.derived(coefSub(p.coef, o.coef))
}

// Mul returns the product p * o.
func (p *Polynomial[T]) Mul(o *Polynomial[T]) *Polynomial[T] {
	return p.derived(p.ops.mul(p.coef, o.coef))
}

// Div returns the quotient and remainder of p / o (numpy divmod).
func (p *Polynomial[T]) Div(o *Polynomial[T]) (*Polynomial[T], *Polynomial[T]) {
	q, r := p.ops.div(p.coef, o.coef)
	return p.derived(q), p.derived(r)
}

// Pow raises the series to the n-th power.
func (p *Polynomial[T]) Pow(n int) *Polynomial[T] { return p.derived(p.ops.pow(p.coef, n)) }

// Deriv returns the m-th derivative (m >= 1).
func (p *Polynomial[T]) Deriv(m int) *Polynomial[T] {
	_, scl := mapparms(p.domain, p.window)
	return p.derived(p.ops.der(p.coef, m, scl))
}

// Integ returns the m-th integral with integration constants k (numpy
// lbnd = 0 convention).
func (p *Polynomial[T]) Integ(m int, k ...T) *Polynomial[T] {
	return p.derived(p.integCoef(m, k))
}

func (p *Polynomial[T]) integCoef(m int, k []T) []T {
	off, scl := mapparms(p.domain, p.window)
	return p.ops.integ(p.coef, m, padConsts(m, k), off, 1/scl)
}

// Trim drops trailing coefficients with |c| <= tol.
func (p *Polynomial[T]) Trim(tol T) *Polynomial[T] { return p.derived(trimcoef(p.coef, tol)) }

// Truncate drops coefficients beyond the first size.
func (p *Polynomial[T]) Truncate(size int) *Polynomial[T] {
	trimTruncate(size)
	if size >= len(p.coef) {
		return p.Copy()
	}
	return p.derived(slices.Clone(p.coef[:size]))
}

// Convert re-parameterizes the same function onto domain/window (numpy
// convert with kind = self): the converted series' coefficients are the old
// series composed with the linear map onto the old window.
func (p *Polynomial[T]) Convert(domain, window [2]T) *Polynomial[T] {
	off, scl := p.convertParms(domain, window)
	q := p.derived(p.ops.compose(p.coef, off, scl))
	q.domain, q.window = domain, window
	return q
}

// Copy returns a deep copy.
func (p *Polynomial[T]) Copy() *Polynomial[T] { return p.derived(slices.Clone(p.coef)) }

// Fit least-squares fits degree deg to (x, y), defaulting the domain to
// [min(x), max(x)] like numpy (ABCPolyBase.fit defaults).
func (p *Polynomial[T]) Fit(x, y *numgo.Array[T], deg int) *Polynomial[T] {
	if x.Ndim() != 1 {
		panicDim1D("x")
	}
	coef, domain := fitCoef(p.ops, x.Copy().Data(), y.Copy().Data(), deg)
	q := p.derived(coef)
	q.domain = domain
	return q
}

// padConsts pads integration constants with zeros to length m.
func padConsts[T numgo.Float](m int, k []T) []T {
	if len(k) > m {
		panicTooManyConsts()
	}
	out := make([]T, m)
	copy(out, k)
	return out
}
