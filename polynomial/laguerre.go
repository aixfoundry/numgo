package polynomial

import (
	"slices"

	"github.com/kaisawind/numgo"
)

// Laguerre is the Laguerre series c[0]*L0 + c[1]*L1 + ... (numpy
// polynomial.laguerre, class Laguerre). Default domain [0,1], window [-1,1].
type Laguerre[T numgo.Float] struct{ Base[T] }

// lagOps ports numpy polynomial.laguerre's coefficient kernels.
type lagOps[T numgo.Float] struct{}

// NewLaguerre builds a Laguerre series from coefficients, lowest degree first.
func NewLaguerre[T numgo.Float](coef []T) *Laguerre[T] {
	return &Laguerre[T]{newBase(lagOps[T]{}, coef)}
}

// NewLaguerreDomain is NewLaguerre with an explicit domain and window.
func NewLaguerreDomain[T numgo.Float](coef []T, domain, window [2]T) *Laguerre[T] {
	l := NewLaguerre(coef)
	l.domain, l.window = domain, window
	return l
}

// NewLaguerreFromRoots builds the monic Laguerre series with the given roots.
func NewLaguerreFromRoots[T numgo.Float](roots []T) *Laguerre[T] {
	return &Laguerre[T]{newBaseFromRoots(lagOps[T]{}, roots)}
}

func (l *Laguerre[T]) derived(coef []T) *Laguerre[T] { return &Laguerre[T]{l.Base.with(coef)} }

func (lagOps[T]) domainWindow() ([2]T, [2]T) { return [2]T{0, 1}, [2]T{0, 1} }
func (lagOps[T]) basisName() string          { return "Laguerre" }

// val is lagval: Clenshaw recurrence.
func (lagOps[T]) val(x, c []T) []T {
	out := make([]T, len(x))
	n := len(c)
	for i, xv := range x {
		var c0, c1 T
		switch {
		case n == 1:
			c0 = c[0]
		case n == 2:
			c0, c1 = c[0], c[1]
		default:
			nd := n
			c0, c1 = c[n-2], c[n-1]
			for k := 3; k <= n; k++ {
				tmp := c0
				nd--
				c0 = c[n-k] - c1*T(nd-1)/T(nd)
				c1 = tmp + c1*(2*T(nd)-1-xv)/T(nd)
			}
		}
		out[i] = c0 + c1*(1-xv)
	}
	return out
}

// vander is lagvander: v[k] = (v[k-1]*(2k-1-x) - v[k-2]*(k-1)) / k, v[1] = 1-x.
func (lagOps[T]) vander(x []T, deg int) []T {
	if deg < 0 {
		panicNegDeg()
	}
	dof := deg + 1
	v := make([]T, len(x)*dof)
	for i, xv := range x {
		v[i*dof] = 1
		if deg > 0 {
			v[i*dof+1] = 1 - xv
			for d := 2; d <= deg; d++ {
				v[i*dof+d] = (v[i*dof+d-1]*(2*T(d)-1-xv) - v[i*dof+d-2]*T(d-1)) / T(d)
			}
		}
	}
	return v
}

func (lagOps[T]) add(c1, c2 []T) []T { return coefAdd(c1, c2) }
func (lagOps[T]) sub(c1, c2 []T) []T { return coefSub(c1, c2) }

// mulx is lagmulx.
func (lagOps[T]) mulx(c []T) []T {
	if len(c) == 1 && c[0] == 0 {
		return c
	}
	prd := make([]T, len(c)+1)
	prd[0] = c[0]
	prd[1] = -c[0]
	for i := 1; i < len(c); i++ {
		prd[i+1] += -c[i] * T(i+1)
		prd[i] += c[i] * (2*T(i) + 1)
		prd[i-1] -= c[i] * T(i)
	}
	return prd
}

// mul is lagmul.
func (z lagOps[T]) mul(c1, c2 []T) []T {
	c1 = trimseq(slices.Clone(c1))
	c2 = trimseq(slices.Clone(c2))
	c, xs := c1, c2
	if len(c1) > len(c2) {
		c, xs = c2, c1
	}
	var c0, c1s []T
	switch len(c) {
	case 1:
		c0, c1s = scale(xs, c[0]), []T{0}
	case 2:
		c0, c1s = scale(xs, c[0]), scale(xs, c[1])
	default:
		nd := len(c)
		c0, c1s = scale(xs, c[len(c)-2]), scale(xs, c[len(c)-1])
		for i := 3; i <= len(c); i++ {
			tmp := c0
			nd--
			c0 = z.sub(scale(xs, c[len(c)-i]), scale(c1s, T(nd-1)/T(nd)))
			c1s = scale(z.sub(scale(c1s, 2*T(nd)-1), z.mulx(c1s)), 1/T(nd))
			c1s = z.add(tmp, c1s)
		}
	}
	return z.add(c0, z.sub(c1s, z.mulx(c1s)))
}

func (z lagOps[T]) div(c1, c2 []T) (quo, rem []T) { return divByMul(z.mul, c1, c2) }
func (z lagOps[T]) pow(c []T, n int) []T          { return powCoef(z.mul, c, n) }

// der is lagder.
func (lagOps[T]) der(c []T, m int, scl T) []T {
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
		for j := range c {
			c[j] *= scl
		}
		dst := make([]T, n)
		for j := n; j >= 2; j-- {
			dst[j-1] = -c[j]
			c[j-1] += c[j]
		}
		dst[0] = -c[1]
		c = dst
	}
	return c
}

// integ is lagint.
func (z lagOps[T]) integ(c []T, m int, k []T, lbnd, scl T) []T {
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
		tmp[0] = c[0]
		tmp[1] = -c[0]
		for j := 1; j < n; j++ {
			tmp[j] += c[j]
			tmp[j+1] = -c[j]
		}
		tmp[0] += k[i] - z.val([]T{lbnd}, tmp)[0]
		c = tmp
	}
	return c
}

// line is lagline: off + scl*x = (off+scl) - scl*L1.
func (lagOps[T]) line(off, scl T) []T {
	if scl != 0 {
		return []T{off + scl, -scl}
	}
	return []T{off}
}

// companion is lagcompanion.
func (lagOps[T]) companion(c []T) []T {
	c = trimseq(slices.Clone(c))
	n := len(c) - 1
	if n < 1 {
		panicDegAtLeast1()
	}
	if n == 1 {
		return []T{1 + c[0]/c[1]}
	}
	mat := make([]T, n*n)
	mat[0] = 1
	for k := 1; k <= n-1; k++ {
		mat[(k-1)*n+k] = -T(k) // top[k-1]
		mat[k*n+k-1] = -T(k)   // bot[k-1]
		mat[k*n+k] = 2*T(k) + 1
	}
	for i := range n {
		mat[i*n+n-1] += c[i] / c[n] * T(n)
	}
	return mat
}

func (z lagOps[T]) fromroots(roots []T) []T { return fromRootsCoef(z.line, z.mul, roots) }

// roots is lagroots: eigenvalues of the rotated companion matrix.
func (z lagOps[T]) roots(c []T) []complex128 {
	c = trimseq(slices.Clone(c))
	switch len(c) {
	case 0, 1:
		return nil
	case 2:
		return []complex128{cx(1 + c[0]/c[1])}
	}
	return rootsEig(rot180(z.companion(c)), len(c)-1)
}

// compose advances the Laguerre recurrence with the series L.
func (z lagOps[T]) compose(c []T, off, scl T) []T {
	L := z.line(off, scl) // (off+scl) - scl*L1 = the series off + scl*x
	acc := []T{c[0]}
	if len(c) > 1 {
		acc = coefAdd(acc, z.mul([]T{c[1]}, L))
	}
	v2, v1 := []T{T(1)}, L
	for d := 2; d < len(c); d++ {
		v := coefSub(z.mul(v1, z.sub([]T{2*T(d) - 1}, L)), scale(v2, T(d-1)))
		v = scale(v, 1/T(d))
		acc = coefAdd(acc, z.mul([]T{c[d]}, v))
		v2, v1 = v1, v
	}
	return acc
}

// Add returns l + o in the Laguerre basis.
func (l *Laguerre[T]) Add(o *Laguerre[T]) *Laguerre[T] {
	return l.derived(coefAdd(l.coef, o.coef))
}

// Sub returns l - o in the Laguerre basis.
func (l *Laguerre[T]) Sub(o *Laguerre[T]) *Laguerre[T] {
	return l.derived(coefSub(l.coef, o.coef))
}

// Mul returns the Laguerre product.
func (l *Laguerre[T]) Mul(o *Laguerre[T]) *Laguerre[T] {
	return l.derived(l.ops.mul(l.coef, o.coef))
}

// Div returns the quotient and remainder of l / o.
func (l *Laguerre[T]) Div(o *Laguerre[T]) (*Laguerre[T], *Laguerre[T]) {
	q, r := l.ops.div(l.coef, o.coef)
	return l.derived(q), l.derived(r)
}

// Pow raises the series to the n-th power.
func (l *Laguerre[T]) Pow(n int) *Laguerre[T] { return l.derived(l.ops.pow(l.coef, n)) }

// Deriv returns the m-th derivative (m >= 1).
func (l *Laguerre[T]) Deriv(m int) *Laguerre[T] {
	_, scl := mapparms(l.domain, l.window)
	return l.derived(l.ops.der(l.coef, m, scl))
}

// Integ returns the m-th integral with integration constants k.
func (l *Laguerre[T]) Integ(m int, k ...T) *Laguerre[T] {
	off, scl := mapparms(l.domain, l.window)
	return l.derived(l.ops.integ(l.coef, m, padConsts(m, k), off, 1/scl))
}

// Trim drops trailing coefficients with |c| <= tol.
func (l *Laguerre[T]) Trim(tol T) *Laguerre[T] { return l.derived(trimcoef(l.coef, tol)) }

// Truncate drops coefficients beyond the first size.
func (l *Laguerre[T]) Truncate(size int) *Laguerre[T] {
	trimTruncate(size)
	if size >= len(l.coef) {
		return l.Copy()
	}
	return l.derived(slices.Clone(l.coef[:size]))
}

// Convert re-parameterizes the same function onto domain/window.
func (l *Laguerre[T]) Convert(domain, window [2]T) *Laguerre[T] {
	off, scl := l.convertParms(domain, window)
	q := l.derived(l.ops.compose(l.coef, off, scl))
	q.domain, q.window = domain, window
	return q
}

// Copy returns a deep copy.
func (l *Laguerre[T]) Copy() *Laguerre[T] { return l.derived(slices.Clone(l.coef)) }

// Fit least-squares fits degree deg to (x, y) with numpy default domain.
func (l *Laguerre[T]) Fit(x, y *numgo.Array[T], deg int) *Laguerre[T] {
	if x.Ndim() != 1 {
		panicDim1D("x")
	}
	coef, domain := fitCoef(l.ops, x.Copy().Data(), y.Copy().Data(), deg)
	q := l.derived(coef)
	q.domain = domain
	return q
}
