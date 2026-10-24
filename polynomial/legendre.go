package polynomial

import (
	"math"
	"slices"

	"github.com/aixfoundry/numgo"
)

// Legendre is the Legendre series c[0]*P0 + c[1]*P1 + ... (numpy
// polynomial.legendre, class Legendre). Default domain and window [-1,1].
type Legendre[T numgo.Float] struct{ Base[T] }

// legOps ports numpy polynomial.legendre's coefficient kernels.
type legOps[T numgo.Float] struct{}

// NewLegendre builds a Legendre series from coefficients, lowest degree first.
func NewLegendre[T numgo.Float](coef []T) *Legendre[T] {
	return &Legendre[T]{newBase(legOps[T]{}, coef)}
}

// NewLegendreDomain is NewLegendre with an explicit domain and window.
func NewLegendreDomain[T numgo.Float](coef []T, domain, window [2]T) *Legendre[T] {
	l := NewLegendre(coef)
	l.domain, l.window = domain, window
	return l
}

// NewLegendreFromRoots builds the monic Legendre series with the given roots.
func NewLegendreFromRoots[T numgo.Float](roots []T) *Legendre[T] {
	return &Legendre[T]{newBaseFromRoots(legOps[T]{}, roots)}
}

func (l *Legendre[T]) derived(coef []T) *Legendre[T] { return &Legendre[T]{l.Base.with(coef)} }

func (legOps[T]) domainWindow() ([2]T, [2]T) { return [2]T{-1, 1}, [2]T{-1, 1} }
func (legOps[T]) basisName() string          { return "Legendre" }

// val is legval: Clenshaw recurrence.
func (legOps[T]) val(x, c []T) []T {
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
				c1 = tmp + c1*xv*(2*T(nd)-1)/T(nd)
			}
		}
		out[i] = c0 + c1*xv
	}
	return out
}

// vander is legvander: (v[k-1]*x*(2k-1) - v[k-2]*(k-1)) / k.
func (legOps[T]) vander(x []T, deg int) []T {
	if deg < 0 {
		panicNegDeg()
	}
	dof := deg + 1
	v := make([]T, len(x)*dof)
	for i, xv := range x {
		v[i*dof] = 1
		if deg > 0 {
			v[i*dof+1] = xv
			for d := 2; d <= deg; d++ {
				v[i*dof+d] = (v[i*dof+d-1]*xv*(2*T(d)-1) - v[i*dof+d-2]*T(d-1)) / T(d)
			}
		}
	}
	return v
}

func (legOps[T]) add(c1, c2 []T) []T { return coefAdd(c1, c2) }
func (legOps[T]) sub(c1, c2 []T) []T { return coefSub(c1, c2) }

// mulx is legmulx: x times a Legendre series via the three-term recurrence.
func (legOps[T]) mulx(c []T) []T {
	if len(c) == 1 && c[0] == 0 {
		return c
	}
	prd := make([]T, len(c)+1)
	prd[1] = c[0]
	for i := 1; i < len(c); i++ {
		s := 2*T(i) + 1
		prd[i+1] += c[i] * (T(i) + 1) / s
		prd[i-1] += c[i] * T(i) / s
	}
	return prd
}

// mul is legmul: repeated multiplication by the shorter series' terms.
func (z legOps[T]) mul(c1, c2 []T) []T {
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
			c1s = z.add(tmp, scale(z.mulx(c1s), (2*T(nd)-1)/T(nd)))
		}
	}
	return z.add(c0, z.mulx(c1s))
}

func (z legOps[T]) div(c1, c2 []T) (quo, rem []T) { return divByMul(z.mul, c1, c2) }
func (z legOps[T]) pow(c []T, n int) []T          { return powCoef(z.mul, c, n) }

// der is legder.
func (legOps[T]) der(c []T, m int, scl T) []T {
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
		for j := n; j >= 3; j-- {
			dst[j-1] = (2*T(j) - 1) * c[j]
			c[j-2] += c[j]
		}
		if n > 1 {
			dst[1] = 3 * c[2]
		}
		dst[0] = c[1]
		c = dst
	}
	return c
}

// integ is legint with lbnd folded into the constant term.
func (z legOps[T]) integ(c []T, m int, k []T, lbnd, scl T) []T {
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
		if n > 1 {
			tmp[2] = c[1] / 3
		}
		for j := 2; j < n; j++ {
			t := c[j] / (2*T(j) + 1)
			tmp[j+1] = t
			tmp[j-1] -= t
		}
		tmp[0] += k[i] - z.val([]T{lbnd}, tmp)[0]
		c = tmp
	}
	return c
}

// line is legline: off + scl*P1.
func (legOps[T]) line(off, scl T) []T {
	if scl != 0 {
		return []T{off, scl}
	}
	return []T{off}
}

// companion is legcompanion.
func (legOps[T]) companion(c []T) []T {
	c = trimseq(slices.Clone(c))
	n := len(c) - 1
	if n < 1 {
		panicDegAtLeast1()
	}
	if n == 1 {
		return []T{-c[0] / c[1]}
	}
	mat := make([]T, n*n)
	// scl[k] = 1/sqrt(2k+1); top/bot[k] = k*scl[k-1]*scl[k] at (k,k+1)/(k+1,k)
	// for k = 0..n-2 (numpy arange(1,n)*scl[:n-1]*scl[1:n]).
	for k := 1; k <= n-1; k++ {
		v := T(f64(T(k)) / math.Sqrt(f64(2*T(k)-1)) / math.Sqrt(f64(2*T(k)+1)))
		mat[(k-1)*n+k] = v // top[k-1]
		mat[k*n+k-1] = v   // bot[k-1]
	}
	// Last column: -(c[:-1]/c[-1]) * (scl/scl[-1]) * n/(2n-1).
	for i := range n {
		f := T(math.Sqrt(f64(2*T(n)-1)) / math.Sqrt(f64(2*T(i)+1)))
		mat[i*n+n-1] -= c[i] / c[n] * f * T(n) / T(2*n-1)
	}
	return mat
}

func (z legOps[T]) fromroots(roots []T) []T { return fromRootsCoef(z.line, z.mul, roots) }

// roots is legroots: eigenvalues of the rotated companion matrix.
func (z legOps[T]) roots(c []T) []complex128 {
	c = trimseq(slices.Clone(c))
	switch len(c) {
	case 0, 1:
		return nil
	case 2:
		return []complex128{-cx(c[0] / c[1])}
	}
	return rootsEig(rot180(z.companion(c)), len(c)-1)
}

// compose advances the Legendre recurrence with the series L.
func (z legOps[T]) compose(c []T, off, scl T) []T {
	L := z.line(off, scl)
	acc := []T{c[0]}
	if len(c) > 1 {
		acc = coefAdd(acc, z.mul([]T{c[1]}, L))
	}
	v2, v1 := []T{T(1)}, L
	for d := 2; d < len(c); d++ {
		v := coefSub(scale(z.mul(v1, L), (2*T(d)-1)/T(d)), scale(v2, T(d-1)/T(d)))
		acc = coefAdd(acc, z.mul([]T{c[d]}, v))
		v2, v1 = v1, v
	}
	return acc
}

// Add returns l + o in the Legendre basis.
func (l *Legendre[T]) Add(o *Legendre[T]) *Legendre[T] {
	return l.derived(coefAdd(l.coef, o.coef))
}

// Sub returns l - o in the Legendre basis.
func (l *Legendre[T]) Sub(o *Legendre[T]) *Legendre[T] {
	return l.derived(coefSub(l.coef, o.coef))
}

// Mul returns the Legendre product.
func (l *Legendre[T]) Mul(o *Legendre[T]) *Legendre[T] {
	return l.derived(l.ops.mul(l.coef, o.coef))
}

// Div returns the quotient and remainder of l / o.
func (l *Legendre[T]) Div(o *Legendre[T]) (*Legendre[T], *Legendre[T]) {
	q, r := l.ops.div(l.coef, o.coef)
	return l.derived(q), l.derived(r)
}

// Pow raises the series to the n-th power.
func (l *Legendre[T]) Pow(n int) *Legendre[T] { return l.derived(l.ops.pow(l.coef, n)) }

// Deriv returns the m-th derivative (m >= 1).
func (l *Legendre[T]) Deriv(m int) *Legendre[T] {
	_, scl := mapparms(l.domain, l.window)
	return l.derived(l.ops.der(l.coef, m, scl))
}

// Integ returns the m-th integral with integration constants k.
func (l *Legendre[T]) Integ(m int, k ...T) *Legendre[T] {
	off, scl := mapparms(l.domain, l.window)
	return l.derived(l.ops.integ(l.coef, m, padConsts(m, k), off, 1/scl))
}

// Trim drops trailing coefficients with |c| <= tol.
func (l *Legendre[T]) Trim(tol T) *Legendre[T] { return l.derived(trimcoef(l.coef, tol)) }

// Truncate drops coefficients beyond the first size.
func (l *Legendre[T]) Truncate(size int) *Legendre[T] {
	trimTruncate(size)
	if size >= len(l.coef) {
		return l.Copy()
	}
	return l.derived(slices.Clone(l.coef[:size]))
}

// Convert re-parameterizes the same function onto domain/window.
func (l *Legendre[T]) Convert(domain, window [2]T) *Legendre[T] {
	off, scl := l.convertParms(domain, window)
	q := l.derived(l.ops.compose(l.coef, off, scl))
	q.domain, q.window = domain, window
	return q
}

// Copy returns a deep copy.
func (l *Legendre[T]) Copy() *Legendre[T] { return l.derived(slices.Clone(l.coef)) }

// Fit least-squares fits degree deg to (x, y) with numpy default domain.
func (l *Legendre[T]) Fit(x, y *numgo.Array[T], deg int) *Legendre[T] {
	if x.Ndim() != 1 {
		panicDim1D("x")
	}
	coef, domain := fitCoef(l.ops, x.Copy().Data(), y.Copy().Data(), deg)
	q := l.derived(coef)
	q.domain = domain
	return q
}
