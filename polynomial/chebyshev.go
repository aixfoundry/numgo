package polynomial

import (
	"math"
	"slices"

	"github.com/kaisawind/numgo"
)

// Chebyshev is the Chebyshev series of the first kind
// c[0]*T0 + c[1]*T1 + ... (numpy polynomial.chebyshev, class Chebyshev).
// Default domain and window [-1,1].
type Chebyshev[T numgo.Float] struct{ Base[T] }

// chebOps ports numpy polynomial.chebyshev's coefficient kernels.
type chebOps[T numgo.Float] struct{}

// NewChebyshev builds a Chebyshev series from coefficients, lowest degree
// first. The coefficients are copied.
func NewChebyshev[T numgo.Float](coef []T) *Chebyshev[T] {
	return &Chebyshev[T]{newBase(chebOps[T]{}, coef)}
}

// NewChebyshevDomain is NewChebyshev with an explicit domain and window.
func NewChebyshevDomain[T numgo.Float](coef []T, domain, window [2]T) *Chebyshev[T] {
	c := NewChebyshev(coef)
	c.domain, c.window = domain, window
	return c
}

// NewChebyshevFromRoots builds the monic Chebyshev series with the given
// roots (numpy Chebyshev.fromroots with default domain).
func NewChebyshevFromRoots[T numgo.Float](roots []T) *Chebyshev[T] {
	return &Chebyshev[T]{newBaseFromRoots(chebOps[T]{}, roots)}
}

func (c *Chebyshev[T]) derived(coef []T) *Chebyshev[T] { return &Chebyshev[T]{c.Base.with(coef)} }

func (chebOps[T]) domainWindow() ([2]T, [2]T) { return [2]T{-1, 1}, [2]T{-1, 1} }
func (chebOps[T]) basisName() string          { return "Chebyshev" }

// val is chebval: Clenshaw recurrence.
func (chebOps[T]) val(x, c []T) []T {
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
			x2 := 2 * xv
			c0, c1 = c[n-2], c[n-1]
			for k := 3; k <= n; k++ {
				tmp := c0
				c0 = c[n-k] - c1
				c1 = tmp + c1*x2
			}
		}
		out[i] = c0 + c1*xv
	}
	return out
}

// vander is chebvander: forward recurrence v[k] = 2x*v[k-1] - v[k-2].
func (chebOps[T]) vander(x []T, deg int) []T {
	if deg < 0 {
		panicNegDeg()
	}
	dof := deg + 1
	v := make([]T, len(x)*dof)
	for i, xv := range x {
		v[i*dof] = 1
		if deg > 0 {
			x2 := 2 * xv
			v[i*dof+1] = xv
			for d := 2; d <= deg; d++ {
				v[i*dof+d] = v[i*dof+d-1]*x2 - v[i*dof+d-2]
			}
		}
	}
	return v
}

func (chebOps[T]) add(c1, c2 []T) []T { return coefAdd(c1, c2) }
func (chebOps[T]) sub(c1, c2 []T) []T { return coefSub(c1, c2) }

// c2z maps Chebyshev coefficients to the z-series y = arccos(x) representation
// (_cseries_to_zseries); z2c is the inverse (_zseries_to_cseries).
func c2z[T numgo.Float](c []T) []T {
	n := len(c)
	init := make([]T, 2*n-1)
	for i, v := range c {
		init[n-1+i] = v / 2
	}
	zs := make([]T, 2*n-1)
	for i := range zs {
		zs[i] = init[i] + init[2*n-2-i]
	}
	return zs
}

func z2c[T numgo.Float](zs []T) []T {
	n := (len(zs) + 1) / 2
	c := slices.Clone(zs[n-1:])
	for i := 1; i < n; i++ {
		c[i] *= 2
	}
	return c
}

// mul is chebmul: convolution in z-series space.
func (chebOps[T]) mul(c1, c2 []T) []T { return trimseq(z2c(convolve(c2z(c1), c2z(c2)))) }

// div is chebdiv via _zseries_div.
func (chebOps[T]) div(c1, c2 []T) (quo, rem []T) {
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
	l1, l2 := 2*lc1-1, 2*lc2-1
	dlen := l1 - l2
	full := c2z(c2)
	scl := full[0]
	z1 := c2z(c1)
	z2 := scale(full, 1/scl)
	quo = make([]T, dlen+1)
	i, j := 0, dlen
	for i < j {
		r := z1[i]
		quo[i] = r
		quo[dlen-i] = r
		tmp := scale(z2, r)
		for k := range z2 {
			z1[i+k] -= tmp[k]
			z1[j+k] -= tmp[k]
		}
		i++
		j--
	}
	r := z1[i]
	quo[i] = r
	tmp2 := scale(z2, r)
	for k := range z2 {
		z1[i+k] -= tmp2[k]
	}
	for k := range quo {
		quo[k] /= scl
	}
	return trimseq(z2c(quo)), trimseq(z2c(z1[i+1 : i-1+l2]))
}

func (z chebOps[T]) pow(c []T, n int) []T {
	return powCoef(chebOps[T]{}.mul, c, n)
}

// der is chebder.
func (chebOps[T]) der(c []T, m int, scl T) []T {
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
			dst[j-1] = 2 * T(j) * c[j]
			c[j-2] += T(j) * c[j] / T(j-2)
		}
		if n > 1 {
			dst[1] = 4 * c[2]
		}
		dst[0] = c[1]
		c = dst
	}
	return c
}

// integ is chebint with lbnd folded into the constant term.
func (z chebOps[T]) integ(c []T, m int, k []T, lbnd, scl T) []T {
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
			tmp[2] = c[1] / 4
		}
		for j := 2; j < n; j++ {
			tmp[j+1] = c[j] / (2 * T(j+1))
			tmp[j-1] -= c[j] / (2 * T(j-1))
		}
		tmp[0] += k[i] - z.val([]T{lbnd}, tmp)[0]
		c = tmp
	}
	return c
}

// line is chebline: off + scl*T1.
func (chebOps[T]) line(off, scl T) []T {
	if scl != 0 {
		return []T{off, scl}
	}
	return []T{off}
}

// companion is chebcompanion with its sqrt(1/2) normalization.
func (chebOps[T]) companion(c []T) []T {
	c = trimseq(slices.Clone(c))
	n := len(c) - 1
	if n < 1 {
		panicDegAtLeast1()
	}
	if n == 1 {
		return []T{-c[0] / c[1]}
	}
	mat := make([]T, n*n)
	s5 := T(math.Sqrt(0.5))
	mat[1] = s5
	mat[n] = s5
	for k := 1; k <= n-2; k++ {
		mat[k*n+k+1] = T(0.5)
		mat[(k+1)*n+k] = T(0.5)
	}
	for i := range n {
		f := T(1)
		if i == 0 {
			f = T(math.Sqrt2)
		}
		mat[i*n+n-1] -= c[i] / c[n] * f * T(0.5)
	}
	return mat
}

func (z chebOps[T]) fromroots(roots []T) []T { return fromRootsCoef(z.line, z.mul, roots) }

// roots is chebroots: eigenvalues of the rotated companion matrix.
func (z chebOps[T]) roots(c []T) []complex128 {
	c = trimseq(slices.Clone(c))
	switch len(c) {
	case 0, 1:
		return nil
	case 2:
		return []complex128{-cx(c[0] / c[1])}
	}
	return rootsEig(rot180(z.companion(c)), len(c)-1)
}

// rot180 flips a flat row-major (n, n) matrix on both axes ([::-1, ::-1]).
func rot180[T numgo.Float](mat []T) []T {
	n := len(mat)
	side := 1
	for side*side < n {
		side++
	}
	out := make([]T, n)
	for i := 0; i < side; i++ {
		for j := 0; j < side; j++ {
			out[i*side+j] = mat[(side-1-i)*side+(side-1-j)]
		}
	}
	return out
}

// compose advances the Chebyshev recurrence with the series L = off + scl*x.
func (z chebOps[T]) compose(c []T, off, scl T) []T {
	L := z.line(off, scl)
	acc := []T{c[0]}
	if len(c) > 1 {
		acc = coefAdd(acc, z.mul([]T{c[1]}, L))
	}
	v2, v1 := []T{T(1)}, L
	for d := 2; d < len(c); d++ {
		v := coefSub(z.mul(v1, scale(L, 2)), v2)
		acc = coefAdd(acc, z.mul([]T{c[d]}, v))
		v2, v1 = v1, v
	}
	return acc
}

// Add returns c + o in the Chebyshev basis.
func (c *Chebyshev[T]) Add(o *Chebyshev[T]) *Chebyshev[T] {
	return c.derived(coefAdd(c.coef, o.coef))
}

// Sub returns c - o in the Chebyshev basis.
func (c *Chebyshev[T]) Sub(o *Chebyshev[T]) *Chebyshev[T] {
	return c.derived(coefSub(c.coef, o.coef))
}

// Mul returns the Chebyshev product.
func (c *Chebyshev[T]) Mul(o *Chebyshev[T]) *Chebyshev[T] {
	return c.derived(c.ops.mul(c.coef, o.coef))
}

// Div returns the quotient and remainder of c / o.
func (c *Chebyshev[T]) Div(o *Chebyshev[T]) (*Chebyshev[T], *Chebyshev[T]) {
	q, r := c.ops.div(c.coef, o.coef)
	return c.derived(q), c.derived(r)
}

// Pow raises the series to the n-th power.
func (c *Chebyshev[T]) Pow(n int) *Chebyshev[T] { return c.derived(c.ops.pow(c.coef, n)) }

// Deriv returns the m-th derivative (m >= 1).
func (c *Chebyshev[T]) Deriv(m int) *Chebyshev[T] {
	_, scl := mapparms(c.domain, c.window)
	return c.derived(c.ops.der(c.coef, m, scl))
}

// Integ returns the m-th integral with integration constants k.
func (c *Chebyshev[T]) Integ(m int, k ...T) *Chebyshev[T] {
	off, scl := mapparms(c.domain, c.window)
	return c.derived(c.ops.integ(c.coef, m, padConsts(m, k), off, 1/scl))
}

// Trim drops trailing coefficients with |c| <= tol.
func (c *Chebyshev[T]) Trim(tol T) *Chebyshev[T] { return c.derived(trimcoef(c.coef, tol)) }

// Truncate drops coefficients beyond the first size.
func (c *Chebyshev[T]) Truncate(size int) *Chebyshev[T] {
	trimTruncate(size)
	if size >= len(c.coef) {
		return c.Copy()
	}
	return c.derived(slices.Clone(c.coef[:size]))
}

// Convert re-parameterizes the same function onto domain/window.
func (c *Chebyshev[T]) Convert(domain, window [2]T) *Chebyshev[T] {
	off, scl := c.convertParms(domain, window)
	q := c.derived(c.ops.compose(c.coef, off, scl))
	q.domain, q.window = domain, window
	return q
}

// Copy returns a deep copy.
func (c *Chebyshev[T]) Copy() *Chebyshev[T] { return c.derived(slices.Clone(c.coef)) }

// Fit least-squares fits degree deg to (x, y) with numpy default domain.
func (c *Chebyshev[T]) Fit(x, y *numgo.Array[T], deg int) *Chebyshev[T] {
	if x.Ndim() != 1 {
		panicDim1D("x")
	}
	coef, domain := fitCoef(c.ops, x.Copy().Data(), y.Copy().Data(), deg)
	q := c.derived(coef)
	q.domain = domain
	return q
}
