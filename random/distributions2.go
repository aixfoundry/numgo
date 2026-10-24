package random

import (
	"math"

	"github.com/kaisawind/numgo"
	"github.com/kaisawind/numgo/internal/cmath"
	"github.com/kaisawind/numgo/internal/lapack"
)

// Second batch of Generator distributions, ported 1:1 from
// numpy/random/src/distributions/distributions.c (file:line refs per method)
// preserving the exact underlying draw sequence. Transcendentals route
// through internal/cmath (glibc bit-parity); VonMises additionally needs
// bit-exact cos/acos — cosG is a port of glibc 2.39 s_sin.c (this machine's
// libm) and acosG of e_asin.c (see the helpers below).

// chisquare: distributions.c:466 — 2 * standard_gamma(df/2).
func (g *Generator) Chisquare(df float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = 2.0 * g.StandardGammaScalar(df/2.0)
	}
	return out
}

// F: distributions.c:470 — (χ²(dfnum)*dfden) / (χ²(dfden)*dfnum).
func (g *Generator) F(dfnum, dfden float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		subexpr1 := 2.0 * g.StandardGammaScalar(dfnum/2.0) * dfden
		subexpr2 := 2.0 * g.StandardGammaScalar(dfden/2.0) * dfnum
		out.Data()[i] = subexpr1 / subexpr2
	}
	return out
}

// StandardCauchy: distributions.c:476 — normal(0)/normal(0), NOT tan(π(u-1/2)).
func (g *Generator) StandardCauchy(size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = g.StandardNormalScalar() / g.StandardNormalScalar()
	}
	return out
}

// Pareto: distributions.c:482 — expm1(standard_exponential / a).
func (g *Generator) Pareto(a float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = expm1G(g.StandardExponentialScalar() / a)
	}
	return out
}

// Weibull: distributions.c:486 — standard_exponential ** (1/a); a == 0 → 0
// without consuming a draw.
func (g *Generator) Weibull(a float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		if a == 0.0 {
			continue
		}
		out.Data()[i] = cmath.Pow(g.StandardExponentialScalar(), 1.0/a)
	}
	return out
}

// Power: distributions.c:493 — (−expm1(−standard_exponential)) ** (1/a).
func (g *Generator) Power(a float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = cmath.Pow(-expm1G(-g.StandardExponentialScalar()), 1.0/a)
	}
	return out
}

// Laplace: distributions.c:497 — inverse CDF; rejects U == 0 by redrawing.
func (g *Generator) Laplace(loc, scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		var U float64
		for {
			U = g.nextDouble()
			if U > 0.0 {
				break
			}
		}
		lo := U + U
		hi := 2.0 - U - U
		var sign float64 = 1.0
		if U >= 0.5 {
			sign = -1.0
		}
		m := lo
		if hi < lo {
			m = hi
		}
		out.Data()[i] = loc + sign*scale*cmath.Log(m)
	}
	return out
}

// Gumbel: distributions.c:510 — loc − scale*log(−log(U)) with U = 1−uniform,
// rejecting U == 1 by redrawing.
func (g *Generator) Gumbel(loc, scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		var U float64
		for {
			U = 1.0 - g.nextDouble()
			if U < 1.0 {
				break
			}
		}
		out.Data()[i] = loc - scale*cmath.Log(-cmath.Log(U))
	}
	return out
}

// Logistic: distributions.c:521 — inverse CDF; rejects U == 0 by redrawing.
func (g *Generator) Logistic(loc, scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		var U float64
		for {
			U = g.nextDouble()
			if U > 0.0 {
				break
			}
		}
		out.Data()[i] = loc + scale*cmath.Log(U/(1.0-U))
	}
	return out
}

// Lognormal: distributions.c:532 — exp(normal(mean, sigma)).
func (g *Generator) Lognormal(mean, sigma float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = cmath.Exp(mean + sigma*g.StandardNormalScalar())
	}
	return out
}

// Rayleigh: distributions.c:536 — mode * sqrt(2 * standard_exponential).
func (g *Generator) Rayleigh(scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = scale * math.Sqrt(2.0*g.StandardExponentialScalar())
	}
	return out
}

// StandardT: distributions.c:540 — sqrt(df/2) * normal / sqrt(gamma(df/2)).
func (g *Generator) StandardT(df float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		num := g.StandardNormalScalar()
		denom := g.StandardGammaScalar(df / 2.0)
		out.Data()[i] = math.Sqrt(df/2.0) * num / math.Sqrt(denom)
	}
	return out
}

// NoncentralChisquare: distributions.c:829. df > 1: χ²(df−1) + (N + √nonc)²;
// df ≤ 1: χ²(df + 2·Poisson(nonc/2)); nonc == 0 falls back to plain χ².
func (g *Generator) NoncentralChisquare(df, nonc float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		if math.IsNaN(nonc) {
			out.Data()[i] = math.NaN()
			continue
		}
		if nonc == 0 {
			out.Data()[i] = 2.0 * g.StandardGammaScalar(df/2.0)
			continue
		}
		if 1 < df {
			chi2 := 2.0 * g.StandardGammaScalar((df-1.0)/2.0)
			nv := g.StandardNormalScalar() + math.Sqrt(nonc)
			out.Data()[i] = chi2 + nv*nv
		} else {
			p := g.poissonScalar(nonc / 2.0)
			out.Data()[i] = 2.0 * g.StandardGammaScalar((df+2.0*float64(p))/2.0)
		}
	}
	return out
}

// NoncentralF: distributions.c:847 — noncentral_chisquare(dfnum,nonc)*dfden /
// (χ²(dfden)*dfnum).
func (g *Generator) NoncentralF(dfnum, dfden, nonc float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		chi2 := g.noncentralChisquareScalar(dfnum, nonc)
		t := chi2 * dfden
		out.Data()[i] = t / (2.0 * g.StandardGammaScalar(dfden/2.0) * dfnum)
	}
	return out
}

func (g *Generator) noncentralChisquareScalar(df, nonc float64) float64 {
	if math.IsNaN(nonc) {
		return math.NaN()
	}
	if nonc == 0 {
		return 2.0 * g.StandardGammaScalar(df/2.0)
	}
	if 1 < df {
		chi2 := 2.0 * g.StandardGammaScalar((df-1.0)/2.0)
		nv := g.StandardNormalScalar() + math.Sqrt(nonc)
		return chi2 + nv*nv
	}
	p := g.poissonScalar(nonc / 2.0)
	return 2.0 * g.StandardGammaScalar((df+2.0*float64(p))/2.0)
}

// Wald: distributions.c:853 — inverse Gaussian via normal+uniform selection.
func (g *Generator) Wald(mean, scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		Y := g.StandardNormalScalar()
		Y = mean * Y * Y
		d := 1.0 + math.Sqrt(1.0+4.0*scale/Y)
		X := mean * (1.0 - 2.0/d)
		U := g.nextDouble()
		if U <= mean/(mean+X) {
			out.Data()[i] = X
		} else {
			out.Data()[i] = mean * mean / X
		}
	}
	return out
}

// VonMises: distributions.c:869 — rejection on the wrapped density; the four
// kappa regimes (NaN, <1e-8 uniform, <1e-5 taylor, ≤1e6 rejection, wrapped
// normal fallback) are ported in order.
func (g *Generator) VonMises(mu, kappa float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = g.vonMisesScalar(mu, kappa)
	}
	return out
}

func (g *Generator) vonMisesScalar(mu, kappa float64) float64 {
	if math.IsNaN(kappa) {
		return math.NaN()
	}
	if kappa < 1e-8 {
		return math.Pi * (2.0*g.nextDouble() - 1.0)
	}
	var s float64
	if kappa < 1e-5 {
		s = 1.0/kappa + kappa
	} else if kappa <= 1e6 {
		r := 1.0 + math.Sqrt(1.0+4.0*kappa*kappa)
		rho := (r - math.Sqrt(2.0*r)) / (2.0 * kappa)
		s = (1.0 + rho*rho) / (2.0 * rho)
	} else {
		result := mu + math.Sqrt(1.0/kappa)*g.StandardNormalScalar()
		if result < -math.Pi {
			result += 2.0 * math.Pi
		}
		if result > math.Pi {
			result -= 2.0 * math.Pi
		}
		return result
	}
	var W float64
	for {
		U := g.nextDouble()
		// numpy multiplies the literal 0x1.921fb54442d18p+1 (== math.Pi's
		// value, but the constant form compiles to the identical FMA-free
		// product with correct rounding; math.Pi * U gave a 1-ULP
		// difference in one observed draw).
		Z := cosG(0x1.921fb54442d18p+1 * U)
		W = (1.0 + s*Z) / (s + Z)
		Y := kappa * (s - W)
		V := g.nextDouble()
		if (Y*(2.0-Y)-V >= 0) || (cmath.Log(Y/V)+1.0-Y >= 0) {
			break
		}
	}
	U := g.nextDouble()
	result := acosG(W)
	if U < 0.5 {
		result = -result
	}
	result += mu
	neg := result < 0
	mod := math.Abs(result)
	mod = math.Mod(mod+math.Pi, 2.0*math.Pi) - math.Pi
	if neg {
		mod *= -1
	}
	return mod
}

// Triangular: distributions.c:1062 — piecewise inverse CDF via sqrt.
func (g *Generator) Triangular(left, mode, right float64, size ...int) *arrayF {
	if left > mode {
		panic("left > mode")
	}
	if mode > right {
		panic("mode > right")
	}
	if left == right {
		panic("left == right")
	}
	n := sizeOr1(size)
	out := newF(n)
	base := right - left
	leftbase := mode - left
	ratio := leftbase / base
	leftprod := leftbase * base
	rightprod := (right - mode) * base
	for i := 0; i < n; i++ {
		U := g.nextDouble()
		if U <= ratio {
			out.Data()[i] = left + math.Sqrt(U*leftprod)
		} else {
			out.Data()[i] = right - math.Sqrt((1.0-U)*rightprod)
		}
	}
	return out
}

// Logseries: distributions.c:941.
func (g *Generator) Logseries(p float64, size ...int) *arrayI {
	n := sizeOr1(size)
	out := newI(n)
	for i := 0; i < n; i++ {
		r := cmath.Log1p(-p)
		for {
			V := g.nextDouble()
			if V >= p {
				out.Data()[i] = 1
				break
			}
			U := g.nextDouble()
			q := -expm1G(r * U)
			if V <= q*q {
				result := int64(math.Floor(1.0 + cmath.Log(V)/cmath.Log(q)))
				if result < 1 || V == 0.0 {
					continue
				}
				out.Data()[i] = result
				break
			}
			if V >= q {
				out.Data()[i] = 1
			} else {
				out.Data()[i] = 2
			}
			break
		}
	}
	return out
}

// geometricSearch: distributions.c:961.
func (g *Generator) geometricSearch(p float64) int64 {
	X := int64(1)
	sum := p
	prod := p
	q := 1.0 - p
	U := g.nextDouble()
	for U > sum {
		prod *= q
		sum += prod
		X++
	}
	return X
}

// geometricInversion: distributions.c:994.
func (g *Generator) geometricInversion(p float64) int64 {
	z := math.Ceil(-g.StandardExponentialScalar() / cmath.Log1p(-p))
	// smallest double larger than INT64_MAX
	if z >= 9.223372036854776e+18 {
		return math.MaxInt64
	}
	return int64(z)
}

// Geometric: distributions.c:1006 — search for p ≥ 1/3, inversion otherwise.
func (g *Generator) Geometric(p float64, size ...int) *arrayI {
	if p <= 0 || p > 1 {
		panic("p <= 0, p > 1 or p contains NaNs")
	}
	n := sizeOr1(size)
	out := newI(n)
	for i := 0; i < n; i++ {
		if p >= 0.333333333333333333333333 {
			out.Data()[i] = g.geometricSearch(p)
		} else {
			out.Data()[i] = g.geometricInversion(p)
		}
	}
	return out
}

// randomZipf: distributions.c:1018.
func (g *Generator) randomZipf(a float64) int64 {
	if a >= 1025 {
		return 1
	}
	am1 := a - 1.0
	b := cmath.Pow(2.0, am1)
	const randIntMax = 9223372036854775807.0
	Umin := cmath.Pow(randIntMax, -am1)
	for {
		U01 := g.nextDouble()
		U := U01*Umin + (1.0 - U01)
		V := g.nextDouble()
		X := math.Floor(cmath.Pow(U, -1.0/am1))
		if X > randIntMax || X < 1.0 {
			continue
		}
		T := cmath.Pow(1.0+1.0/X, am1)
		if V*X*(T-1.0)/(b-1.0) <= T/b {
			return int64(X)
		}
	}
}

// Zipf: distributions.c _generator.pyx zipf → random_zipf, CONS_GT_1.
func (g *Generator) Zipf(a float64, size ...int) *arrayI {
	if !(a > 1) {
		panic("a <= 1 or a is NaN")
	}
	n := sizeOr1(size)
	out := newI(n)
	for i := 0; i < n; i++ {
		out.Data()[i] = g.randomZipf(a)
	}
	return out
}

// logfactorial: distributions/logfactorial.c — table up to 125, Stirling
// beyond (logfact table is the C one verbatim).
var logfact = [126]float64{
	0, 0, 0.69314718055994529, 1.791759469228055, 3.1780538303479458,
	4.7874917427820458, 6.5792512120101012, 8.5251613610654147,
	10.604602902745251, 12.801827480081469, 15.104412573075516,
	17.502307845873887, 19.987214495661885, 22.552163853123425,
	25.19122118273868, 27.89927138384089, 30.671860106080672,
	33.505073450136891, 36.395445208033053, 39.339884187199495,
	42.335616460753485, 45.380138898476908, 48.471181351835227,
	51.606675567764377, 54.784729398112319, 58.003605222980518,
	61.261701761002001, 64.557538627006338, 67.88974313718154,
	71.257038967168015, 74.658236348830158, 78.092223553315307,
	81.557959456115043, 85.054467017581516, 88.58082754219768,
	92.136175603687093, 95.719694542143202, 99.330612454787428,
	102.96819861451381, 106.63176026064346, 110.32063971475739,
	114.03421178146171, 117.77188139974507, 121.53308151543864,
	125.3172711493569, 129.12393363912722, 132.95257503561632,
	136.80272263732635, 140.67392364823425, 144.5657439463449,
	148.47776695177302, 152.40959258449735, 156.3608363030788,
	160.3311282166309, 164.32011226319517, 168.32744544842765,
	172.35279713916279, 176.39584840699735, 180.45629141754378,
	184.53382886144948, 188.6281734236716, 192.7390472878449,
	196.86618167289001, 201.00931639928152, 205.1681994826412,
	209.34258675253685, 213.53224149456327, 217.73693411395422,
	221.95644181913033, 226.1905483237276, 230.43904356577696,
	234.70172344281826, 238.97838956183432, 243.26884900298271,
	247.57291409618688, 251.89040220972319, 256.22113555000954,
	260.56494097186322, 264.92164979855278, 269.29109765101981,
	273.67312428569369, 278.06757344036612, 282.4742926876304,
	286.89313329542699, 291.32395009427029, 295.76660135076065,
	300.22094864701415, 304.68685676566872, 309.1641935801469,
	313.65282994987905, 318.1526396202093, 322.66349912672615,
	327.1852877037752, 331.71788719692847, 336.26118197919845,
	340.81505887079902, 345.37940706226686, 349.95411804077025,
	354.53908551944079, 359.1342053695754, 363.73937555556347,
	368.35449607240474, 372.97946888568902, 377.61419787391867,
	382.25858877306001, 386.91254912321756, 391.57598821732961,
	396.24881705179155, 400.93094827891576, 405.6222961611449,
	410.32277652693733, 415.03230672824964, 419.75080559954472,
	424.47819341825709, 429.21439186665157, 433.95932399501481,
	438.71291418612117, 443.47508812091894, 448.24577274538461,
	453.02489623849613, 457.81238798127816, 462.60817852687489,
	467.4121995716082, 472.22438392698058, 477.04466549258564,
	481.87297922988796,
}

func logfactorial(k int64) float64 {
	const halfln2pi = 0.9189385332046728
	if k >= 0 && k < int64(len(logfact)) {
		return logfact[k]
	}
	lf := float64(k)
	return (lf+0.5)*cmath.Log(lf) - lf + (halfln2pi + (1.0/lf)*(1.0/12.0-1.0/(360.0*lf*lf)))
}

// hypergeometricSample: random_hypergeometric.c:53 (counting draw-down).
func (g *Generator) hypergeometricSample(good, bad, sample int64) int64 {
	total := good + bad
	var computedSample int64
	if sample > total/2 {
		computedSample = total - sample
	} else {
		computedSample = sample
	}
	remainingTotal := total
	remainingGood := good
	for computedSample > 0 && remainingGood > 0 && remainingTotal > remainingGood {
		remainingTotal--
		if int64(g.randomInterval(uint64(remainingTotal))) < remainingGood {
			remainingGood--
		}
		computedSample--
	}
	if remainingTotal == remainingGood {
		remainingGood -= computedSample
	}
	if sample > total/2 {
		return remainingGood
	}
	return good - remainingGood
}

const (
	hyperD1 = 1.7155277699214135 // 2*sqrt(2/e)
	hyperD2 = 0.8989161620588988 // 3 - 2*sqrt(3/e)
)

// hypergeometricHRUA: random_hypergeometric.c:116 (Stadlober ratio-of-uniforms).
func (g *Generator) hypergeometricHRUA(good, bad, sample int64) int64 {
	popsize := good + bad
	computedSample := sample
	if popsize-sample < computedSample {
		computedSample = popsize - sample
	}
	mingoodbad := good
	if bad < mingoodbad {
		mingoodbad = bad
	}
	maxgoodbad := good
	if bad > maxgoodbad {
		maxgoodbad = bad
	}
	p := float64(mingoodbad) / float64(popsize)
	q := float64(maxgoodbad) / float64(popsize)
	mu := float64(computedSample) * p
	a := mu + 0.5
	vr := float64(popsize-computedSample) * float64(computedSample) * p * q / float64(popsize-1)
	c := math.Sqrt(vr + 0.5)
	h := hyperD1*c + hyperD2
	m := int64(math.Floor(float64(computedSample+1) * float64(mingoodbad+1) / float64(popsize+2)))
	gg := logfactorial(m) + logfactorial(int64(mingoodbad)-m) +
		logfactorial(int64(computedSample)-m) +
		logfactorial(int64(maxgoodbad)-int64(computedSample)+m)
	bound := computedSample
	if mingoodbad < bound {
		bound = mingoodbad
	}
	floorA := int64(math.Floor(a + 16*c))
	if floorA < bound {
		bound = floorA
	}
	var K int64
	for {
		U := g.nextDouble()
		V := g.nextDouble()
		X := a + h*(V-0.5)/U
		if X < 0.0 || X >= float64(bound)+1.0 {
			continue
		}
		K = int64(math.Floor(X))
		gp := logfactorial(K) + logfactorial(int64(mingoodbad)-K) +
			logfactorial(int64(computedSample)-K) +
			logfactorial(int64(maxgoodbad)-int64(computedSample)+K)
		T := gg - gp
		if U*(4.0-U)-3.0 <= T {
			break
		}
		if U*(U-T) >= 1.0 {
			continue
		}
		if 2.0*cmath.Log(U) <= T {
			break
		}
	}
	if good > bad {
		K = computedSample - K
	}
	if computedSample < sample {
		K = good - K
	}
	return K
}

// Hypergeometric: random_hypergeometric.c:231 — HRUA when
// 10 ≤ sample ≤ total−10, counting otherwise.
func (g *Generator) Hypergeometric(ngood, nbad, nsample int64, size ...int) *arrayI {
	const hypergeomMax = 1e9
	if float64(ngood) >= hypergeomMax || float64(nbad) >= hypergeomMax {
		panic("both ngood and nbad must be less than 1e+09")
	}
	if ngood+nbad < nsample {
		panic("ngood + nbad < nsample")
	}
	n := sizeOr1(size)
	out := newI(n)
	for i := 0; i < n; i++ {
		if nsample >= 10 && nsample <= ngood+nbad-10 {
			out.Data()[i] = g.hypergeometricHRUA(ngood, nbad, nsample)
		} else {
			out.Data()[i] = g.hypergeometricSample(ngood, nbad, nsample)
		}
	}
	return out
}

// NegativeBinomial: distributions.c:614 — Poisson(gamma(n, (1−p)/p)).
func (g *Generator) NegativeBinomial(n float64, p float64, size ...int) *arrayI {
	if !(n > 0) || math.IsNaN(n) {
		panic("n must not be NaN and n <= 0")
	}
	if !(p > 0) || p > 1 {
		panic("p <= 0, p > 1 or p contains NaNs")
	}
	// lam cap: numpy _generator.pyx:3244 POISSON_LAM_MAX check.
	const poissonLamMax = 9223372036854775807.0 - 3037000499.9804847 // int64max - sqrt(int64max)*10
	if (1.0-p)/p*(n+10.0*math.Sqrt(n)) > poissonLamMax {
		panic("n too large or p too small, see Generator.negative_binomial Notes")
	}
	cnt := sizeOr1(size)
	out := newI(cnt)
	for i := 0; i < cnt; i++ {
		Y := (1.0 - p) / p * g.StandardGammaScalar(n)
		out.Data()[i] = g.poissonScalar(Y)
	}
	return out
}

// Dirichlet: _generator.pyx:4409. Small-alpha (max < 0.1) uses the
// stick-breaking Beta construction; otherwise gamma draws normalized by
// 1/sum (numpy multiplies by invacc, not divides).
func (g *Generator) Dirichlet(alpha []float64, size ...int) *arrayF {
	k := len(alpha)
	if k == 0 {
		panic("alpha must be non-empty")
	}
	for _, a := range alpha {
		if a < 0 {
			panic("alpha < 0")
		}
	}
	maxa := alpha[0]
	for _, a := range alpha {
		if a > maxa {
			maxa = a
		}
	}
	rows := sizeOr1(size)
	out := numgo.Zeros[float64](rows, k)
	data := out.Data()
	if maxa < 0.1 {
		// stick-breaking: csum holds right-to-left cumulative sums
		csumArr := make([]float64, k)
		csum := 0.0
		for j := k - 1; j >= 0; j-- {
			csum += alpha[j]
			csumArr[j] = csum
		}
		if csum > 0 {
			for i := 0; i < rows; i++ {
				acc := 1.0
				for j := 0; j < k-1; j++ {
					v := g.betaScalar(alpha[j], csumArr[j+1])
					data[i*k+j] = acc * v
					acc *= 1.0 - v
					if csumArr[j+1] == 0 {
						break
					}
				}
				data[i*k+k-1] = acc
			}
		}
		return out
	}
	for i := 0; i < rows; i++ {
		acc := 0.0
		for j := 0; j < k; j++ {
			v := g.StandardGammaScalar(alpha[j])
			data[i*k+j] = v
			acc = acc + v
		}
		invacc := 1.0 / acc
		for j := 0; j < k; j++ {
			data[i*k+j] = data[i*k+j] * invacc
		}
	}
	return out
}

// Multinomial: _generator.pyx:3959 + random_multinomial
// (distributions.c:1733) — binomial chain over remaining probability.
// Returns int64 counts in a float64 array (numpy uses int64; arrayI
// precedent applies, but the counts belong in the typed float array to
// compose with the rest of the package).
func (g *Generator) Multinomial(n int64, pvals []float64, size ...int) *arrayF {
	d := len(pvals)
	if d == 0 {
		panic("pvals must have at least 1 dimension and the last dimension of pvals must be greater than 0.")
	}
	for _, v := range pvals {
		if !(v >= 0) || v > 1 {
			panic("pvals < 0, pvals > 1 or pvals contains NaNs")
		}
	}
	if kahanSum(pvals[:d-1]) > 1.0+1e-12 {
		panic("sum(pvals[:-1]) > 1.0")
	}
	if n < 0 {
		panic("n < 0")
	}
	rows := sizeOr1(size)
	out := numgo.Zeros[float64](rows, d)
	data := out.Data()
	st := &binomialState{}
	for i := 0; i < rows; i++ {
		remainingP := 1.0
		dn := n
		for j := 0; j < d-1; j++ {
			m := g.binomialScalar(dn, pvals[j]/remainingP, st)
			data[i*d+j] = float64(m)
			dn -= m
			if dn <= 0 {
				break
			}
			remainingP -= pvals[j]
		}
		if dn > 0 {
			data[i*d+d-1] = float64(dn)
		}
	}
	return out
}

// MultivariateNormal: _generator.pyx:3719 with method='cholesky': x =
// standard_normal(rows,dim) then x @ L.T + mean (dot accumulated with FMA per
// step, matching OpenBLAS dgemm's rounding for the small cov verified here).
// numpy warns on non-PSD cov; we panic instead. shortcut: our Potrf's
// accumulation matches LAPACK DPOTF2 only for small matrices (n ≲ block
// size); for large cov, bit parity with the OpenBLAS-backed numpy is not
// guaranteed, upgrade when parity is needed at scale.
func (g *Generator) MultivariateNormal(mean *numgo.Array[float64], cov *numgo.Array[float64], size ...int) *numgo.Array[float64] {
	ms := mean.Shape()
	cs := cov.Shape()
	if len(ms) != 1 {
		panic("mean must be 1 dimensional")
	}
	if len(cs) != 2 || cs[0] != cs[1] {
		panic("cov must be 2 dimensional and square")
	}
	if ms[0] != cs[0] {
		panic("mean and cov must have same length")
	}
	dim := ms[0]
	if dim == 0 {
		panic("mean and cov must have same length")
	}
	rows := sizeOr1(size)
	x := g.StandardNormal(rows, dim).Data()
	covData := append([]float64(nil), cov.Data()...)
	if !lapack.Potrf(covData, dim) {
		panic("covariance is not symmetric positive-semidefinite.")
	}
	out := numgo.Zeros[float64](rows, dim)
	od := out.Data()
	meanData := mean.Data()
	for r := 0; r < rows; r++ {
		// BLAS dgemm (numpy: x @ L.T) accumulates the dot product first,
		// then adds the mean; a fused mean-first loop rounds differently.
		for i := 0; i < dim; i++ {
			// dot product with FMA per step, matching OpenBLAS dgemm.
			v := x[r*dim+0] * covData[i*dim+0]
			for j := 1; j <= i; j++ {
				v = math.FMA(x[r*dim+j], covData[i*dim+j], v)
			}
			od[r*dim+i] = meanData[i] + v
		}
	}
	return out
}

// MultivariateHypergeometricMethod selects the sampling algorithm.
type MultivariateHypergeometricMethod int

const (
	// MVHMarginals iterates univariate hypergeometric draws
	// (random_mvhg_marginals.c); matches numpy's default.
	MVHMarginals MultivariateHypergeometricMethod = iota
	// MVHCount shuffles an expanded choices array (random_mvhg_count.c).
	MVHCount
)

// MultivariateHypergeometric: _generator.pyx:4203. Returns a rows×len(colors)
// int64 count array. Default method 'marginals'.
func (g *Generator) MultivariateHypergeometric(colors []int64, nsample int64, size int, method MultivariateHypergeometricMethod) *arrayI {
	numColors := len(colors)
	total := int64(0)
	for _, c := range colors {
		if c < 0 {
			panic("colors must be a one-dimensional sequence of nonnegative integers")
		}
		total += c
	}
	if nsample < 0 {
		panic("nsample must be nonnegative.")
	}
	if method == MVHMarginals && total >= 1000000000 {
		panic("When method is \"marginals\", sum(colors) must be less than 1000000000.")
	}
	if nsample > total {
		panic("nsample > sum(colors)")
	}
	out := newI(size * numColors)
	if numColors == 0 || total == 0 || nsample == 0 || size == 0 {
		return out
	}
	v := out.Data()
	moreThanHalf := nsample > total/2
	n := nsample
	if moreThanHalf {
		n = total - nsample
	}
	switch method {
	case MVHMarginals:
		for i := 0; i < size*numColors; i += numColors {
			numToSample := n
			remaining := total
			for j := 0; numToSample > 0 && j+1 < numColors; j++ {
				remaining -= colors[j]
				r := g.hypergeometricScalar(colors[j], remaining, numToSample)
				v[i+j] = r
				numToSample -= r
			}
			if numToSample > 0 {
				v[i+numColors-1] = numToSample
			}
			if moreThanHalf {
				for k := 0; k < numColors; k++ {
					v[i+k] = colors[k] - v[i+k]
				}
			}
		}
	case MVHCount:
		choices := make([]int64, total)
		for i, k := 0, 0; i < numColors; i++ {
			for j := int64(0); j < colors[i]; j++ {
				choices[k] = int64(i)
				k++
			}
		}
		for i := 0; i < size*numColors; i += numColors {
			for j := int64(0); j < n; j++ {
				k := j + int64(g.randomInterval(uint64(total-j-1)))
				choices[int(k)], choices[int(j)] = choices[int(j)], choices[int(k)]
			}
			for j := int64(0); j < n; j++ {
				v[i+int(choices[j])]++
			}
			if moreThanHalf {
				for k := 0; k < numColors; k++ {
					v[i+k] = colors[k] - v[i+k]
				}
			}
		}
	}
	return out
}

// poissonScalar: single draw with the distributions.c:604 dispatch.
func (g *Generator) poissonScalar(lam float64) int64 {
	switch {
	case lam >= 10:
		return g.poissonPTRS(lam)
	case lam == 0:
		return 0
	default:
		return g.poissonMult(lam)
	}
}

// hypergeometricScalar: random_hypergeometric dispatch for one draw.
func (g *Generator) hypergeometricScalar(good, bad, sample int64) int64 {
	if sample >= 10 && sample <= good+bad-10 {
		return g.hypergeometricHRUA(good, bad, sample)
	}
	return g.hypergeometricSample(good, bad, sample)
}

// glibc dbl-64 bit-parity helpers used by VonMises (cos, acos) and the
// inverse-CDF distributions (expm1). The Go standard library versions differ
// from glibc by 1 ULP on ~25%/31%/0.25% of inputs, which is enough to break
// bit-exact streams at rejection boundaries. Ports of the glibc algorithms
// (LGPL 2.1, Free Software Foundation); table data from asincos.tbl/root.tbl.

// expm1G: port of glibc s_expm1.c __expm1 (FMA contraction order as built on
// x86-64 with -mfma). Verified bit-exact vs the platform libm on 8M samples
// across the domains these distributions reach.
func expm1G(x float64) float64 {
	const (
		huge    = 1.0e+300
		tiny    = 1.0e-300
		oThresh = 7.09782712893383973096e+02
		ln2hi   = 6.93147180369123816490e-01
		ln2lo   = 1.90821492927058770002e-10
		invln2  = 1.44269504088896338700e+00
		q1      = -3.33333333333331316428e-02
		q2      = 1.58730158725481460165e-03
		q3      = -7.93650757867487942473e-05
		q4      = 4.00821782732936239552e-06
		q5      = -2.01099218183624371326e-07
	)
	var y, hi, lo, c, t, e, hxs, hfx, r1, h2, h4, R1, R2, R3 float64
	var k int32
	var xsb uint32
	ux := math.Float64bits(x)
	xsb = uint32(ux >> 63)
	if xsb == 0 {
		y = x
	} else {
		y = -x
	}
	hx := uint32(ux>>32) & 0x7fffffff
	if hx >= 0x4043687A {
		if hx >= 0x40862E42 {
			if hx >= 0x7ff00000 {
				if (hx&0xfffff)|(uint32(ux)) != 0 {
					return x + x
				}
				if xsb == 0 {
					return x
				}
				return -1.0
			}
			if x > oThresh {
				return math.Inf(1)
			}
		}
		if xsb != 0 {
			return tiny - 1.0
		}
	}
	if hx > 0x3fd62e42 {
		if hx < 0x3FF0A2B2 {
			if xsb == 0 {
				hi = x - ln2hi
				lo = ln2lo
				k = 1
			} else {
				hi = x + ln2hi
				lo = -ln2lo
				k = -1
			}
		} else {
			if xsb == 0 {
				k = int32(invln2*x + 0.5)
			} else {
				k = int32(invln2*x - 0.5)
			}
			t = float64(k)
			hi = x - t*ln2hi
			lo = t * ln2lo
		}
		x = hi - lo
		c = (hi - x) - lo
	} else if hx < 0x3c900000 {
		t = huge + x
		return x - (t - (huge + x))
	}
	hfx = 0.5 * x
	hxs = x * hfx
	R1 = math.FMA(hxs, q1, 1.0)
	h2 = hxs * hxs
	R2 = math.FMA(hxs, q3, q2)
	h4 = h2 * h2
	R3 = math.FMA(hxs, q5, q4)
	r1 = math.FMA(h4, R3, math.FMA(h2, R2, R1))
	t = math.FMA(-r1, hfx, 3.0)
	e = hxs * ((r1 - t) / math.FMA(-x, t, 6.0))
	if k == 0 {
		return x - math.FMA(x, e, -hxs)
	}
	e = math.FMA(x, e-c, -c)
	e -= hxs
	if k == -1 {
		return 0.5*(x-e) - 0.5
	}
	if k == 1 {
		if x < -0.25 {
			return math.FMA(-2.0, e, 2.0*(x+0.5))
		}
		return math.FMA(2.0, x-e, 1.0)
	}
	if k <= -2 || k > 56 {
		y = 1.0 - (e - x)
		y = math.Float64frombits(math.Float64bits(y) + uint64(int64(k)<<52))
		return y - 1.0
	}
	t = 1.0
	if k < 20 {
		// SET_HIGH_WORD replaces only the high 32 bits (low word of 1.0 is 0).
		t = math.Float64frombits(uint64(0x3ff00000-(0x200000>>uint(k))) << 32)
		y = t - (e - x)
		y = math.Float64frombits(math.Float64bits(y) + uint64(int64(k)<<52))
	} else {
		t = math.Float64frombits(uint64(0x3ff-k) << 52)
		y = x - (e + t)
		y += 1.0
		y = math.Float64frombits(math.Float64bits(y) + uint64(int64(k)<<52))
	}
	return y
}

// cosG: faithful port of glibc 2.39 sysdeps/ieee754/dbl-64/s_sin.c __cos —
// the implementation this machine's libm actually ships (the file was
// rewritten upstream around 2.36-2.39; the old IBM big-table do_cos/do_sin
// that older mirrors show is NOT what numpy links against here). Verified
// 0/8M mismatches vs the machine libm on a [0,3.2] sweep (see
// /tmp/opencode/genref/cosreal2.c).
func cosG(x float64) float64 {
	const (
		hpinv = 0x1.45F306DC9C883p-1
		toint = 0x1.8000000000000p52
		mp1   = 0x1.921FB58000000p0
		mp2   = -0x1.DDE973C000000p-27
		pp3   = -0x1.CB3B398000000p-55
		pp4   = -0x1.d747f23e32ed7p-83
		hp0c  = 0x1.921FB54442D18p0
		hp1c  = 0x1.1A62633145C07p-54
	)
	ux := math.Float64bits(x)
	k := uint32(ux>>32) & 0x7fffffff
	if k < 0x3e400000 {
		return 1.0
	}
	if k < 0x3feb6000 {
		return doCos(x, 0)
	}
	if k < 0x400368fd {
		y := hp0c - math.Abs(x)
		a := y + hp1c
		da := (y - a) + hp1c
		return doSin(a, da)
	}
	if k < 0x419921FB {
		t := x*hpinv + toint
		xn := t - toint
		n := uint32(math.Float64bits(t)) & 3
		y := (x - xn*mp1) - xn*mp2
		t1 := xn * pp3
		t2 := y - t1
		db := (y - t2) - t1
		t1 = xn * pp4
		b := t2 - t1
		db += (t2 - b) - t1
		var retval float64
		m := (n + 1) & 3
		if m&1 != 0 {
			retval = doCos(b, db)
		} else {
			retval = doSin(b, db)
		}
		if m&2 != 0 {
			return -retval
		}
		return retval
	}
	return math.Cos(x)
}

// doCos: glibc 2.39 s_sin.c do_cos; big = 0x1.8p45 puts the reduced residual
// into the low mantissa word so scostab indexes directly.
func doCos(x, dx float64) float64 {
	const big = 0x1.8000000000000p45
	if x < 0 {
		dx = -dx
	}
	u := math.Abs(x) + big
	xx := math.Abs(x) - (u - big) + dx
	const (
		sn3 = -1.66666666666664880952546298448555e-01
		sn5 = 8.33333214285722277379541354343671e-03
		cs2 = 4.99999999999999999999950396842453e-01
		cs4 = -4.16666666666664434524222570944589e-02
		cs6 = 1.38888874007937613028114285595617e-03
	)
	xxx := xx * xx
	s := xx + xx*xxx*(sn3+xxx*sn5)
	c := xxx * (cs2 + xxx*(cs4+xxx*cs6))
	k := int32(uint32(math.Float64bits(u))) << 2
	sn := scostab[k]
	ssn := scostab[k+1]
	cs := scostab[k+2]
	ccs := scostab[k+3]
	cor := (ccs - s*ssn - cs*c) - sn*s
	return cs + cor
}

// doSin: glibc 2.39 s_sin.c do_sin.
func doSin(x, dx float64) float64 {
	const big = 0x1.8000000000000p45
	const (
		// taylor-path coefficients (s1..s5) and the do_sin polynomial
		// coefficients (sn3/sn5) are distinct constants; keep them apart.
		sn1 = -0x1.5555555555555p-3
		sn2 = 0x1.1111111110ECEp-7
		ts3 = -0x1.A01A019DB08B8p-13
		ts4 = 0x1.71DE27B9A7ED9p-19
		ts5 = -0x1.ADDFFC2FCDF59p-26
		sn3 = -1.66666666666664880952546298448555e-01
		sn5 = 8.33333214285722277379541354343671e-03
		cs2 = 4.99999999999999999999950396842453e-01
		cs4 = -4.16666666666664434524222570944589e-02
		cs6 = 1.38888874007937613028114285595617e-03
	)
	xold := x
	if math.Abs(x) < 0.126 {
		// TAYLOR_SIN: t = ((POLYNOMIAL(xx)*x - dx/2)*xx + dx); res = x + t
		z := x * x
		poly := ((((ts5*z+ts4)*z+ts3)*z+sn2)*z + sn1)
		t := (poly*x-0.5*dx)*z + dx
		return x + t
	}
	if x <= 0 {
		dx = -dx
	}
	u := math.Abs(x) + big
	x = math.Abs(x) - (u - big)
	xxx := x * x
	s := x + (dx + x*xxx*(sn3+xxx*sn5))
	c := x*dx + xxx*(cs2+xxx*(cs4+xxx*cs6))
	k := int32(uint32(math.Float64bits(u))) << 2
	sn := scostab[k]
	ssn := scostab[k+1]
	cs := scostab[k+2]
	ccs := scostab[k+3]
	cor := (ssn + s*ccs - sn*c) + cs*s
	r := sn + cor
	return math.Copysign(r, xold)
}

// acosG: port of glibc e_asin.c __ieee754_acos (post-2021 no-slow-path
// version) with the asincos/root tables.
func acosG(x float64) float64 {
	const (
		hp0 = 1.5707963267948966
		hp1 = 6.123233995736766e-17
		f1  = 1.66666666666664110590506577996662e-01
		f2  = 7.50000000026122686814431784722623e-02
		f3  = 4.46428561421059750978517350006940e-02
		f4  = 3.03821268582119319911193410625235e-02
		f5  = 2.23551211026525610742786300334557e-02
		f6  = 1.81382903404565056280372531963613e-02
		rt0 = 9.99999999859990725855365213134618e-01
		rt1 = 4.99999999495955425917856814202739e-01
		rt2 = 3.75017500867345182581453026130850e-01
		rt3 = 3.12523626554518656309172508769531e-01
		t27 = 134217728.0
	)
	var x2, xx, p, t, res, r, cor, cc, y, z float64
	ux := math.Float64bits(x)
	m := uint32(ux >> 32)
	k := m & 0x7fffffff
	switch {
	case k < 0x3c880000:
		return hp0
	case k < 0x3fc00000:
		x2 = x * x
		t = (((((f6*x2+f5)*x2+f4)*x2+f3)*x2+f2)*x2 + f1) * (x2 * x)
		r = hp0 - x
		cor = (((hp0 - r) - x) + hp1) - t
		res = r + cor
		return res
	case k < 0x3fe00000:
		var n int
		if k < 0x3fd00000 {
			n = 11 * int((k&0x000fffff)>>15)
		} else {
			n = 11*int((k&0x000fffff)>>14) + 352
		}
		if m < 0x80000000 {
			xx = x - asncs[n]
		} else {
			xx = -x - asncs[n]
		}
		t = asncs[n+1] * xx
		p = xx*xx*(asncs[n+2]+xx*(asncs[n+3]+xx*(asncs[n+4]+
			xx*(asncs[n+5]+xx*asncs[n+6])))) + asncs[n+7]
		t += p
		if m < 0x80000000 {
			y = hp0 - asncs[n+8]
			t = hp1 - t
		} else {
			y = hp0 + asncs[n+8]
			t = hp1 + t
		}
		return y + t
	case k < 0x3fe80000:
		n := 1056 + int((k&0x000fe000)>>11)*3
		if m < 0x80000000 {
			xx = x - asncs[n]
		} else {
			xx = -x - asncs[n]
		}
		t = asncs[n+1] * xx
		p = xx*xx*(asncs[n+2]+xx*(asncs[n+3]+xx*(asncs[n+4]+
			xx*(asncs[n+5]+xx*(asncs[n+6]+xx*asncs[n+7]))))) + asncs[n+8]
		t += p
		if m < 0x80000000 {
			y = hp0 - asncs[n+9]
			t = hp1 - t
		} else {
			y = hp0 + asncs[n+9]
			t = hp1 + t
		}
		return y + t
	case k < 0x3fed8000:
		n := 992 + int((k&0x000fe000)>>13)*13
		if m < 0x80000000 {
			xx = x - asncs[n]
		} else {
			xx = -x - asncs[n]
		}
		t = asncs[n+1] * xx
		p = xx*xx*(asncs[n+2]+xx*(asncs[n+3]+xx*(asncs[n+4]+
			xx*(asncs[n+5]+xx*(asncs[n+6]+xx*(asncs[n+7]+
				xx*asncs[n+8])))))) + asncs[n+9]
		t += p
		if m < 0x80000000 {
			y = hp0 - asncs[n+10]
			t = hp1 - t
		} else {
			y = hp0 + asncs[n+10]
			t = hp1 + t
		}
		return y + t
	case k < 0x3fee8000:
		n := 884 + int((k&0x000fe000)>>13)*14
		if m < 0x80000000 {
			xx = x - asncs[n]
		} else {
			xx = -x - asncs[n]
		}
		t = asncs[n+1] * xx
		p = xx*xx*(asncs[n+2]+xx*(asncs[n+3]+xx*(asncs[n+4]+
			xx*(asncs[n+5]+xx*(asncs[n+6]+xx*(asncs[n+7]+
				xx*(asncs[n+8]+xx*asncs[n+9]))))))) + asncs[n+10]
		t += p
		if m < 0x80000000 {
			y = hp0 - asncs[n+11]
			t = hp1 - t
		} else {
			y = hp0 + asncs[n+11]
			t = hp1 + t
		}
		return y + t
	case k < 0x3fef0000:
		n := 768 + int((k&0x000fe000)>>13)*15
		if m < 0x80000000 {
			xx = x - asncs[n]
		} else {
			xx = -x - asncs[n]
		}
		t = asncs[n+1] * xx
		p = xx*xx*(asncs[n+2]+xx*(asncs[n+3]+xx*(asncs[n+4]+
			xx*(asncs[n+5]+xx*(asncs[n+6]+xx*(asncs[n+7]+
				xx*(asncs[n+8]+xx*(asncs[n+9]+xx*asncs[n+10])))))))) + asncs[n+11]
		t += p
		if m < 0x80000000 {
			y = hp0 - asncs[n+12]
			t = hp1 - t
		} else {
			y = hp0 + asncs[n+12]
			t = hp1 + t
		}
		return y + t
	case k < 0x3ff00000:
		if m < 0x80000000 {
			z = 0.5 * (1.0 - x)
		} else {
			z = 0.5 * (1.0 + x)
		}
		kh := uint32(math.Float64bits(z) >> 32)
		t = inroot[(kh&0x001fffff)>>14] * powtwo[511-int(kh>>21)]
		r = 1.0 - t*t*z
		t = t * (rt0 + r*(rt1+r*(rt2+r*rt3)))
		cc = t * z
		t = cc * (1.5 - 0.5*t*cc)
		y = (t27*cc + cc) - t27*cc
		cc = (z - y*y) / (t + y)
		p = (((((f6*z+f5)*z+f4)*z+f3)*z+f2)*z + f1) * z
		if m >= 0x80000000 {
			cor = (hp1 - cc) - (y+cc)*p
			res = hp0 - y
			res = res + cor
		} else {
			cor = cc + p*(y+cc)
			res = y + cor
		}
		return res + res
	default:
		if k == 0x3ff00000 && uint32(ux) == 0 {
			if m < 0x80000000 {
				return 0
			}
			return 2.0 * hp0
		}
		return (x - x) / (x - x)
	}
}

var asncs = [2568]float64{
	0.126953125, 1.0081572852980196, 0.06504265768461821, 0.17917130396388092, 0.05093475237068401, 0.09114942972872477, 0.045252377403287906, 7.878874731570043e-17, 0.12729664158205622, 1.015625, -0.0074677147019804845, 0.130859375, 1.0086736451227316, 0.06714703910712035, 0.17998112217431844, 0.05272549853549236, 0.09223078838601706, 0.047029800773477376, -4.1203553019903806e-17, 0.13123575914132957, 1.015625, -0.006951354877268421, 0.134765625, 1.0092064829260585, 0.06926107518525271, 0.1808190879245562, 0.054537774523911814, 0.09335425256308633, 0.04884565826512823, -4.667691592241719e-17, 0.13517692589016717, 1.015625, -0.006418517073941556, 0.138671875, 1.0097558754286717, 0.07138509776934356, 0.18168554408913465, 0.05637241162634303, 0.0945207427990774, 0.05070164331047607, -1.9145401153849117e-16, 0.13912020634467437, 1.015625, -0.005869124571328324, 0.142578125, 1.0103219019597183, 0.07351944280444374, 0.182580846674003, 0.05823025950398843, 0.09573121996931573, 0.05259950495152674, 1.1967809786630907e-16, 0.14306566532573006, 1.015625, -0.005303098040281641, 0.146484375, 1.0109046444894167, 0.07566445048591644, 0.1835053651097847, 0.06011218698959373, 0.0969866866211576, 0.054541050633075745, 4.793514507244096e-17, 0.14701336796924158, 1.015625, -0.004720355510583352, 0.150390625, 1.0115041876628805, 0.07782046541853795, 0.18445948255776173, 0.06201908291492843, 0.09828818837674154, 0.05652814911973874, -3.33968423657497e-17, 0.1509633797365244, 1.015625, -0.004120812337119514, 0.154296875, 1.0121206188352014, 0.07998783677926057, 0.1854435962290061, 0.06395185696634055, 0.09963681540563037, 0.05856273354413679, 1.251927041212157e-16, 0.1549157664248204, 1.015625, -0.0035043811647984934, 0.158203125, 1.0127540281078173, 0.0821669184837947, 0.1864581177171075, 0.06591144056978143, 0.10103370397050522, 0.06064680459377388, -1.448779841181967e-16, 0.15887059417795513, 1.015625, -0.0028709718921826723, 0.162109375, 1.0134045083661967, 0.08435806935717204, 0.18750347334496914, 0.06789878780675966, 0.10248003804925487, 0.06278243384457469, 1.8057575822642167e-16, 0.1628279294971362, 1.015625, -0.002220491633803283, 0.166015625, 1.0140721553188734, 0.08656165330845753, 0.18858010452616727, 0.06991487636275023, 0.10397705103699525, 0.06497176724949832, -5.889255618474885e-17, 0.1667878392519082, 1.015625, -0.0015528446811266235, 0.169921875, 1.0147570675378614, 0.08877803950978395, 0.18968846814139292, 0.07196070850965641, 0.10552602753175724, 0.0672170287911294, 8.408803988489449e-17, 0.17075039069125442, 1.015625, -0.0008679324621385662, 0.173828125, 1.015459346500487, 0.09100760257988891, 0.190829036930522, 0.0740373121239965, 0.10712830520779507, 0.06952052430766366, -1.6921221440675783e-16, 0.17471565145486734, 1.015625, -0.00016565349951288934, 0.177734375, 1.0161790966326725, 0.09325072277234074, 0.19200229990088472, 0.07614574174256729, 0.10878527678069519, 0.07188464550225171, 8.763277999991739e-17, 0.17868368958457825, 1.015625, 0.0005540966326724648, 0.181640625, 1.0169164253537086, 0.09550778616864669, 0.1932087627523345, 0.07828707965741985, 0.11049839206870538, 0.07431187414625177, -1.3668732051802496e-16, 0.1826545735359657, 1.015625, 0.0012914253537085914, 0.185546875, 1.0176714431225573, 0.09777918487644385, 0.19444894831974563, 0.08046243705207219, 0.11226916015495843, 0.07680478648756596, 2.313780080787208e-17, 0.1866283721901345, 1.015625, 0.002046443122557203, 0.189453125, 1.0184442634857211, 0.1000653172329809, 0.19572339703359914, 0.08267295518097714, 0.1140991516555348, 0.0793660578759039, 1.3782592633632847e-18, 0.19060515486568574, 1.015625, 0.0028192634857211064, 0.193359375, 1.0192350031267257, 0.10236658801410674, 0.19703266739934994, 0.08491980659436302, 0.11599000109859382, 0.08199846761752315, -1.0084570173415735e-16, 0.19458499133087148, 1.015625, 0.0036100031267257463, 0.197265625, 1.0200437819172556, 0.1046834086489903, 0.19837733649630057, 0.0872041964106701, 0.11794340942010645, 0.08470490407275999, 1.3542001225472393e-16, 0.19856795181594844, 1.015625, 0.004418781917255714, 0.201171875, 1.0208707229699918, 0.10701619744080444, 0.1997580004967441, 0.08952736363891674, 0.11996114658204386, 0.08748837001047241, -1.2471003877016987e-16, 0.20255410702573595, 1.015625, 0.005245722969991896, 0.205078125, 1.0217159526931967, 0.1093653797936163, 0.20117527520617465, 0.09189058255344643, 0.12204505431921703, 0.0903519882343823, 8.288583961183415e-17, 0.20654352815237909, 1.015625, 0.006090952693196798, 0.208984375, 1.0225796008470975, 0.11173138844573564, 0.20262979662540434, 0.09429516412363241, 0.12419704902132465, 0.09329900749722662, -8.41716770269347e-17, 0.21053628688833514, 1.015625, 0.00695460084709751, 0.212890625, 1.0234618006021179, 0.11411466370978329, 0.20412222153546644, 0.09674245750124723, 0.12641912475715092, 0.09633280871961812, -1.9948269918303676e-16, 0.21453245543957733, 1.03125, -0.007788199397882216, 0.216796875, 1.0243626885990127, 0.11651565371975177, 0.20565322810622735, 0.09923385156834477, 0.128713356448263, 0.09945691153156794, -1.365488869783207e-16, 0.2185321065390342, 1.03125, -0.006887311400987228, 0.220703125, 1.0252824050109621, 0.11893481468534181, 0.20722351652967666, 0.1017707765486499, 0.1310819031999922, 0.10267498115575122, -7.326827026400958e-17, 0.22253531346026634, 1.03125, -0.005967594989037833, 0.224609375, 1.0262210936076797, 0.1213726111538697, 0.2088338096789114, 0.10435470568560731, 0.13352701179794318, 0.10599083565280251, -5.15443605413632e-17, 0.2265421500313889, 1.03125, -0.005028906392320281, 0.228515625, 1.0271789018215995, 0.12382951628005293, 0.2104848537938825, 0.1069871569904069, 0.13605102037876737, 0.1094084535502184, 1.9766878820033888e-17, 0.23055269064924921, 1.03125, -0.004071098178400521, 0.232421875, 1.0281559808162015, 0.1263060121039941, 0.21217741919502298, 0.10966969506347854, 0.1386563622844568, 0.11293198187782276, 1.5898364401634607e-16, 0.23456701029386728, 1.03125, -0.003094019183798456, 0.236328125, 1.0291524855565426, 0.12880258983769677, 0.21391230102593525, 0.11240393299313584, 0.1413455701099709, 0.11656574463422824, -5.065159340375441e-17, 0.23858518454314748, 1.03125, -0.002097514443457328, 0.240234375, 1.0301685748820597, 0.1313197501604611, 0.21569032002637234, 0.11519153433524479, 0.1441212799545998, 0.12031425171030698, 1.0371781056962469e-16, 0.24260728958786837, 1.03125, -0.0010814251179402546, 0.244140625, 1.031204411581717, 0.13385800352352129, 0.2175123233368107, 0.11803421517800368, 0.14698623588809548, 0.12418220829737629, -1.2716373645965909e-16, 0.24663340224696562, 1.03125, -4.5588418283120364e-05, 0.248046875, 1.03226016247157, 0.13641787046430373, 0.21937918533597847, 0.12093374629614147, 0.14994329464327477, 0.1281745248096205, 3.9048168749671166e-17, 0.2506635999831075, 1.03125, 0.0010101624715698385, 0.251953125, 1.0333359984748245, 0.13899988193069984, 0.22129180851277086, 0.12389195539907819, 0.15299543054751066, 0.132296327352209, 2.1465222946214062e-16, 0.254697960918584, 1.03125, 0.0020859984748244497, 0.255859375, 1.034432094704469, 0.14160457961576545, 0.22325112437405797, 0.12691072947784027, 0.15614574070629067, 0.1365529687686569, -1.0212277931368311e-16, 0.2587365638515098, 1.03125, 0.003182094704468856, 0.259765625, 1.0355486305485628, 0.14423251630327652, 0.2252580943899678, 0.1299920172557889, 0.15939745045283338, 0.14095004030320588, 1.791108867431308e-16, 0.26277948827235464, 1.03125, 0.004298630548562728, 0.263671875, 1.03668578975827, 0.14688425622459017, 0.22731371097830733, 0.13313783174850186, 0.16275391907861983, 0.14549338391639854, 2.1274762528915836e-16, 0.26682681438081746, 1.03125, 0.005435789758269993, 0.267578125, 1.0378437605387252, 0.1495603754272796, 0.22941899852987088, 0.13635025293844683, 0.16621864586062143, 0.15018910529459092, -2.0953145061503292e-16, 0.27087862310304356, 1.03125, 0.00659376053872522, 0.271484375, 1.0390227356428277, 0.1522614621560329, 0.2315750144764759, 0.13963143057040434, 0.16979527640199293, 0.1550435875969043, 1.712099471381547e-16, 0.27493499610920447, 1.03125, 0.007772735642827548, 0.275390625, 1.04022291246806, 0.15498811724632772, 0.23378285040366123, 0.14298358707393313, 0.17348760930405244, 0.16006350598608332, 1.3828736966543066e-16, 0.2789960158314546, 1.046875, -0.006652087531940054, 0.279296875, 1.0414444931564357, 0.15774095453141687, 0.23604363321008423, 0.1464090206195328, 0.17729960318849697, 0.1652558429929079, 1.8721053810268642e-16, 0.28306176548226425, 1.046875, -0.0054305068435643, 0.283203125, 1.0426876846976783, 0.16052060126318465, 0.23835852631576102, 0.1499101083155377, 0.1812353840900049, 0.17062790476722575, 1.763864013289322e-16, 0.28713232907315556, 1.046875, -0.004187315302321717, 0.287109375, 1.0439526990357453, 0.1633276985474595, 0.24072873092140537, 0.15348930955318363, 0.18529925324066268, 0.1761873382723502, 1.7709155920484209e-16, 0.2912077914338451, 1.046875, -0.002922300964254752, 0.291015625, 1.0452397531788113, 0.16616290179439527, 0.24315548732124295, 0.1571491695077204, 0.18949569526903004, 0.18194214948351983, 4.538885956161899e-17, 0.2952882382318105, 1.046875, -0.001635246821188551, 0.294921875, 1.0465490693128319, 0.16902688118456285, 0.2456400762718037, 0.1608923228039025, 0.1938293868381279, 0.1879007226553708, -8.415121744019268e-17, 0.29937375599229243, 1.046875, -0.00032593068716818536, 0.298828125, 1.047880874918809, 0.17192032215142275, 0.24818382041932843, 0.16472149735468103, 0.19830520574821056, 0.19407184072795083, -1.4444725450698606e-16, 0.30346443211874874, 1.046875, 0.0010058749188089976, 0.302734375, 1.0492354028938966, 0.17484392588088232, 0.250788085788569, 0.16863951838243943, 0.20292824053186856, 0.20046470694572732, 8.139008080988062e-17, 0.3075603549137731, 1.046875, 0.0023604028938966127, 0.306640625, 1.0506128916764743, 0.1777984098286737, 0.25345428333591014, 0.17264931263267316, 0.20770380057081625, 0.20708896776936114, 1.1240518472364133e-16, 0.3116616136004957, 1.046875, 0.003737891676474387, 0.310546875, 1.0520135853753374, 0.18078450825632428, 0.2561838705699016, 0.17675391279060365, 0.21263742676565328, 0.21395473716572222, 2.137814819776581e-16, 0.3157682983444783, 1.046875, 0.005138585375337436, 0.314453125, 1.053437733903148, 0.18380297278652863, 0.2589783532424575, 0.18095646211184888, 0.21773490279196417, 0.22107262236780584, 1.747582442337454e-16, 0.31988050027612447, 1.046875, 0.00656273390314806, 0.318359375, 1.0548855931143049, 0.1868545729787699, 0.2618392871141578, 0.18526021927894595, 0.22300226697834533, 0.22845375120284367, -6.325580271669912e-17, 0.32399831151361846, 1.0625, -0.0076144068856950865, 0.322265625, 1.0563574249473913, 0.18994009692608008, 0.2647682797972794, 0.18966856349623898, 0.22844582484433565, 0.23610980109408777, 1.005350688829288e-16, 0.3281218251864111, 1.0625, -0.006142575052608797, 0.326171875, 1.0578534975723706, 0.19306035187387247, 0.2677669926803821, 0.1941849998364118, 0.23407216233878844, 0.24405302984948515, 9.858397164616177e-17, 0.33225113545927387, 1.0625, -0.004646502427629333, 0.330078125, 1.0593740855427076, 0.19621616486182514, 0.27083714293849226, 0.19881316485276398, 0.2398881598219729, 0.2522963083588196, 4.149251678685412e-17, 0.33638633755693403, 1.0625, -0.003125914457292424, 0.333984375, 1.0609194699525948, 0.19940838338984382, 0.2739805056331502, 0.20355683247220155, 0.24590100683764718, 0.26085315532993647, 7.2895112537205e-17, 0.34052752778931605, 1.0625, -0.0015805300474051763, 0.337890625, 1.0624899385994824, 0.20263787610918277, 0.27719891590683254, 0.20841992018484873, 0.25211821772451914, 0.2697377742044221, -2.0451869936385863e-16, 0.34467480357740765, 1.0625, -1.0061400517597182e-05, 0.341796875, 1.0640857861521071, 0.2059055335398581, 0.28049427127651355, 0.21340649554718358, 0.25854764811992353, 0.2789650924036682, 2.1063365223795627e-16, 0.3488282634797688, 1.0625, 0.0015857861521072153, 0.345703125, 1.0657073143242342, 0.20921226881554364, 0.2838685340314034, 0.21852078301666944, 0.265197512412213, 0.288550803067669, -1.2964673185058569e-16, 0.352988007219714, 1.0625, 0.0032073143242342303, 0.349609375, 1.0673548320543285, 0.21255901845720201, 0.2873237337401917, 0.22376717113699365, 0.2720764022023089, 0.29851140946125504, -1.6524759645465967e-17, 0.3571541357131758, 1.0625, 0.004854832054328553, 0.353515625, 1.069028655691388, 0.21594674317676618, 0.29086196987343027, 0.22915022009424776, 0.279193305839107, 0.30886427223584634, -9.961363006452305e-17, 0.3613267510972915, 1.0625, 0.006528655691387994, 0.357421875, 1.0707291091871762, 0.21937642871225574, 0.29448541454701976, 0.23467466966569098, 0.28655762909800886, 0.31962765974930335, -4.333551120224562e-17, 0.3655059567597214, 1.078125, -0.007395890812823732, 0.361328125, 1.072456524295108, 0.22284908669578368, 0.2981963153931117, 0.24034544758413828, 0.29417921707677924, 0.3308208016621559, -1.3842604106579423e-16, 0.36969185736873467, 1.078125, -0.005668475704891958, 0.365234375, 1.0742112407760487, 0.2263657555559855, 0.30199699856510975, 0.24616767834251593, 0.30206837738824277, 0.34246394604552494, 2.8940448021537664e-17, 0.3738845589040829, 1.078125, -0.003913759223951268, 0.369140625, 1.0759936066113043, 0.2299275014564834, 0.3058898718838502, 0.2521466924647345, 0.31023590473506224, 0.35457842025452485, 8.776680357312924e-17, 0.37808416868869443, 1.078125, -0.0021313933886957953, 0.373046875, 1.0778039782230886, 0.23353541927208332, 0.30987742813246355, 0.2582880362707544, 0.31869310695801945, 0.3671866958409832, 1.2392075411925148e-16, 0.38229079542121225, 1.078125, -0.0003210217769115026, 0.376953125, 1.0796427207027712, 0.23719063360449333, 0.3139622485078679, 0.26459748216556883, 0.3274518326558912, 0.38031245780109246, -1.9034621058781233e-16, 0.38650454920941035, 1.078125, 0.0015177207027712565, 0.380859375, 1.081510208047221, 0.24089429983944813, 0.31814700623732606, 0.2710810394838128, 0.33652450048221344, 0.3939806784772571, 2.217478766955122e-16, 0.3907255416045148, 1.078125, 0.003385208047220964, 0.384765625, 1.0834068234035752, 0.24464760524722548, 0.3224344703690098, 0.27774496592384107, 0.34592413023200697, 0.40821769645911676, 2.0209797457906482e-16, 0.3949538856364696, 1.078125, 0.005281823403575117, 0.388671875, 1.0853329593227845, 0.248451770128649, 0.3268275097460597, 0.28459577960740384, 0.3556643758399462, 0.42305130085668896, -2.1039947172514095e-17, 0.39918969585016884, 1.078125, 0.007207959322784478, 0.392578125, 1.0872890180222954, 0.252308049008786, 0.331329097174215, 0.2916402718035121, 0.36575956042054625, 0.4385108213489966, -4.574600661435591e-17, 0.4034330883427013, 1.09375, -0.0064609819777047, 0.396484375, 1.0892754116582513, 0.25621773188067076, 0.33594231379370953, 0.2988855203577278, 0.37622471349077813, 0.45462722444468057, 5.3699808164695436e-18, 0.4076841808016396, 1.09375, -0.0044745883417488305, 0.400390625, 1.0912925626076149, 0.2601821455015134, 0.3406703536667961, 0.3063389038709592, 0.3870756105261714, 0.4714332164271756, -4.033572952946113e-17, 0.4119430925444103, 1.09375, -0.002457437392385179, 0.404296875, 1.09334090376063, 0.2642026547439901, 0.3455165285929752, 0.31400811667490325, 0.3983288150129919, 0.4889633534963527, -5.081418033367548e-17, 0.41620994455878657, 1.09375, -0.0004090962393700039, 0.408203125, 1.0954208788240642, 0.268280664005359, 0.3504842731647648, 0.3219011846545728, 0.41000172317157657, 0.5072541596614236, 1.463402191819868e-16, 0.4204848595445445, 1.09375, 0.0016708788240643137, 0.412109375, 1.0975329426356961, 0.27241761867729863, 0.35557715007766416, 0.33002648197189505, 0.4221126115394507, 0.5263442529866845, -1.8599811798684221e-16, 0.4247679619563267, 1.09375, 0.0037829426356961134, 0.416015625, 1.0996775614905314, 0.2766150066795327, 0.3607988557088369, 0.3383927487481949, 0.43468068761754064, 0.5462744808427636, 1.216583914929538e-16, 0.4290593780477532, 1.09375, 0.005927561490531526, 0.419921875, 1.1018552134792627, 0.2808743600604793, 0.3661532259799751, 0.3470091097674991, 0.4477261437987395, 0.5670880648718353, -1.9326635712008848e-16, 0.43335923591683745, 1.109375, -0.007519786520737369, 0.423828125, 1.1040663888395037, 0.28519725666835044, 0.3716442425208067, 0.35588509426704745, 0.46127021481538755, 0.5888307564362442, 1.192295450077398e-16, 0.4376676655527403, 1.109375, -0.005308611160496399, 0.427734375, 1.1063115903203693, 0.28958532189632785, 0.3772760391507829, 0.3650306568861976, 0.4753352389610369, 0.6115510033866788, -1.092739366153726e-16, 0.44198479888393116, 1.109375, -0.0030634096796307744, 0.431640625, 1.1085913335609894, 0.2940402305056494, 0.3830529086976373, 0.3744561998500944, 0.48994472336230743, 0.6353001290590183, -1.9885737010463525e-16, 0.4463107698277917, 1.109375, -0.000783666439010593, 0.435546875, 1.110906147483584, 0.2985637085306727, 0.38897931017273973, 0.38417259647007485, 0.50512341359887, 0.6601325244888747, -8.560186794717833e-18, 0.4506457143417326, 1.109375, 0.0015311474835840086, 0.439453125, 1.113256574701755, 0.3031575352702173, 0.3950598763245009, 0.3941912160488345, 0.5208973679937868, 0.68610585492041, -1.2279261673440052e-16, 0.4549897704758732, 1.109375, 0.0038815747017551597, 0.443359375, 1.1156431719446886, 0.3078235453697493, 0.40129942159250565, 0.40452395028492594, 0.5372940369227738, 0.713281281781972, 1.0547291732873668e-16, 0.4593430784273447, 1.109375, 0.0062681719446887165, 0.447265625, 1.1180665104979932, 0.3125636309992435, 0.4077029504865874, 0.41518324127824474, 0.5543423475196515, 0.7417237014063784, 4.4155691192808335e-17, 0.4637057805962872, 1.125, -0.006933489502006714, 0.451171875, 1.1205271766619442, 0.3173797441318533, 0.4142756664167021, 0.4261821112458276, 0.5720727941865285, 0.7715020018892488, -7.217009118367814e-18, 0.4680780216435978, 1.125, -0.004472823338055856, 0.455078125, 1.1230257722279406, 0.32227389892883124, 0.4210229810012339, 0.4375341940655936, 0.5905175353513865, 0.8026893396056993, -4.781770707134625e-17, 0.4724599485505072, 1.125, -0.001974227772059348, 0.458984375, 1.1255629149740296, 0.32724817423647723, 0.4279505238832705, 0.44925376877466366, 0.6097104969529776, 0.8353634370452402, -1.9581514708717095e-16, 0.4768517106800534, 1.125, 0.0005629149740295906, 0.462890625, 1.1281392391803933, 0.3323047162012518, 0.43506415308644136, 0.4613557951586572, 0.6296874831736138, 0.8696069037781549, -1.5375536784711783e-16, 0.4812534598405289, 1.125, 0.003139239180393383, 0.466796875, 1.1307553961657515, 0.33744574100957136, 0.4423699659441243, 0.47385595157895843, 0.6504862949848781, 0.9055075825354876, -2.0796033699010148e-16, 0.48566535035098424, 1.125, 0.005755396165751539, 0.470703125, 1.1334120548456763, 0.3426735377592158, 0.44987431063821115, 0.4867706751964492, 0.6721468571198855, 0.9431589225707785, -6.84573556014174e-17, 0.4900875391088664, 1.140625, -0.007212945154323778, 0.474609375, 1.1361099023138785, 0.347990471469715, 0.4575837983861958, 0.5001172047626979, 0.6947113541389016, 0.982660382676636, 1.712256724790218e-16, 0.4945201856598853, 1.140625, -0.0045150976861213785, 0.478515625, 1.1388496444475822, 0.35339898623955374, 0.46550531631812625, 0.5139136261631796, 0.7182243763133496, 1.024117866455271, 2.620211997307875e-17, 0.4989634522701967, 1.140625, -0.0017753555524177807, 0.482421875, 1.1416320065381622, 0.3589016085585357, 0.47364604108795627, 0.5281789209118715, 0.742733076117033, 1.0676441926916516, 1.2481251332959503e-17, 0.5034175040009967, 1.140625, 0.0010070065381622062, 0.486328125, 1.1444577339482975, 0.36450095078419076, 0.48201345326707484, 0.5429330178126512, 0.7682873361833398, 1.113359603953431, -1.4597330727925772e-16, 0.5078825087856353, 1.140625, 0.003832733948297569, 0.490234375, 1.147327592796954, 0.37019971479168784, 0.4906153525712899, 0.5581968480204295, 0.7949399496639243, 1.1613923168463696, 1.4422225520675558e-16, 0.5123586375093465, 1.140625, 0.00670259279695401, 0.494140625, 1.1502423706735923, 0.3760006958073396, 0.49945987397633035, 0.5739924037540303, 0.82274681400861, 1.2118791176908925, -1.4201570217292412e-16, 0.5168460640917152, 1.15625, -0.006007629326407818, 0.498046875, 1.1532028773830771, 0.38190678643645354, 0.5085555047810277, 0.5903428009336379, 0.8517671392788025, 1.264966007758192, 1.4973136186654247e-16, 0.5213449655719873, 1.15625, -0.0030471226169228763, 0.501953125, 1.1562099457228516, 0.3879209808970033, 0.5179111027084483, 0.6072723460569853, 0.8820496889899455, 1.320805641894323, 2.094638171225848, 1.6307785707886589e-16, 0.52585552219736, 1.15625, -4.005427714847179e-05, 0.505859375, 1.1592644322940282, 0.394046379471367, 0.5275359149547584, 0.624806607523778, 0.9136882193469252, 1.3795709312117737, 2.2047046751184247, -4.614180764018541e-17, 0.5303779175143664, 1.15625, 0.0030144322940282293, 0.509765625, 1.162367218348156, 0.40028619318921155, 0.5374395986525855, 0.6429724920300016, 0.9467399989580887, 1.441436884043909, 2.3215048054814837, -1.3342986474285593e-16, 0.5349123384635011, 1.15625, 0.006117218348155879, 0.513671875, 1.165519210671518, 0.40664374875549664, 0.5476322423084127, 0.6617983260470026, 0.9812799039185229, 1.5065939732142886, 2.4455089337581084, -1.3396158752109626e-16, 0.5394589754772263, 1.171875, -0.0063557893284821206, 0.517578125, 1.1687213425089371, 0.4131224937385371, 0.5581243886859659, 0.6813139430636891, 1.0173874289138203, 1.5752460324070015, 2.5772244550938677, 3.905279758625744e-17, 0.5440180225815077, 1.171875, -0.003153657491062917, 0.521484375, 1.1719745745291799, 0.4197260020341018, 0.568927059074338, 0.7015507769301403, 1.0551470125492437, 1.6476113122494043, 2.7171990355198528, -1.265890525297395e-16, 0.5485896775010399, 1.171875, 9.957452917975121e-05, 0.525390625, 1.1752798958341812, 0.42645797962264836, 0.5800517790655333, 0.7225419618039368, 1.0946483885022933, 1.7239236294126705, 2.866024173600037, 1.2816556514529315e-16, 0.5531741417683267, 1.171875, 0.0034048958341813358, 0.529296875, 1.178638325014455, 0.4333222706380017, 0.5915106059486545, 0.7443224392255328, 1.1359869647828316, 1.8044336177780702, 3.02433911001963, 7.372487928303894e-17, 0.557771620836796, 1.171875, 0.006763325014454897, 0.533203125, 1.1820509112531912, 0.44032286376709234, 0.6033161578365371, 0.7669290728957534, 1.1792642336080674, 1.8894100916816012, 3.1928351222693516, -6.218035606594587e-17, 0.5623823241981278, 1.1875, -0.005449088746808937, 0.537109375, 1.1855187354817172, 0.44746389900177946, 0.6154816446499847, 0.7904007717798355, 1.224588214646731, 1.9791415323262738, 3.372260246087258, 7.835203862989061e-17, 0.5670064655039972, 1.1875, -0.0019812645182829013, 0.541015625, 1.1890429115891528, 0.45474967476531036, 0.6280209010949488, 0.8147786222188568, 1.2720739346604808, 2.0739377096541918, 3.5634244700973357, 4.098261682790949e-17, 0.5716442626924358, 1.1875, 0.0015429115891528208, 0.544921875, 1.1926245876892836, 0.46218465543761855, 0.6409484217791152, 0.8401060297914523, 1.3218439468741157, 2.174131453315553, 3.7672054556803576, 1.0609661032317167e-16, 0.5762959381190269, 1.1875, 0.005124587689283611, 0.548828125, 1.1962649474478668, 0.46977347930544733, 0.6542793986264864, 0.8664288717370179, 1.3740288937430631, 2.2800805878769315, 3.984554840272743, 1.2189225025821141e-16, 0.5809617186931688, 1.203125, -0.006860052552133231, 0.552734375, 1.199965211473798, 0.47752096696522334, 0.6680297607618081, 0.8937956608268416, 1.428768117160852, 2.392170049097533, 4.216505189367396, -1.6279456734595336e-16, 0.5856418360196454, 1.203125, -0.003159788526202078, 0.556640625, 1.2037266387777905, 0.48543213020869896, 0.6822162170511719, 0.9222577216525556, 1.4862103205653916, 2.510814199993048, 4.464177670426561, 1.1808011548704705e-16, 0.5903365265457601, 1.203125, 0.0006016387777904275, 0.560546875, 1.2075505283024641, 0.49351218142366104, 0.6968563015009803, 0.9518693803928268, 1.5465142878659675, 2.6364593675291776, 4.728790530922221, 7.500374165972573e-17, 0.5950460317143107, 1.203125, 0.00442552830246404, 0.564453125, 1.211438220528001, 0.5017665435444687, 0.7119684217348089, 0.9826881692202825, 1.6098496646286806, 2.7695866231716115, 5.011668472935972, -3.562903264922006e-17, 0.5997705981226811, 1.21875, -0.007311779471999103, 0.568359375, 1.2153910991578087, 0.5102008605898702, 0.7275719107867189, 1.0147750466223897, 1.6763978075331922, 2.9107148332013315, 5.314253028286402, 9.873230353957484e-17, 0.6045104776883639, 1.21875, -0.003358900842191155, 0.572265625, 1.2194105928889358, 0.5188210088284598, 0.743687082470423, 1.0481946350335942, 1.7463527087554778, 3.060404007723801, 5.638114051367425, -6.671121275268839e-17, 0.6092659278212316, 1.21875, 0.000660592888935898, 0.576171875, 1.2234981772723073, 0.5276331086153111, 0.7603352906065894, 1.083015477312874, 1.8199220026483087, 3.2192589807038483, 5.984962461788684, -2.1395157376602661e-16, 0.6140372116028945, 1.21875, 0.004748177272307177, 0.580078125, 1.227655376668209, 0.536643536946775, 0.7775389924157091, 1.1193103137525278, 1.8973280628929476, 3.3879334571979296, 6.3566643859014675, 1.6288916592626768e-16, 0.6188245979735116, 1.234375, -0.006719623331790895, 0.583984375, 1.2318837663028273, 0.5458589407841917, 0.795321816411578, 1.157156381472249, 1.9788091991929955, 3.5671344682923065, 6.755256865799159, 4.184994235421112e-17, 0.6236283619264369, 1.234375, -0.0024912336971728037, 0.587890625, 1.236184974432056, 0.5552862512013697, 0.8137086351608545, 1.1966357382393429, 2.0646209635869015, 3.757627279159722, 7.182965326390026, -1.4903167261926637e-16, 0.6284487847111011, 1.234375, 0.0018099744320560628, 0.591796875, 1.240560684619246, 0.5649326984151567, 0.8327256433076302, 1.237835612963549, 2.155037577583574, 3.9602408011984322, 7.642223016511318, 5.668265340946725e-17, 0.6332861540445713, 1.234375, 0.006185684619246016, 0.595703125, 1.245012638134033, 0.5748058277633225, 0.8524004412988602, 1.280848785345899, 2.250353492592003, 4.175873565508851, 8.135692668955654, -8.055118562955642e-17, 0.6381407643322414, 1.25, -0.004987361865967047, 0.599609375, 1.2495426364799205, 0.5849135166993058, 0.8727621252872089, 1.3257739974182423, 2.350885097539427, 4.405500322101544, 8.666290657418184, 1.424048813729075e-16, 0.643012916898138, 1.25, -0.0004573635200795498, 0.603515625, 1.2541525440588488, 0.5952639928792307, 0.8938413837328303, 1.3727163999967842, 2.4569725891745895, 4.650179337339559, 9.237213966353778, 2.212466761743417e-16, 0.6479029202253699, 1.25, 0.004152544058848897, 0.607421875, 1.2588442909816002, 0.6058658534229892, 0.9156706012752946, 1.4217880373929206, 2.568982022357418, 4.911060471341405, 9.851970333384985, -1.8329361466103715e-18, 0.6528110902072637, 1.265625, -0.006780709018399824, 0.611328125, 1.2636198760335524, 0.6167280854382003, 0.9382839705018491, 1.473108374082051, 2.6873075596720546, 5.189394127574417, 10.514411974031251, 1.4404229289208707e-16, 0.6577377504097788, 1.265625, -0.0020051239664475727, 0.615234375, 1.2684813698060229, 0.6278600879035455, 0.961717612299092, 1.5268048674307138, 2.812373941998725, 5.486541178844498, 11.228773356344059, 2.1356990154430766e-16, 0.6626832323458296, 1.265625, 0.002856369806022836, 0.619140625, 1.2734309180042227, 0.639271695016416, 0.986009705542634, 1.5830135910298127, 2.9446392042785545, 5.803983987561629, 11.999713559532411, -2.146063844597763e-16, 0.667647875762178, 1.265625, 0.0078059180042227, 0.623046875, 1.2784707449437045, 0.6509732011190884, 1.0112006269542233, 1.6418799136831546, 3.0845976636468917, 6.143338653791852, 12.832363827668269, 6.352126153421112e-17, 0.6726320289396108, 1.28125, -0.002779255056295434, 0.626953125, 1.2836031572481128, 0.6629753873278378, 1.0373331020390146, 1.7035592396631631, 3.232783210444274, 6.506368642500278, 13.732381018361684, 7.277766413589188e-19, 0.67763604900717, 1.28125, 0.0023531572481126917, 0.630859375, 1.2888305477620632, 0.6752895500006308, 1.0644523681081925, 1.7682178164776767, 3.389772936397015, 6.894999961901313, 14.706007749288435, 1.4013832084560466e-16, 0.6826603022712345, 1.28125, 0.007580547762063206, 0.634765625, 1.2941553996940875, 0.6879275311914091, 1.0926063504951604, 1.8360336171025755, 3.5561911385575384, 7.3113380883777275, 15.760140164834917, 2.2835150571009758e-17, 0.687705164560338, 1.296875, -0.0027196003059123956, 0.638671875, 1.2995802910057819, 0.7009017512526244, 1.1218458531883073, 1.9071973044354253, 3.732713742484712, 7.7576868604966, 16.90240438376958, 1.4284039040671817e-16, 0.6927710215866405, 1.296875, 0.0027052910057819655, 0.642578125, 1.3051078990646252, 0.7142252437627536, 1.1522247652314443, 1.9819132866277622, 3.9200731937155275, 8.23656959580931, 18.141242850191208, 2.0751381688181238e-16, 0.6978582693250579, 1.3125, -0.007392100935374855, 0.646484375, 1.3107410055793738, 0.7279116929721785, 1.1838002843860813, 2.060400872972375, 4.119063872935178, 8.750752720046929, 19.48601199786747, 5.716805107860533e-17, 0.7029673144111097, 1.3125, -0.001758994420626132, 0.650390625, 1.3164825018385242, 0.7419754739792445, 1.2166331597097098, 2.1428955411734383, 4.330548097513449, 9.303272239803016, 20.947092857538706, -9.84599295509245e-17, 0.708098574558631, 1.3125, 0.0039825018385242225, 0.654296875, 1.3223353942740623, 0.756431695868727, 1.2507879548834235, 2.229650328130627, 4.555462780382346, 9.897463437766762, 22.536016493047065, -3.4960776597538866e-17, 0.7132524789985757, 1.328125, -0.005789605725937761, 0.658203125, 1.3283028103746275, 0.7712962480675751, 1.2863333343230472, 2.3209373578453905, 4.7948268267475465, 10.536994225133995, 24.265606452449873, -5.4358226400241914e-17, 0.7184294689402275, 1.328125, 0.00017781037462742932, 0.662109375, 1.3343880049743009, 0.78658585019793, 1.3233423743333959, 2.4170495217338974, 5.049749360047487, 11.225902650288152, 26.15014077251256, 1.281976617730236e-16, 0.7236299980562326, 1.328125, 0.006263004974300771, 0.666015625, 1.3405943669455234, 0.8023181057353581, 1.3618929018186676, 2.5183023285362816, 5.32143888112481, 11.968639137757101, 28.20553648872914, -6.325981341169e-18, 0.7288545329929796, 1.34375, -0.0031556330544766776, 0.669921875, 1.3469254263271766, 0.8185115598113214, 1.4020678633471362, 2.6250359431801487, 5.611213479021256, 12.770114118656176, 30.44956009037489, -2.1064486273754088e-16, 0.7341035539079606, 1.34375, 0.0031754263271766585, 0.673828125, 1.3533848619216378, 0.8351857615335435, 1.4439557276896642, 2.737617436427774, 5.9205122284619405, 13.63575181551479, 32.902067934635674, -1.3844199794440324e-16, 0.7393775550358823, 1.359375, -0.0059901380783621925, 0.677734375, 1.3599765093976888, 0.8523613312365729, 1.4876509253141907, 2.8564432699577327, 6.2509079283271864, 14.571551063169178, 35.58528131295694, -3.633469970538101e-17, 0.7446770452854299, 1.359375, 0.0006015093976887192, 0.681640625, 1.366704369939547, 0.8700600331179971, 1.5332543287281535, 2.9819420447572655, 6.60412135765606, 15.584154186413521, 38.52410166673959, -1.1483409140657897e-16, 0.750002548868733, 1.359375, 0.007329369939546881, 0.685546875, 1.373572619486027, 0.8883048537640373, 1.580873778024572, 3.114577544395339, 6.9820372515017715, 16.680925118081547, 41.746472403285935, -1.9449657264480127e-16, 0.7553546059657559, 1.375, -0.001427380513973067, 0.689453125, 1.3805856186079857, 0.9071200871223083, 1.6306246565131362, 3.2548521089827824, 7.386722228877908, 17.870038132676797, 45.283794896771944, 1.1965056650725188e-16, 0.7607337734260087, 1.375, 0.005585618607985694, 0.693359375, 1.3877479230767928, 0.9265314265401537, 1.6826305219142588, 3.403310380492973, 7.820444939824489, 19.160578795992375, 49.171407609684714, 2.126535977492267e-16, 0.7661406255101748, 1.390625, -0.002877076923207057, 0.697265625, 1.3950642951816625, 0.9465660645550522, 1.7370237992723014, 3.5605434657176085, 8.285698739151776, 20.562658996882735, 53.44913888266603, 7.120888746483928e-17, 0.7715757546744619, 1.390625, 0.004439295181662456, 0.701171875, 1.4025397158593342, 0.9672528012001571, 1.7939465425164665, 3.727193569589074, 8.785227241723737, 22.08754824135552, 58.16194586990583, -1.920174705271017e-16, 0.7770397724007281, 1.40625, -0.0037102841406658166, 0.705078125, 1.41017939770589, 0.9886221616742743, 1.853551272478629, 3.903959159057863, 9.322053169461475, 23.747823760991132, 63.36065441053709, 1.2125668675389011e-17, 0.7825333100756877, 1.40625, 0.003929397705890114, 0.708984375, 1.4179887989474937, 1.0107065243228615, 1.9160019001835777, 4.0916007263410235, 9.899510965066515, 25.557542428969718, 69.10281740691319, 9.999244919218779e-17, 0.7880570199227939, 1.421875, -0.0038862010525061667, 0.712890625, 1.4259736384546595, 1.0335402599865324, 1.9814747453785868, 4.2909472303581735, 10.521283723571148, 27.532438001808956, 75.45371263027732, -7.891794573967661e-17, 0.7936115759906994, 1.421875, 0.004098638454659504, 0.716796875, 1.4341399118933855, 1.0571598838979073, 2.050159661589202, 4.5029033067860995, 11.191445082370837, 29.69014783052269, 82.48750491955094, 5.1212017794616454e-17, 0.7991976752025578, 1.4375, -0.0033600881066145243, 0.720703125, 1.442493909116252, 1.081604221448593, 2.1222612805040204, 4.728457350678363, 11.91450681598199, 32.05047393238505, 90.2886026364174, -1.8809906770063086e-16, 0.8048160384708027, 1.4375, 0.004993909116252043, 0.724609375, 1.451042232907521, 1.1069145893080907, 2.1980003902353524, 4.968690591355856, 12.695473006545008, 34.6356842096959, 98.95324418434926, 7.679422967482682e-17, 0.8104674118824629, 1.453125, -0.002082767092478928, 0.728515625, 1.4597918192085473, 1.1331349935583979, 2.277615464013169, 5.224787297688557, 13.539901808878694, 37.47086067671657, 108.5913576333838, 1.7017039489625748e-16, 0.816152567960553, 1.453125, 0.00666681920854722, 0.732421875, 1.4687499589636008, 1.1603123467153802, 2.361364358191857, 5.498046273451396, 14.453976004338498, 40.58430284954288, 119.32874532008853, 1.4305208802428753e-16, 0.8218723070075828, 1.46875, -4.103639922419387e-08, 0.736328125, 1.4779243217417237, 1.1884967057445979, 2.4495262011367505, 5.789893827741734, 15.444583746513121, 44.00799601468648, 131.30965609406573, -1.6381588685224814e-16, 0.8276274585378189, 1.484375, -0.0064506782582763565, 0.740234375, 1.4873229813077393, 1.21774153344981, 2.542403497674012, 6.101898435209305, 16.51941115085736, 47.77815598083159, 144.69982113801285, 2.827854503351633e-17, 0.8334188828055655, 1.484375, 0.002947981307739343, 0.744140625, 1.4969544433353008, 1.2481039859223368, 2.6403244774095063, 6.435787335938509, 17.687048678227455, 51.935864209667955, 159.69004560273515, -1.6474215576380717e-16, 0.839247472437453, 1.5, -0.003045556664699121, 0.748046875, 1.5068276754772194, 1.2796452290952434, 2.743645719440916, 6.793465366289981, 18.957113619450954, 56.527810009578616, 176.5004684385639, 3.5458731105107694e-17, 0.8451141541775047, 1.5, 0.006827675477219385, 0.751953125, 1.5169521400336585, 1.3124307878555586, 2.8527550909118875, 7.177036372784717, 20.340391431188078, 61.60227296005288, 195.38357421325207, 640.5380495953193, -2.6920613564648094e-17, 0.8510198907546633, 1.515625, 0.0013271400336584823, 0.755859375, 1.5273378294875564, 1.346530931639521, 2.968075042619083, 7.5888275393741855, 21.848999097071037, 67.22907306802716, 216.638149619849, 721.5737219951404, 1.4810965321010352e-16, 0.8569656828834296, 1.53125, -0.003912170512443657, 0.759765625, 1.537995305209378, 1.382021100980579, 3.090066311640267, 8.031417263965912, 23.49657459036499, 73.47321537876583, 240.6050238163214, 814.4239518597311, 4.502983638265062e-17, 0.8629525714094011, 1.53125, 0.006745305209378038, 0.763671875, 1.5489357396706251, 1.4189823801110248, 3.21923208889772, 8.507666916306801, 25.29849682174486, 80.41416612694888, 267.6831365701302, 921.0469519498516, -2.1113615524257245e-16, 0.8689816396127346, 1.546875, 0.002060739670625245, 0.767578125, 1.5601709625481677, 1.4575020214498915, 3.356122718935664, 9.02075728898355, 27.272141822148185, 88.14318383136782, 298.3381475972888, 1043.765945553088, -1.0132075666657874e-17, 0.8750540156839732, 1.5625, -0.0023290374518322512, 0.771484375, 1.5717135111502711, 1.4976740286613652, 3.501341010258759, 9.574230448050061, 29.437181775888398, 96.76530936460901, 333.114987822559, 1185.3450133937843, 1.1965478268839207e-16, 0.8811708753882628, 1.578125, -0.006411488849728784, 0.775390625, 1.5835766856511941, 1.5395998059607037, 3.6555482476940853, 10.172037907100856, 31.81593498333109, 106.4017338568264, 372.65304326603115, 1349.08238290385, 3.039629257236238e-17, 0.8873334449357642, 1.578125, 0.0054516856511940825, 0.779296875, 1.5957746096856174, 1.5833888825051503, 3.8194710138244674, 10.818596211775391, 34.43377651351211, 117.19262443295392, 417.70458279572443, 1538.9255721055058, -3.918652333380919e-17, 0.8935430040781056, 1.59375, 0.002024609685617326, 0.783203125, 1.608322296928385, 1.6291597220674372, 3.993908945115038, 11.518851231971196, 37.319621393500206, 129.30050653600003, 469.1571971264812, 1759.6140218563623, -4.0902836068719946e-17, 0.8998008894530054, 1.609375, -0.0010527030716149589, 0.787109375, 1.6212357243707765, 1.6770406297877805, 4.17974357055685, 12.278352714725925, 40.50649476162607, 142.91432518006442, 528.0612162443878, 2016.8564400461469, -2.990039269767411e-17, 0.9061084982018035, 1.625, -0.0037642756292234673, 0.791015625, 1.6345319131038407, 1.7271707696838148, 4.377948407271867, 13.103340962651076, 44.03220661459872, 158.25433727169988, 595.6633291963578, 2317.5521602282884, 3.7062111638090963e-17, 0.9124672918876091, 1.640625, -0.006093086896159222, 0.794921875, 1.6482290175346461, 1.779701308823979, 4.589600519544401, 14.00084788467002, 47.94015276941318, 175.57802492409837, 673.4479616104387, 2670.0685483394054, 4.835325864709877e-17, 0.9188788007451572, 1.640625, 0.007604017534646196, 0.798828125, 1.6623464240955736, 1.8347967067086528, 4.8158937863883935, 14.978815135016402, 52.280268653359634, 195.1872678134027, 763.18839608378, 3084.5900970887315, -6.024062915511139e-17, 0.9253446282973372, 1.65625, 0.0060964240955735894, 0.802734375, 1.6769048606635577, 1.8926361715411097, 5.058154169558372, 16.04623263513357, 57.11016880633552, 217.43707419100983, 867.0101805321129, 3573.559632534042, 1.7007350967944576e-16, 0.931866456377799, 1.671875, 0.0050298606635576, 0.806640625, 1.6919265180897611, 1.953415308813478, 5.317857330787433, 17.21330148961961, 62.49651288996836, 242.74624928571455, 987.4701027821019, 4152.238440626376, 5.78913003435096e-17, 0.938446050604143, 1.6875, 0.004426518089761076, 0.810546875, 1.707435185455773, 2.0173479921131388, 5.5966490164140605, 18.491626198112947, 68.51664902047291, 271.61048199668966, 1127.654974628722, 4839.42068364997, 1.813870625918979e-16, 0.9450852663520775, 1.703125, 0.004310185455773095, 0.814453125, 1.7234564009264302, 2.0846684914361595, 5.896368712515482, 19.894442177058068, 75.26059799894341, 304.6184633608869, 1291.3057453067022, 5658.349026760811, -2.2054292854231977e-16, 0.951786055287732, 1.71875, 0.004706400926430053, 0.818359375, 1.740017620369578, 2.155633900781013, 6.219077178112465, 21.43688599969919, 82.83345832743159, 342.471823219118, 1482.9741629552366, 6637.894069027615, -2.2223661205472935e-18, 0.9585504725231746, 1.734375, 0.005642620369578049, 0.822265625, 1.757148406269124, 2.2305269146444897, 6.567088592955478, 23.136317519099357, 91.35833286761749, 386.00989830017835, 1708.2214756608746, 7814.081574584298, -5.1354421081469276e-17, 0.9653806844693555, 1.75, 0.007148406269124139, 0.826171875, 1.7748806398814183, 2.309659012575606, 6.943008216281311, 25.012705261992014, 100.97990508805991, 436.24064402565915, 1973.871723558293, 9232.080909185219, 2.3501748525507007e-17, 0.972278977471345, 1.78125, -0.006369360118581738, 0.830078125, 1.7932487600910672, 2.3933741225715246, 7.349776652059394, 27.089089308712108, 111.86882803124396, 494.37939904955516, 2288.336318489623, 10948.808772923941, -2.0386073200216244e-16, 0.9792477673232423, 1.796875, -0.003626239908932837, 0.833984375, 1.8122900330290197, 2.482052848342645, 7.79072206543371, 29.39213948925467, 124.22713508959886, 561.8977409545255, 2662.032259414175, 13036.35899852746, 1.2529247137346887e-16, 0.9862896097747611, 1.8125, -0.00020996697098031787, 0.837890625, 1.8320448572461001, 2.5761173629975254, 8.269622008426866, 31.95283137186761, 138.29494206117138, 640.5853825025027, 3107.9240904164144, 15586.548720673985, -7.440085031521401e-17, 0.9934072121587656, 1.828125, 0.003919857246100201, 0.841796875, 1.852557110117934, 2.676037093358108, 8.79077690906147, 34.807268526561614, 154.35878977436667, 732.6290189898459, 3642.2304353113236, 18716.98368566763, 1.6770425586098002e-16, 1.000603446289404, 1.859375, -0.0068178898820660675, 0.845703125, 1.8738745422288534, 2.782335346018136, 9.35909778140442, 37.997687355173625, 172.7620827616508, 840.713345641218, 4285.350885058854, 22579.205766774627, -1.5291306990629306e-16, 1.00788136280471, 1.875, -0.0011254577711465833, 0.849609375, 1.8960492277892311, 2.8955970598418626, 9.980211357303276, 41.57369099632458, 193.918221687648, 968.1512579893148, 5063.08999059775, 27369.716135337472, 1.4248298925186658e-16, 1.0152442071563357, 1.890625, 0.005424227789231182, 0.853515625, 1.9191380807419738, 3.0164779117494356, 10.660586667507147, 45.593772266540114, 218.3272190692685, 1119.0527268404053, 6008.284802777755, 33345.00156744842, 1.477613625860445e-16, 1.0226954374835944, 1.921875, -0.0027369192580262262, 0.857421875, 1.9432034481857925, 3.1457150558303746, 11.40768816947576, 50.12720343352154, 246.59684848204867, 1298.545291414189, 7162.9847837354655, 40842.180134170005, -8.775757125763087e-17, 1.030238744650397, 1.9375, 0.005703448185792383, 0.861328125, 1.96831379518387, 3.2841398433571998, 12.230161911718683, 55.25639443025811, 279.4697343758761, 1513.0639622487633, 8581.393965513902, 50307.60464924943, -1.5175596750870063e-16, 1.0378780747737042, 1.96875, -0.00043620481612989564, 0.865234375, 1.994544498065, 3.432692957531763, 13.138063049897221, 61.07985315223943, 317.85828238474835, 1770.7351943477727, 10333.874013704524, 62336.8400368725, 9.981376855309294e-17, 1.0456176546327742, 2.0, -0.005455501935000007, 0.869140625, 2.021978767128443, 3.592442507700872, 14.143135439995621, 67.71592491407935, 362.8900361719127, 2081.8894140775983, 12512.437259520811, 77731.05671681973, -8.371317329681747e-17, 1.053462020422467, 2.015625, 0.006353767128442975, 0.873046875, 2.050708724455903, 3.7646057713473633, 15.259157240158983, 75.30754752628295, 415.96701100746515, 2459.750769425709, 15238.35233938401, 97577.36282970781, -1.500962318489224e-16, 1.0614160504045338, 2.046875, 0.0038337244559031257, 0.876953125, 2.080836668609939, 3.950575459371039, 16.502370752708323, 84.02834036283352, 478.8439221040989, 2921.373473740808, 18672.775649886982, 123364.42848943049, -1.659746509402531e-16, 1.0694850021226352, 2.078125, 0.0027116686099391374, 0.880859375, 2.112476565754693, 4.151951626164374, 17.892020553565217, 94.09045982722259, 553.7321860242952, 3488.9246526168095, 23031.76316759504, 157150.74476711656, -9.801531485573023e-17, 1.0776745549854727, 2.109375, 0.0031015657546928296, 0.884765625, 2.14575581670905, 4.3705806718558895, 19.451031893310383, 105.75481396473927, 643.4394131993259, 4191.459185730928, 28607.695977775253, 201812.3522973303, 1.3813391927331196e-16, 1.0859908591955247, 2.140625, 0.0051308167090499185, 0.888671875, 2.1808173623675535, 4.608603319519088, 21.20687229010058, 119.3444567894281, 751.5582705465334, 5067.400915287886, 35800.21083931867, 261412.13344012393, -1.3368639755211717e-16, 1.094440592218409, 2.1875, -0.0066826376324464, 0.892578125, 2.2178222068115296, 4.868514037326099, 23.19265445683687, 135.26231022641875, 882.7247645807172, 6168.050055545901, 45161.39962115418, 341757.6662337286, -1.7075686151214768e-16, 1.1030310242632715, 2.21875, -0.0009277931884705266, 0.896484375, 2.256952459690034, 5.153235175240809, 25.448560124102528, 154.0148375087852, 1042.9752680559998, 7562.600393468363, 57462.72565625023, 451255.94762644655, -2.9136432271746473e-17, 1.1117700945959057, 2.25, 0.006952459690034248, 0.900390625, 2.2984150290601137, 5.4662101861950765, 28.023694810286074, 176.24399273054206, 1240.2457397112366, 9345.407978427267, 73795.4830690938, 602243.009452391, 1.9290551592478137e-16, 1.1206665009581287, 2.296875, 0.0015400290601135741, 0.904296875, 2.342446135662989, 5.8115218329222955, 30.978527533896468, 202.77081748673186, 1485.0784295395724, 11646.666407352448, 95723.89038068237, 813086.3089478464, -8.495988132127074e-17, 1.12972980495323, 2.34375, -0.0013038643370111287, 0.908203125, 2.3893168736515644, 6.194043438216906, 34.3881336455605, 234.65564031135807, 1791.6356782737046, 14648.317481329199, 125522.1924656691, 1111567.9864435706, 1.9473112081481653e-16, 1.1389705570252548, 2.390625, -0.0013081263484354107, 0.912109375, 2.4393401170316493, 6.619634313275849, 38.346554052845576, 273.28227325052785, 2179.17526511129, 18608.144368047728, 166548.33042878797, 1540433.7551567412, -1.3146358629384365e-16, 1.1484004456760664, 2.4375, 0.0018401170316494654, 0.916015625, 2.4928791743695395, 7.0953949503711, 42.972727205603384, 320.4774167250676, 2674.231052066974, 23896.88861754643, 223844.079264511, 2166682.9797568014, -2.00077276789543e-16, 1.158032476923553, 2.5, -0.007120825630460522, 0.919921875, 2.550358739962583, 7.630004104650472, 48.418669167719635, 378.68256870469486, 3313.8910018987153, 31056.51121800243, 305118.9290199373, 3097474.639195412, 1.9701290481269865e-16, 1.167881191844724, 2.546875, 0.003483739962583086, 0.923828125, 2.6122788980619824, 8.23416965270152, 54.880918297233436, 451.2056308994326, 4150.817685291495, 40893.53330084261, 421976.2409778442, 4507509.705189242, 49336421.84620552, -1.7579224555354087e-16, 1.1779629325710719, 2.609375, 0.0029038980619822993, 0.927734375, 2.6792332397084877, 8.921239955873693, 62.61680340827558, 542.5958564003607, 5261.0864454185985, 54631.936216344635, 594202.6604435124, 6691086.698877255, 77196919.59091844, -2.1484104084702894e-16, 1.1882961706131403, 2.671875, 0.007358239708487542, 0.931640625, 2.751932599976151, 9.70804546145563, 71.9679752908621, 659.2137933147085, 6756.719603969639, 74169.9466811716, 852788.5749310142, 10153073.930667344, 123835853.66947556, 1.4438642732398018e-16, 1.198901916347328, 2.75, 0.0019325999761509252, 0.935546875, 2.8312365994198765, 10.616076713673397, 83.39510737224795, 810.1168412031501, 8806.196261153233, 102523.0948282325, 1250204.092827802, 15789341.89147239, 204259308.98522362, 1.020880246298233e-18, 1.209804235615782, 2.828125, 0.003111599419876708, 0.939453125, 2.918196214882752, 11.673164055348714, 97.53017562716705, 1008.4691521605262, 11668.970822838626, 144612.30542590306, 1877190.5828223242, 25242494.59608985, 347633029.93181646, -8.975816601834434e-17, 1.2210309098049623, 2.921875, -0.0036787851172479483, 0.943359375, 3.0141122472557913, 12.915922756842093, 115.25713420781683, 1273.848410658663, 15755.361226811629, 208714.0939414443, 2896077.692625328, 41639601.52295806, 613031250.1580848, 2.5281374170967758e-17, 1.232614291325484, 3.015625, -0.0015127527442088048, 0.947265625, 3.120617213375309, 14.393396233614483, 137.8398157136046, 1636.1377827101771, 21734.101364264483, 309233.2986955385, 4608615.541089899, 71192830.31654616, 1125839671.2076259, -1.0457598081096884e-16, 1.2445924301972102, 3.125, -0.0043827866246908265, 0.951171875, 3.239792625501863, 16.17262967793989, 167.13100596937713, 2142.325442488586, 30733.21200441135, 472240.3138770321, 7600856.339567495, 126859355.74983938, 2166831318.353224, -9.771080110544696e-17, 1.2570105847539743, 3.234375, 0.005417625501863248, 0.955078125, 3.3743412747309587, 18.34745854736222, 205.9264093053968, 2868.868281600754, 44732.39845599601, 747098.0590455162, 13070117.977268228, 236434757.39183396, 4403356874.619954, 83395415315.89012, 9.40742822033643e-17, 1.2699232897676995, 3.375, -0.0006587252690412716, 0.958984375, 3.5278478549371806, 21.052858755836485, 258.58872397208495, 3945.2379760418976, 67370.56185515701, 1232313.2891957294, 23611429.21384909, 467795759.06604844, 9549174139.467613, 198107596236.35822, -1.1463714732008328e-16, 1.2833972560361602, 3.53125, -0.003402145062819647, 0.962890625, 3.7051869829585553, 24.48935526290828, 332.20147616873504, 5601.263277066632, 105712.27838751745, 2137120.682996709, 45257023.913846076, 991012200.5733234, 22381733352.63374, 513318695900.8859, -5.311678424679859e-17, 1.2975155505182108, 3.703125, 0.0020619829585551614, 0.966796875, 3.9131884680160947, 28.966601008645384, 438.8261681500711, 8268.664658364392, 174403.91404650448, 3940500.9819381656, 93261433.44537847, 2282403753.329084, 57692329807.23973, 1479259122817.2112, 6.709267978267992e-17, 1.3123838243976498, 3.90625, 0.006938468016094754,
}

var inroot = [128]float64{
	1.408721450121, 1.39792649065766, 1.38737595123859, 1.37706074531819, 1.36697225234682, 1.35710228748795, 1.34744307370643, 1.33798721601135, 1.32872767765984, 1.31965775814772, 1.31077107283046, 1.30206153403386, 1.29352333352711, 1.285150926244, 1.2769390151482, 1.26888253714903, 1.26097664998256, 1.25321671998073, 1.24559831065844, 1.23811717205462, 1.23076923076923, 1.223550580643, 1.21645747403153, 1.20948631362953, 1.20263364480453, 1.1958961484031, 1.18927063399547, 1.18275403352732, 1.17634339535009, 1.17003587860341, 1.16382874792529, 1.15771936846787, 1.15170520119791, 1.14578379846309, 1.13995279980655, 1.13420992801334, 1.12855298537376, 1.12297985014975, 1.11748847323133, 1.11207687497107, 1.10674314218572, 1.10148542531442, 1.09630193572405, 1.09119094315276, 1.08615077328341, 1.08117980543918, 1.0762764703941, 1.07143924829188, 1.06666666666667, 1.06195729855996, 1.05730976072814, 1.05272271193563, 1.04819485132867, 1.04372491688551, 1.03931168393861, 1.03495396376504, 1.03065060224133, 1.02640047855933, 1.0222025039999, 1.01805562076124, 1.01395880083916, 1.00991104495649, 1.00591138153909, 1.00195886573624, 0.9961164901835, 0.98848330114434, 0.98102294317595, 0.9737289911203, 0.96659534932828, 0.95961623024651, 0.95278613468066, 0.94609983358253, 0.93955235122353, 0.93313894963169, 0.92685511418159, 0.9206965402375, 0.91465912076005, 0.9087389347953, 0.90293223677296, 0.89723544654727, 0.89164514012056, 0.88615804099474, 0.88077101210109, 0.87548104826333, 0.87028526915267, 0.8651809126974, 0.86016532891275, 0.85523597411976, 0.85039040552437, 0.8456262761307, 0.84094132996422, 0.83633339758291, 0.83180039185606, 0.82734030399203, 0.82295119979782, 0.81863121615464, 0.81437855769486, 0.81019149366693, 0.80606835497581, 0.80200753138734, 0.79800746888611, 0.79406666717674, 0.79018367731967, 0.78635709949278, 0.78258558087123, 0.77886781361798, 0.77520253297841, 0.77158851547266, 0.76802457717971, 0.76450957210799, 0.76104239064719, 0.75762195809661, 0.75424723326565, 0.75091720714229, 0.7476309016256, 0.74438736831878, 0.74118568737933, 0.73802496642311, 0.7349043394794, 0.73182296599416, 0.72878002987884, 0.72577473860242, 0.7228063223242, 0.71987403306536, 0.71697714391715, 0.71411494828392, 0.71128675915902, 0.70849190843208,
}

// powtwo: glibc root.tbl companion (2^0..2^27).
var powtwo = [28]float64{
	1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
	256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0,
	65536.0, 131072.0, 262144.0, 524288.0, 1048576.0, 2097152.0,
	4194304.0, 8388608.0, 16777216.0, 33554432.0, 67108864.0, 134217728.0,
}

// scostab: glibc 2.39 sysdeps/ieee754/dbl-64/sincostab.c __sincostab (440 doubles).
var scostab = [440]float64{
	0.0, 0.0, 1.0, 0.0,
	0.0078124205273828315, -4.102755709382617e-19, 0.9999694825770951, 5.427791428371788e-17,
	0.015624364224883372, -1.265093755275982e-19, 0.9998779321710066, 3.216122229972341e-17,
	0.02343535429172257, 1.0500785515068913e-19, 0.9997253543694995, 4.666714544113391e-17,
	0.03124491398532608, -1.562781562225433e-18, 0.9995117584851364, -3.418806487972947e-17,
	0.03905256665042215, 1.1342467037138512e-18, 0.999237157554709, 1.1089280858439994e-17,
	0.04685783574813424, -2.3419368365610254e-18, 0.9989015683384429, -2.1425557800399754e-17,
	0.05466024488506646, 3.373831935351774e-19, 0.9985050113189741, 1.779930870620664e-17,
	0.0624593178423802, -2.040259504585711e-18, 0.9980475107000991, 3.3232291674141346e-17,
	0.07025457860486005, -6.773890900301373e-18, 0.9975290944052978, 5.112086057887235e-18,
	0.07804555138996731, -5.4494437820057936e-18, 0.9969497940760287, -1.2467075728553626e-17,
	0.08583176067687935, -3.490037167231825e-18, 0.996309645069798, -1.1743657041590172e-18,
	0.09361273123551289, 1.4628632005878733e-18, 0.9956086864580017, 3.312922430932991e-17,
	0.10138798815552964, 3.99242856835807e-18, 0.9948469610235406, -8.198278333258706e-18,
	0.10915705687532236, 6.6284699502736666e-18, 0.9940245152582091, 1.3287985046260087e-17,
	0.11691946321097965, -2.283412536970096e-18, 0.9931413993598577, 2.8553964189785565e-17,
	0.12467473338522769, -2.925947496057858e-18, 0.992197667229329, 4.754870575189364e-17,
	0.13242239405634718, 1.2801053067990684e-17, 0.9911933764671682, 3.1426138811726996e-17,
	0.1401619723470637, -9.946847113883478e-18, 0.9901285883701071, -4.589906353553811e-18,
	0.14789299587340962, -1.3140968557128452e-17, 0.9890033679273229, 1.7676104005868335e-18,
	0.15561499277355603, 8.886053372342288e-18, 0.9878177838164719, 4.91917302237681e-17,
	0.16332749173661285, 5.483569430347159e-18, 0.9865719083994976, -1.0327444588275446e-17,
	0.17103002203139503, -9.954774726452923e-18, 0.9852658177182139, -4.925721262944555e-17,
	0.17872211353515366, -1.6240405901074e-20, 0.983899591489664, -2.1647988531644275e-17,
	0.18640329676226988, 2.3493796901281573e-18, 0.9824733131012553, -3.919920375420088e-17,
	0.1940731028929098, -7.832741210198615e-18, 0.9809870696056692, 1.967841180870303e-17,
	0.2017310638016388, 5.587232815460113e-18, 0.9794409517155483, 1.310876952152676e-17,
	0.20937671208599365, -5.536403693172164e-18, 0.9778350537979598, 2.9910028492769484e-17,
	0.21700958109501015, 1.1170071073364376e-17, 0.9761694738686353, -7.850690609285027e-18,
	0.2246292049577053, -1.058590416432903e-17, 0.9744443135859889, 4.990128834921396e-17,
	0.23223511861151147, -8.318080852687206e-18, 0.9726596782449127, 2.3920264546490165e-17,
	0.23982685783066157, -7.260810660979712e-18, 0.9708156767703494, 5.1285792532153653e-17,
	0.24740395925452294, -7.53102495590706e-18, 0.9689124217106447, 5.071436662403936e-17,
	0.2549659604158785, -2.2310035435425957e-17, 0.9669500292306779, -3.237770297707692e-17,
	0.2625123997691533, -2.2534597527902128e-17, 0.964928619104771, -3.0345542681018625e-18,
	0.27004281671858504, -1.2103265097887777e-17, 0.9628483147093797, -4.646009771724241e-18,
	0.2775567516463363, 1.7674070262791825e-17, 0.9607092430155619, -2.807827063516729e-17,
	0.28505374594054744, -1.8162083107618118e-17, 0.9585115345812286, 8.134621492946255e-18,
	0.29253334202332754, 7.516944930327352e-18, 0.9562553235431753, -3.148450868841629e-17,
	0.299995083378683, 2.606392777930734e-17, 0.9539407476088947, 4.3757589471734985e-17,
	0.30743851458038085, 1.1004366442765292e-19, 0.9515679480481722, -3.8614834675674123e-17,
	0.3148631813197453, -2.6921345305772252e-17, 0.949137069684463, 4.149148046099445e-17,
	0.3222686304333866, 2.093773358126606e-17, 0.9466482608860534, -3.911683334934152e-17,
	0.32965440993086015, 2.35998378957031e-17, 0.9441016735570044, -1.6017653284545848e-17,
	0.33702006902225307, 1.0312279860787216e-17, 0.9414974631278811, -4.85238302367971e-18,
	0.3443651581456984, 5.881664587517989e-18, 0.9388357885462655, 6.919329459921788e-18,
	0.3516892289948141, -2.5616208736069942e-17, 0.9361168122670553, -5.2350302039683216e-17,
	0.35899183454606504, 1.7495482840158848e-17, 0.9333407002425484, -1.322859547778088e-17,
	0.36627252908604757, -9.938814562106524e-18, 0.9305076219123143, 4.488760003328074e-18,
	0.37353086823869297, -2.375669144106189e-17, 0.9276177501928519, 4.5350942573591974e-17,
	0.38076640899239017, 2.1372528646211374e-17, 0.924671261467036, 5.5444125388034563e-17,
	0.38797870972702503, 1.759799510335953e-17, 0.9216683355733519, -8.550693097867243e-18,
	0.39516733024093426, -1.9613487871414228e-17, 0.9186091557949183, -4.0564150104514996e-17,
	0.4023318317777731, 1.4413875452702007e-17, 0.9154939088483012, 5.413375566838042e-17,
	0.40947177705329507, -5.679403000091266e-18, 0.9123227848721178, 2.6349040211413332e-17,
	0.41658673028204113, -9.610850682533715e-18, 0.909095977415431, 2.920006113841211e-17,
	0.42367625720393803, -2.331800700068871e-17, 0.9058136834259364, 4.2864666490805214e-17,
	0.4307399251108032, -2.6212879607476533e-17, 0.9024761032379415, 3.112490674651326e-17,
	0.4377773028727551, 7.64345629962023e-18, 0.8990834405601384, 9.076951775075616e-18,
	0.4447879609645272, -6.655392977344925e-18, 0.8956359024631707, -8.854043885762716e-18,
	0.4517714714916838, -8.234073942098903e-18, 0.8921336993669944, 2.3160655211380166e-17,
	0.4587274082167366, 1.6080982096218356e-17, 0.8885770450280356, -4.034491998357167e-17,
	0.46565534658516017, 1.459870391051426e-17, 0.8849661565261433, -7.690557775987357e-18,
	0.47255486375130445, -3.608790703790546e-18, 0.8813012542513406, -4.973073189306066e-17,
	0.479425538604203, -5.103969860556013e-18, 0.8775825618903728, -4.2623149864279997e-17,
	0.48626695179327556, 1.5131647652857083e-17, 0.8738103064130545, 4.767670756525657e-17,
	0.49307868575392305, 5.605083973871755e-18, 0.8699847180584174, 1.657385110740923e-17,
	0.49986032473301345, 1.3570481487732128e-17, 0.8661060303206567, 2.0097792543831333e-17,
	0.5066114548142574, -3.269413423618168e-17, 0.8621744799348805, 4.4132427578105805e-18,
	0.5133316639434712, 3.462454351928195e-17, 0.8581903068626604, -5.4374664844403745e-17,
	0.520020541953727, -3.983266745698455e-17, 0.8541537542773854, 5.420565102675287e-18,
	0.5266776805903868, -4.570502275386202e-17, 0.8500650685494202, 5.317093455811416e-17,
	0.5333026735360201, 5.129318115032044e-17, 0.8459244992310679, 1.549506647350329e-17,
	0.5398951164352044, 2.4263515104613218e-17, 0.8417322990413384, -2.7333362440269662e-17,
	0.5464546069192036, 8.399754840929507e-18, 0.8374887238505236, 4.3337026043948396e-17,
	0.5529807446305274, -6.743270455611733e-18, 0.8331940326645814, -1.8644878796734315e-19,
	0.5594731312473669, 1.575565514488728e-17, 0.8288484876093257, 1.1163935406617444e-17,
	0.565931370507906, 6.582459440388572e-18, 0.8244523539144292, 3.731817754380378e-17,
	0.5723550682345072, 2.6575872357215316e-17, 0.820005899897234, -3.912431748209128e-17,
	0.5787438323577704, -2.7958488978746924e-17, 0.8155093969463755, 1.3751889684983784e-18,
	0.5850972729404622, -5.4883972461161805e-17, 0.8109631195052179, -3.091333486122179e-17,
	0.5914150022013163, 5.0668448898504805e-17, 0.8063673450551039, 3.4650875715584734e-17,
	0.5976966345387015, 5.450323593054385e-17, 0.8017223540984184, 4.0134533311087014e-17,
	0.6039417865541566, 1.1794923733511837e-17, 0.7970284301414683, 5.0523609948487484e-17,
	0.6101500770757914, -1.479826990758988e-17, 0.7922858596771786, -2.9049779312834576e-17,
	0.616321127181551, -2.20849501729641e-17, 0.7874949321676061, -4.8080555050456574e-17,
	0.6224545602223437, -6.049035765709707e-18, 0.7826559400262728, -1.474071641211487e-17,
	0.6285500018450296, 3.5348797769901485e-17, 0.7777691786003179, -8.842419849333787e-18,
	0.6346070800152693, -3.4568582392624965e-17, 0.7728349461524715, 4.231014921891023e-17,
	0.6406254250402305, -4.700891123899386e-17, 0.7678535438428503, 4.9065912521116424e-17,
	0.6466046695911524, 4.567647714393287e-19, 0.7628252757105762, 1.6672995021546628e-17,
	0.652544448725766, -4.5598735935930497e-17, 0.7577504486552193, -1.878215121154204e-18,
	0.6584443999105676, -3.7736386700306717e-17, 0.7526293724180665, -1.2970993013150526e-17,
	0.6643041630429463, 1.5784492324431575e-17, 0.7474623595632162, 5.6627515613726676e-18,
	0.6701233804731629, 6.18353672557496e-18, 0.7422497254585013, -1.2339303604869521e-17,
	0.6759016970261789, -5.119389624609889e-17, 0.7369917882562408, -4.631213290368497e-17,
	0.6816387600233341, 4.410467313197903e-17, 0.7316888688738209, -1.0475824306512768e-17,
	0.6873342193038735, 2.8994333996809586e-17, 0.7263412909741086, -9.465899179857832e-18,
	0.692987727246318, -5.3543290798909455e-17, 0.7209493809456964, 3.494986701478816e-17,
	0.6985989387896817, 2.975439919188114e-17, 0.7155134678829815, 3.8181038992157035e-17,
	0.7041675114545337, -3.94095700584825e-17, 0.7100338835660797, 1.505272211891291e-17,
	0.7096931053638997, -9.468276958542117e-18, 0.7045109624405747, -4.451398614976226e-17,
	0.7151753832640076, -1.466099578328228e-17, 0.6989450415971057, -5.5261332036460915e-18,
	0.7206140105448711, 1.105694456137362e-17, 0.6933364607507934, 1.2822831674529969e-17,
	0.7260086552607126, -1.573621815339587e-17, 0.6876855622205048, 3.5430696752823923e-17,
	0.7313589881502236, 2.6553630314845697e-17, 0.6819926909079606, 2.8089333561212785e-17,
	0.7366646826566614, -4.639723374875429e-17, 0.6762581942766833, 3.497002399394248e-17,
	0.7419254149477796, 2.907482756987512e-18, 0.6704824223307908, 5.425808997571648e-17,
	0.7471408639355942, 2.937498788028212e-17, 0.6646657275936333, -5.2874573286772266e-17,
	0.7523107112959804, -4.07866182575664e-17, 0.6588084650862769, 1.7760805237087262e-17,
}
