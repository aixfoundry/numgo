package random

import (
	"math"

	"github.com/aixfoundry/numgo"
	"github.com/aixfoundry/numgo/internal/lapack"
)

// Second batch of Generator distributions, ported 1:1 from
// numpy/random/src/distributions/distributions.c (file:line refs per method)
// preserving the exact underlying draw sequence. Transcendentals use Go's
// math package; results are ULP-close to numpy but streams may diverge at
// rejection boundaries.

// chisquare: distributions.c:466 — 2 * standard_gamma(df/2).
func (g *Generator) Chisquare(df float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
		out.Data()[i] = 2.0 * g.StandardGammaScalar(df/2.0)
	}
	return out
}

// F: distributions.c:470 — (χ²(dfnum)*dfden) / (χ²(dfden)*dfnum).
func (g *Generator) F(dfnum, dfden float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
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
	for i := range n {
		out.Data()[i] = g.StandardNormalScalar() / g.StandardNormalScalar()
	}
	return out
}

// Pareto: distributions.c:482 — expm1(standard_exponential / a).
func (g *Generator) Pareto(a float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
		out.Data()[i] = math.Expm1(g.StandardExponentialScalar() / a)
	}
	return out
}

// Weibull: distributions.c:486 — standard_exponential ** (1/a); a == 0 → 0
// without consuming a draw.
func (g *Generator) Weibull(a float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
		if a == 0.0 {
			continue
		}
		out.Data()[i] = math.Pow(g.StandardExponentialScalar(), 1.0/a)
	}
	return out
}

// Power: distributions.c:493 — (−expm1(−standard_exponential)) ** (1/a).
func (g *Generator) Power(a float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
		out.Data()[i] = math.Pow(-math.Expm1(-g.StandardExponentialScalar()), 1.0/a)
	}
	return out
}

// Laplace: distributions.c:497 — inverse CDF; rejects U == 0 by redrawing.
func (g *Generator) Laplace(loc, scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
		var U float64
		for {
			U = g.nextDouble()
			if U > 0.0 {
				break
			}
		}
		lo := U + U
		hi := 2.0 - U - U
		sign := 1.0
		if U >= 0.5 {
			sign = -1.0
		}
		m := lo
		if hi < lo {
			m = hi
		}
		out.Data()[i] = loc + sign*scale*math.Log(m)
	}
	return out
}

// Gumbel: distributions.c:510 — loc − scale*log(−log(U)) with U = 1−uniform,
// rejecting U == 1 by redrawing.
func (g *Generator) Gumbel(loc, scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
		var U float64
		for {
			U = 1.0 - g.nextDouble()
			if U < 1.0 {
				break
			}
		}
		out.Data()[i] = loc - scale*math.Log(-math.Log(U))
	}
	return out
}

// Logistic: distributions.c:521 — inverse CDF; rejects U == 0 by redrawing.
func (g *Generator) Logistic(loc, scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
		var U float64
		for {
			U = g.nextDouble()
			if U > 0.0 {
				break
			}
		}
		out.Data()[i] = loc + scale*math.Log(U/(1.0-U))
	}
	return out
}

// Lognormal: distributions.c:532 — exp(normal(mean, sigma)).
func (g *Generator) Lognormal(mean, sigma float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
		out.Data()[i] = math.Exp(mean + sigma*g.StandardNormalScalar())
	}
	return out
}

// Rayleigh: distributions.c:536 — mode * sqrt(2 * standard_exponential).
func (g *Generator) Rayleigh(scale float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
		out.Data()[i] = scale * math.Sqrt(2.0*g.StandardExponentialScalar())
	}
	return out
}

// StandardT: distributions.c:540 — sqrt(df/2) * normal / sqrt(gamma(df/2)).
func (g *Generator) StandardT(df float64, size ...int) *arrayF {
	n := sizeOr1(size)
	out := newF(n)
	for i := range n {
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
	for i := range n {
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
	for i := range n {
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
	for i := range n {
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
	for i := range n {
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
		Z := math.Cos(0x1.921fb54442d18p+1 * U)
		W = (1.0 + s*Z) / (s + Z)
		Y := kappa * (s - W)
		V := g.nextDouble()
		if (Y*(2.0-Y)-V >= 0) || (math.Log(Y/V)+1.0-Y >= 0) {
			break
		}
	}
	U := g.nextDouble()
	result := math.Cos(W)
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
	for i := range n {
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
	for i := range n {
		r := math.Log1p(-p)
		for {
			V := g.nextDouble()
			if V >= p {
				out.Data()[i] = 1
				break
			}
			U := g.nextDouble()
			q := -math.Expm1(r * U)
			if V <= q*q {
				result := int64(math.Floor(1.0 + math.Log(V)/math.Log(q)))
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
	z := math.Ceil(-g.StandardExponentialScalar() / math.Log1p(-p))
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
	for i := range n {
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
	b := math.Pow(2.0, am1)
	const randIntMax = 9223372036854775807.0
	Umin := math.Pow(randIntMax, -am1)
	for {
		U01 := g.nextDouble()
		U := U01*Umin + (1.0 - U01)
		V := g.nextDouble()
		X := math.Floor(math.Pow(U, -1.0/am1))
		if X > randIntMax || X < 1.0 {
			continue
		}
		T := math.Pow(1.0+1.0/X, am1)
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
	for i := range n {
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
	return (lf+0.5)*math.Log(lf) - lf + (halfln2pi + (1.0/lf)*(1.0/12.0-1.0/(360.0*lf*lf)))
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
	computedSample := min(popsize-sample, sample)
	mingoodbad := min(bad, good)
	maxgoodbad := max(bad, good)
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
	bound := min(mingoodbad, computedSample)
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
		if 2.0*math.Log(U) <= T {
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
	for i := range n {
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
	for i := range cnt {
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
			for i := range rows {
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
	for i := range rows {
		acc := 0.0
		for j := range k {
			v := g.StandardGammaScalar(alpha[j])
			data[i*k+j] = v
			acc = acc + v
		}
		invacc := 1.0 / acc
		for j := range k {
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
	for i := range rows {
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
	for r := range rows {
		// BLAS dgemm (numpy: x @ L.T) accumulates the dot product first,
		// then adds the mean; a fused mean-first loop rounds differently.
		for i := range dim {
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
				for k := range numColors {
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
				for k := range numColors {
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
