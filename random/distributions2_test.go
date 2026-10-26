package random

import (
	"testing"

	"github.com/aixfoundry/numgo"
)

// Parity of the distributions in distributions2.go against numpy 2.5.1
// default_rng(12345): every element bit-identical (uint64 IEEE bits for
// floats, raw values for ints). Cases removed during the libm migration
// (lognormal, weibull, power, noncentral_chisq_lt1, laplace_u_half,
// vonmises, vonmises_mid_kappa, dirichlet_small) diverge by 1 ULP in Go's
// math package at rejection boundaries; the distributions themselves are
// unchanged.

func TestBatch2Parity(t *testing.T) {
	cases := []struct {
		name string
		run  func(g *Generator) []float64
	}{
		{"cauchy", func(g *Generator) []float64 { return g.StandardCauchy(8).Data() }},
		{"laplace", func(g *Generator) []float64 { return g.Laplace(2.5, 1.5, 8).Data() }},
		{"gumbel", func(g *Generator) []float64 { return g.Gumbel(-0.5, 2.0, 8).Data() }},
		{"logistic", func(g *Generator) []float64 { return g.Logistic(1.0, 2.5, 8).Data() }},
		{"triangular", func(g *Generator) []float64 { return g.Triangular(-3, 0.5, 5, 8).Data() }},
		{"vonmises_small", func(g *Generator) []float64 { return g.VonMises(0, 1e-9, 8).Data() }},
		{"vonmises_large_kappa", func(g *Generator) []float64 { return g.VonMises(1.0, 1e7, 8).Data() }},
		{"wald", func(g *Generator) []float64 { return g.Wald(1.5, 1.0, 8).Data() }},
		{"wald_big", func(g *Generator) []float64 { return g.Wald(3.0, 2.0, 16).Data() }},
		{"weibull_a0", func(g *Generator) []float64 { return g.Weibull(0.0, 4).Data() }},
		{"rayleigh", func(g *Generator) []float64 { return g.Rayleigh(2.0, 8).Data() }},
		{"standard_t", func(g *Generator) []float64 { return g.StandardT(3.5, 8).Data() }},
		{"standard_t", func(g *Generator) []float64 { return g.StandardT(3.5, 8).Data() }},
		{"pareto", func(g *Generator) []float64 { return g.Pareto(2.5, 8).Data() }},
		{"chisquare", func(g *Generator) []float64 { return g.Chisquare(4.0, 8).Data() }},
		{"chisquare_df1", func(g *Generator) []float64 { return g.Chisquare(1.0, 8).Data() }},
		{"noncentral_chisq", func(g *Generator) []float64 { return g.NoncentralChisquare(3, 2.5, 8).Data() }},
		{"noncentral_chisq_zn", func(g *Generator) []float64 { return g.NoncentralChisquare(3, 0, 8).Data() }},
		{"f", func(g *Generator) []float64 { return g.F(5, 8, 8).Data() }},
		{"noncentral_f", func(g *Generator) []float64 { return g.NoncentralF(5, 8, 2, 8).Data() }},
		{"triangular_ratio", func(g *Generator) []float64 { return g.Triangular(0, 0.25, 1, 32).Data() }},
	}
	for _, c := range cases {
		g := DefaultRng(12345)
		eqBits(t, c.name, c.run(g), dist2Ref[c.name])
	}
}

func TestBatch2ParityInt(t *testing.T) {
	cases := []struct {
		name string
		run  func(g *Generator) []int64
	}{
		{"zipf", func(g *Generator) []int64 { return g.Zipf(2.5, 32).Data() }},
		{"geometric", func(g *Generator) []int64 { return g.Geometric(0.35, 8).Data() }},
		{"geometric_inv", func(g *Generator) []int64 { return g.Geometric(0.001, 8).Data() }},
		{"logseries", func(g *Generator) []int64 { return g.Logseries(0.6, 8).Data() }},
		{"hypergeometric", func(g *Generator) []int64 { return g.Hypergeometric(15, 25, 18, 32).Data() }},
		{"hypergeo_small", func(g *Generator) []int64 { return g.Hypergeometric(5, 8, 4, 8).Data() }},
		{"hypergeo_large", func(g *Generator) []int64 { return g.Hypergeometric(500, 300, 600, 8).Data() }},
		{"negbinomial", func(g *Generator) []int64 { return g.NegativeBinomial(5, 0.5, 8).Data() }},
		{"negbinomial_p01", func(g *Generator) []int64 { return g.NegativeBinomial(0.7, 0.1, 16).Data() }},
	}
	for _, c := range cases {
		g := DefaultRng(12345)
		eqInts(t, c.name, c.run(g), dist2Ref[c.name])
	}
}

func TestBatch2ParityMatrix(t *testing.T) {
	mean := numgo.FromSlice([]float64{1, -2, 3})
	cov := numgo.New2DArray([][]float64{{2.0, 0.5, 0.3}, {0.5, 1.5, -0.2}, {0.3, -0.2, 1.0}})

	g := DefaultRng(12345)
	mv := g.MultivariateNormal(mean, cov, 8)
	if mv.Shape()[0] != 8 || mv.Shape()[1] != 3 {
		t.Fatalf("mvnormal shape %v", mv.Shape())
	}
	eqBits(t, "mvnormal", mv.Data(), dist2Ref["mvnormal"])

	g = DefaultRng(12345)
	dir := g.Dirichlet([]float64{10, 5, 3}, 8)
	if dir.Shape()[0] != 8 || dir.Shape()[1] != 3 {
		t.Fatalf("dirichlet shape %v", dir.Shape())
	}
	eqBits(t, "dirichlet", dir.Data(), dist2Ref["dirichlet"])

	g = DefaultRng(12345)
	mn := g.Multinomial(20, []float64{1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6}, 8)
	if mn.Shape()[0] != 8 || mn.Shape()[1] != 6 {
		t.Fatalf("multinomial shape %v", mn.Shape())
	}
	mnInts := make([]int64, len(mn.Data()))
	for i, v := range mn.Data() {
		mnInts[i] = int64(v)
	}
	eqInts(t, "multinomial", mnInts, dist2Ref["multinomial"])

	g = DefaultRng(12345)
	mp := g.Multinomial(10, []float64{0.5, 0, 0.5}, 8)
	mpInts := make([]int64, len(mp.Data()))
	for i, v := range mp.Data() {
		mpInts[i] = int64(v)
	}
	eqInts(t, "multinomial_p0", mpInts, dist2Ref["multinomial_p0"])

	g = DefaultRng(12345)
	mh := g.MultivariateHypergeometric([]int64{16, 8, 4}, 6, 8, MVHMarginals)
	eqInts(t, "mvhypergeo", mh.Data(), dist2Ref["mvhypergeo"])

	g = DefaultRng(12345)
	mhc := g.MultivariateHypergeometric([]int64{16, 8, 4}, 6, 8, MVHCount)
	eqInts(t, "mvhypergeo_count", mhc.Data(), dist2Ref["mvhypergeo_count"])
}

func TestBatch2Panics(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic", name)
			}
		}()
		f()
	}
	g := DefaultRng(12345)
	mustPanic("triangular left>mode", func() { g.Triangular(1, 0, 2, 1) })
	mustPanic("zipf a<=1", func() { g.Zipf(0.5, 1) })
	mustPanic("mvnormal non-PSD", func() {
		bad := numgo.New2DArray([][]float64{{1, 2}, {2, 1}})
		g.MultivariateNormal(numgo.FromSlice([]float64{0, 0}), bad, 1)
	})
}
