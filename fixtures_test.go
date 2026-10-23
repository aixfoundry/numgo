package numgo

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Golden-fixture harness: every case in testdata/fixtures/*.json is replayed
// through the dispatch table below and compared to numpy's output (generated
// by scripts/gen_fixtures.py) with tolerance 1e-12 rel + 1e-12 abs.
//
// Case schema:
//
//	{"name", "func", "args": [...], "inputs": [ {shape, data}, ... ],
//	 "want": {shape, data}}
//
// Conventions (mirrored by the generator's header):
//   - Methods are called on inputs[0]; package-level functions receive all
//     inputs (create functions none).
//   - args decode positionally per specTable: f=float64, i=[]int,
//     F=[]float64, b=[]bool, r=Range{start,stop,step}. Varying-arity args
//     (axes, shapes) arrive as one []int.
//
// NaN/Inf arrive as the JSON strings "NaN"/"Inf"/"-Inf" and compare equal to
// the corresponding IEEE values (NaN == NaN in want positions).

const (
	tolAbs = 1e-12
	tolRel = 1e-12
)

type fixtureCase struct {
	Name string            `json:"name"`
	Func string            `json:"func"`
	Args []json.RawMessage `json:"args"`
	In   []fixtureArr      `json:"inputs"`
	Want fixtureArr        `json:"want"`
}

type fixtureArr struct {
	Shape []int             `json:"shape"`
	Data  []json.RawMessage `json:"data"`
}

// decodeNum parses one JSON element, resolving the "NaN"/"Inf"/"-Inf"
// string encoding used by the generator.
func decodeNum(raw json.RawMessage) (float64, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "NaN":
			return math.NaN(), nil
		case "Inf":
			return math.Inf(1), nil
		case "-Inf":
			return math.Inf(-1), nil
		default:
			return 0, fmt.Errorf("bad special value %q", s)
		}
	}
	var f float64
	return f, json.Unmarshal(raw, &f)
}

func (fa fixtureArr) values(t *testing.T) []float64 {
	t.Helper()
	out := make([]float64, len(fa.Data))
	for i, raw := range fa.Data {
		v, err := decodeNum(raw)
		if err != nil {
			t.Fatalf("decode data[%d]: %v", i, err)
		}
		out[i] = v
	}
	return out
}

func (fa fixtureArr) array(t *testing.T) *Array[float64] {
	t.Helper()
	return FromSlice(fa.values(t), fa.Shape...)
}

// decodeArgs turns the case's raw args into []any per the spec string. A
// trailing '*' repeats the last letter for every remaining arg (variadic
// axes/shapes); arity is checked exactly when there is no '*'.
func decodeArgs(t *testing.T, raws []json.RawMessage, spec string) []any {
	t.Helper()
	if rest := strings.TrimSuffix(spec, "*"); rest != spec {
		if len(raws) < len(rest) {
			t.Fatalf("spec %q needs >=%d args, case has %d", spec, len(rest), len(raws))
		}
		spec = rest + strings.Repeat(string(rest[len(rest)-1]), len(raws)-len(rest))
	}
	if len(spec) != len(raws) {
		t.Fatalf("arg spec %q has %d entries, case has %d args", spec, len(spec), len(raws))
	}
	out := make([]any, len(raws))
	for i, raw := range raws {
		switch spec[i] {
		case 'f':
			var v float64
			fatalIf(t, json.Unmarshal(raw, &v), "arg %d", i)
			out[i] = v
		case 'd': // single int arriving as a JSON number
			var v float64
			fatalIf(t, json.Unmarshal(raw, &v), "arg %d", i)
			out[i] = int(v)
		case 'i':
			var v []int
			fatalIf(t, json.Unmarshal(raw, &v), "arg %d", i)
			out[i] = v
		case 'F':
			var v []float64
			fatalIf(t, json.Unmarshal(raw, &v), "arg %d", i)
			out[i] = v
		case 'b':
			var v []bool
			fatalIf(t, json.Unmarshal(raw, &v), "arg %d", i)
			out[i] = v
		case 'r':
			var v struct {
				Start, Stop, Step int
			}
			fatalIf(t, json.Unmarshal(raw, &v), "arg %d", i)
			out[i] = Range{Start: v.Start, Stop: v.Stop, Step: v.Step}
		default:
			t.Fatalf("bad spec letter %q", spec[i])
		}
	}
	return out
}

func fatalIf(t *testing.T, err error, format string, a ...any) {
	t.Helper()
	if err != nil {
		t.Fatalf(format+": %v", append(a, err)...)
	}
}

func floats(args []any, i int) []float64 { return args[i].([]float64) }
func ints(args []any, i int) []int       { return args[i].([]int) }
func boolean(args []any, i int) []bool   { return args[i].([]bool) }
func num(args []any, i int) float64      { return args[i].(float64) }
func whole(args []any, i int) int        { return args[i].(int) }
func rng(args []any, i int) Range        { return args[i].(Range) }

// runCase executes one fixture case through the dispatch table and compares
// the result with the recorded want.
func runCase(t *testing.T, c fixtureCase) {
	t.Helper()
	spec, ok := specTable[c.Func]
	if !ok {
		t.Fatalf("no dispatch entry for func %s", c.Func)
	}
	args := decodeArgs(t, c.Args, spec)
	a := func() *Array[float64] {
		if len(c.In) == 0 {
			return nil // create functions have no receiver
		}
		return c.In[0].array(t)
	}()

	// arrResult holds every shape/value result; idxResult holds []int
	// results (ArgMax/ArgMin/Argsort); scalarResult holds bool/float64.
	var got *Array[float64]
	var gotIdx []int
	var gotBool bool
	var gotScalar float64
	switch c.Func {
	// ---- create (package-level, no receiver) ----
	case "Zeros":
		got = Zeros[float64](ints(args, 0)...)
	case "Ones":
		got = Ones[float64](ints(args, 0)...)
	case "Full":
		got = Full(num(args, 0), ints(args, 1)...)
	case "Arange":
		got = Arange(num(args, 0), num(args, 1), num(args, 2))
	case "Linspace":
		got = Linspace(num(args, 0), num(args, 1), whole(args, 2))
	case "Eye":
		got = Eye[float64](whole(args, 0))
	case "Identity":
		got = Identity[float64](whole(args, 0))
	// ---- views (methods on a) ----
	case "Slice":
		rs := make([]Range, len(args))
		for i := range args {
			rs[i] = rng(args, i)
		}
		got = a.Slice(rs...)
	case "Transpose":
		got = a.Transpose(ints(args, 0)...)
	case "Reshape":
		got = a.Reshape(ints(args, 0)...)
	case "Squeeze":
		got = a.Squeeze(ints(args, 0)...)
	case "ExpandDims":
		got = a.ExpandDims(whole(args, 0))
	case "SwapAxes":
		got = a.SwapAxes(whole(args, 0), whole(args, 1))
	case "Ravel":
		got = a.Ravel()
	case "BroadcastTo":
		got = a.BroadcastTo(ints(args, 0)...)
	// ---- ufunc ----
	case "Add":
		got = a.Add(c.In[1].array(t))
	case "Sub":
		got = a.Sub(c.In[1].array(t))
	case "Mul":
		got = a.Mul(c.In[1].array(t))
	case "Div":
		got = a.Div(c.In[1].array(t))
	case "AddValue":
		got = a.AddValue(num(args, 0))
	case "SubValue":
		got = a.SubValue(num(args, 0))
	case "MulValue":
		got = a.MulValue(num(args, 0))
	case "DivValue":
		got = a.DivValue(num(args, 0))
	// ---- reduce ----
	case "Sum":
		got = a.Sum(ints(args, 0)...)
	case "SumKeepdims":
		got = a.SumKeepdims(ints(args, 0)...)
	case "Prod":
		got = a.Prod(ints(args, 0)...)
	case "ProdKeepdims":
		got = a.ProdKeepdims(ints(args, 0)...)
	case "Max":
		got = a.Max(ints(args, 0)...)
	case "MaxKeepdims":
		got = a.MaxKeepdims(ints(args, 0)...)
	case "Min":
		got = a.Min(ints(args, 0)...)
	case "MinKeepdims":
		got = a.MinKeepdims(ints(args, 0)...)
	case "Mean":
		got = a.Mean(ints(args, 0)...)
	case "MeanKeepdims":
		got = a.MeanKeepdims(ints(args, 0)...)
	case "Var":
		got = a.Var(ints(args, 0)...)
	case "VarKeepdims":
		got = a.VarKeepdims(ints(args, 0)...)
	case "VarDDof":
		got = a.VarDDof(whole(args, 0), ints(args, 1)...)
	case "Std":
		got = a.Std(ints(args, 0)...)
	case "StdDDof":
		got = a.StdDDof(whole(args, 0), ints(args, 1)...)
	case "ArgMax":
		gotIdx = a.ArgMax(whole(args, 0))
	case "ArgMin":
		gotIdx = a.ArgMin(whole(args, 0))
	case "Any":
		gotBool = a.Any()
	case "All":
		gotBool = a.All()
	case "CumSum":
		got = a.CumSum(ints(args, 0)...)
	case "CumProd":
		got = a.CumProd(ints(args, 0)...)
	// ---- manip (package-level) ----
	case "Concatenate":
		others := make([]*Array[float64], 0, len(c.In)-1)
		for _, in := range c.In[1:] {
			others = append(others, in.array(t))
		}
		got = Concatenate(whole(args, 0), append([]*Array[float64]{a}, others...)...)
	case "Stack":
		others := make([]*Array[float64], 0, len(c.In)-1)
		for _, in := range c.In[1:] {
			others = append(others, in.array(t))
		}
		got = Stack(whole(args, 0), append([]*Array[float64]{a}, others...)...)
	case "Tile":
		got = Tile(a, ints(args, 0)...)
	case "Repeat":
		got = Repeat(a, whole(args, 0), whole(args, 1))
	case "Flip":
		got = Flip(a, ints(args, 0)...)
	case "Roll":
		got = Roll(a, whole(args, 0), ints(args, 1)...)
	// ---- sort ----
	case "Sort":
		got = a.Sort(ints(args, 0)...)
	case "Argsort":
		gotIdx = a.Argsort(whole(args, 0))
	case "Unique":
		got = a.Unique()
	// ---- math (unary methods) ----
	case "Abs", "Sqrt", "Cbrt", "Exp", "Exp2", "Log", "Log2", "Log10",
		"Log1p", "Expm1", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan",
		"Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh", "Floor",
		"Ceil", "Round", "Trunc":
		got = unaryDispatch[c.Func](a)
	// ---- dot ----
	case "Dot":
		got = a.Dot(c.In[1].array(t))
	case "Matmul":
		got = a.Matmul(c.In[1].array(t))
	case "Vdot":
		gotScalar = a.Vdot(c.In[1].array(t))
	// ---- fancy ----
	case "Take":
		got = a.Take(ints(args, 0), whole(args, 1))
	case "Compress":
		got = a.Compress(boolean(args, 0), whole(args, 1))
	default:
		t.Fatalf("no dispatch for %s", c.Func)
	}

	want := c.Want
	switch {
	case got != nil:
		checkArray(t, c, got, want)
	case gotIdx != nil:
		wantVals := want.values(t)
		if len(gotIdx) != len(wantVals) {
			t.Fatalf("%s: got %d indices, want %d", c.Name, len(gotIdx), len(wantVals))
		}
		for i, g := range gotIdx {
			if float64(g) != wantVals[i] {
				t.Fatalf("%s: index %d = %d, want %v", c.Name, i, g, wantVals[i])
			}
		}
	case c.Func == "Any" || c.Func == "All":
		var wantBool bool
		fatalIf(t, json.Unmarshal(want.Data[0], &wantBool), "%s: decode want", c.Name)
		if gotBool != wantBool {
			t.Fatalf("%s: got %v, want %v", c.Name, gotBool, wantBool)
		}
	default: // Vdot scalar
		wantVals := want.values(t)
		if len(wantVals) != 1 || !closeEnough(gotScalar, wantVals[0]) {
			t.Fatalf("%s: Vdot = %v, want %v", c.Name, gotScalar, wantVals)
		}
	}
}

// checkArray compares shape and every element with rel+abs tolerance and
// exact NaN/Inf matching.
func checkArray(t *testing.T, c fixtureCase, got *Array[float64], want fixtureArr) {
	t.Helper()
	gotShape := got.Shape()
	if fmt.Sprint(gotShape) != fmt.Sprint(want.Shape) {
		t.Fatalf("%s: shape = %v, want %v", c.Name, gotShape, want.Shape)
	}
	wantVals := want.values(t)
	if got.Size() != len(wantVals) {
		t.Fatalf("%s: size = %d, want %d", c.Name, got.Size(), len(wantVals))
	}
	for i, w := range wantVals {
		g := float64(got.Ravel().Data()[i])
		if !closeEnough(g, w) {
			t.Fatalf("%s: element %d = %v, want %v", c.Name, i, g, w)
		}
	}
}

func closeEnough(g, w float64) bool {
	if math.IsNaN(w) {
		return math.IsNaN(g)
	}
	if math.IsInf(w, 0) {
		return g == w
	}
	return math.Abs(g-w) <= tolAbs+tolRel*math.Abs(w)
}

// unaryDispatch maps the unary math names to their methods. One entry per
// function; the dispatch in runCase routes whole names here verbatim.
var unaryDispatch = map[string]func(*Array[float64]) *Array[float64]{
	"Abs":   (*Array[float64]).Abs,
	"Sqrt":  (*Array[float64]).Sqrt,
	"Cbrt":  (*Array[float64]).Cbrt,
	"Exp":   (*Array[float64]).Exp,
	"Exp2":  (*Array[float64]).Exp2,
	"Log":   (*Array[float64]).Log,
	"Log2":  (*Array[float64]).Log2,
	"Log10": (*Array[float64]).Log10,
	"Log1p": (*Array[float64]).Log1p,
	"Expm1": (*Array[float64]).Expm1,
	"Sin":   (*Array[float64]).Sin,
	"Cos":   (*Array[float64]).Cos,
	"Tan":   (*Array[float64]).Tan,
	"Asin":  (*Array[float64]).Asin,
	"Acos":  (*Array[float64]).Acos,
	"Atan":  (*Array[float64]).Atan,
	"Sinh":  (*Array[float64]).Sinh,
	"Cosh":  (*Array[float64]).Cosh,
	"Tanh":  (*Array[float64]).Tanh,
	"Asinh": (*Array[float64]).Asinh,
	"Acosh": (*Array[float64]).Acosh,
	"Atanh": (*Array[float64]).Atanh,
	"Floor": (*Array[float64]).Floor,
	"Ceil":  (*Array[float64]).Ceil,
	"Round": (*Array[float64]).Round,
	"Trunc": (*Array[float64]).Trunc,
}

// specTable maps func -> positional arg kinds (see header comment). A
// trailing '*' repeats the last kind for any remaining args (variadic).
// Keys without entries here take no args.
var specTable = map[string]string{
	"Zeros": "i*", "Ones": "i*", "Full": "fi*", "Arange": "fff", "Linspace": "ffd",
	"Eye": "d", "Identity": "d",
	"Slice": "r*", "Transpose": "i*", "Reshape": "i*", "Squeeze": "i*",
	"ExpandDims": "d", "SwapAxes": "dd", "Ravel": "", "BroadcastTo": "i*",
	"Add": "", "Sub": "", "Mul": "", "Div": "",
	"AddValue": "f", "SubValue": "f", "MulValue": "f", "DivValue": "f",
	"Sum": "i*", "SumKeepdims": "i*", "Prod": "i*", "ProdKeepdims": "i*",
	"Max": "i*", "MaxKeepdims": "i*", "Min": "i*", "MinKeepdims": "i*",
	"Mean": "i*", "MeanKeepdims": "i*", "Var": "i*", "VarKeepdims": "i*",
	"VarDDof": "di*", "Std": "i*", "StdDDof": "di*",
	"ArgMax": "d", "ArgMin": "d", "Any": "", "All": "",
	"CumSum": "i*", "CumProd": "i*",
	"Concatenate": "d", "Stack": "d", "Tile": "i*", "Repeat": "dd",
	"Flip": "i*", "Roll": "di*",
	"Sort": "i*", "Argsort": "d", "Unique": "",
	"Dot": "", "Matmul": "", "Vdot": "",
	"Take": "id", "Compress": "bd",
	"Abs": "", "Sqrt": "", "Cbrt": "", "Exp": "", "Exp2": "",
	"Log": "", "Log2": "", "Log10": "", "Log1p": "", "Expm1": "",
	"Sin": "", "Cos": "", "Tan": "", "Asin": "", "Acos": "", "Atan": "",
	"Sinh": "", "Cosh": "", "Tanh": "", "Asinh": "", "Acosh": "", "Atanh": "",
	"Floor": "", "Ceil": "", "Round": "", "Trunc": "",
}

// TestFixtures replays every golden case from testdata/fixtures.
func TestFixtures(t *testing.T) {
	dir := testdataFixturesDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".json" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		var cases []fixtureCase
		if err := json.Unmarshal(b, &cases); err != nil {
			t.Fatalf("%s: %v", e.Name(), err)
		}
		domain := e.Name()
		t.Run(domain, func(t *testing.T) {
			for _, c := range cases {
				c := c
				total++
				t.Run(c.Name, func(t *testing.T) {
					t.Parallel()
					runCase(t, c)
				})
			}
		})
	}
}

func testdataFixturesDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "testdata", "fixtures")
}
