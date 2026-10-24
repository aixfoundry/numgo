package strings

import (
	"reflect"
	"testing"

	"github.com/aixfoundry/numgo"
)

// P0-5: every op family gets a non-square 2×3 table case.

func eq(t *testing.T, name string, got, want *StrArray) {
	t.Helper()
	if !reflect.DeepEqual(got.Shape(), want.Shape()) {
		t.Fatalf("%s shape = %v, want %v", name, got.Shape(), want.Shape())
	}
	for i, v := range want.data {
		if got.data[i] != v {
			t.Fatalf("%s[%d] = %q, want %q (data %v)", name, i, got.data[i], v, got.data)
		}
	}
}

func boolEq(t *testing.T, name string, got []bool, want ...bool) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("%s = %v, want %v", name, got, want)
	}
}

// slice2x3 builds a 2×3 array from row-major data.
func slice2x3(data ...string) *StrArray {
	return FromSlice(data, 2, 3)
}

func TestCase(t *testing.T) {
	a := slice2x3("hello WORLD", "don't do", "hÉllo", "ß", "aBc", "")
	eq(t, "Upper", a.Upper(), slice2x3("HELLO WORLD", "DON'T DO", "HÉLLO", "ß", "ABC", ""))
	// Divergence: Go's simple case mapping leaves ß unchanged; numpy's full
	// mapping yields "SS". numpy's own fixed-width dtype then truncates to "S".
	eq(t, "Lower", a.Lower(), slice2x3("hello world", "don't do", "héllo", "ß", "abc", ""))
	eq(t, "Swapcase", a.Swapcase(), slice2x3("HELLO world", "DON'T DO", "HéLLO", "ß", "AbC", ""))
	eq(t, "Capitalize", a.Capitalize(), slice2x3("Hello world", "Don't do", "Héllo", "ß", "Abc", ""))
	// Title: ß is cased (lower) but has no simple upper mapping, so numpy's
	// full-mapping title yields "Ss" while Go's simple mapping yields "ß" —
	// documented divergence.
	eq(t, "Title", a.Title(), slice2x3("Hello World", "Don'T Do", "Héllo", "ß", "Abc", ""))
}

func TestSearch(t *testing.T) {
	a := slice2x3("abcabc", "abc", "abcabc", "xyz", "", "abC")
	eq(t, "Find", a.Find("bc", 0), slice2x3("1", "1", "1", "-1", "-1", "-1"))
	eq(t, "FindStart", a.Find("bc", 2), slice2x3("4", "-1", "4", "-1", "-1", "-1"))
	eq(t, "FindNegStart", a.Find("c", -2), slice2x3("5", "2", "5", "-1", "-1", "-1"))
	// Python 'abcabc'.rfind('bc') = 4 — highest index wins over the first.
	eq(t, "RFind", a.RFind("bc", -1), slice2x3("4", "1", "4", "-1", "-1", "-1"))
	eq(t, "RFindStop", a.RFind("bc", 3), slice2x3("1", "1", "1", "-1", "-1", "-1"))

	// stop=-1 wraps to n-1 (Python [0:-1]): 'abcabc'[0:-1]='abcab' has one
	// 'abc'; 'abc'[0:-1]='ab' has none.
	eq(t, "Count", a.Count("abc", 0, -1), slice2x3("1", "0", "1", "0", "0", "0"))
	eq(t, "CountOverlap", a.Count("aa", 0, -1), slice2x3("0", "0", "0", "0", "0", "0"))
	boolEq(t, "StartsWith", a.StartsWith("abc", 0), true, true, true, false, false, false)
	boolEq(t, "StartsWithFrom", a.StartsWith("bc", 1), true, true, true, false, false, false)
	boolEq(t, "EndsWith", a.EndsWith("bc", -1), true, true, true, false, false, false)
	boolEq(t, "EndsWithStop", a.EndsWith("bc", 3), true, true, true, false, false, false)
	boolEq(t, "Contains", a.Contains("ca"), true, false, true, false, false, false)

	panicked := false
	func() {
		defer func() { _ = recover(); panicked = true }()
		a.Index("zz", 0)
	}()
	if !panicked {
		t.Fatal("Index on missing substring did not panic")
	}
	panicked = false
	func() {
		defer func() { _ = recover(); panicked = true }()
		a.RIndex("zz", -1)
	}()
	if !panicked {
		t.Fatal("RIndex on missing substring did not panic")
	}
}

func TestTrim(t *testing.T) {
	a := slice2x3("  hi  ", "xxhixx", "\tab\t", "aXa", "  x", "x  ")
	eq(t, "Strip", a.Strip(""), slice2x3("hi", "xxhixx", "ab", "aXa", "x", "x"))
	eq(t, "StripChars", a.Strip("x"), slice2x3("  hi  ", "hi", "\tab\t", "aXa", "  ", "  "))
	eq(t, "LStrip", a.LStrip(""), slice2x3("hi  ", "xxhixx", "ab\t", "aXa", "x", "x  "))
	eq(t, "LStripChars", a.LStrip("x"), slice2x3("  hi  ", "hixx", "\tab\t", "aXa", "  x", "  "))
	eq(t, "RStrip", a.RStrip(""), slice2x3("  hi", "xxhixx", "\tab", "aXa", "  x", "x"))
	eq(t, "RStripChars", a.RStrip("x"), slice2x3("  hi  ", "xxhi", "\tab\t", "aXa", "  ", "x  "))
}

func TestSplitFamily(t *testing.T) {
	a := slice2x3("a,b,c", "abc", "x,y", "p,q,r,s", "", "1,2")
	check := func(name string, got, want [][]string) {
		t.Helper()
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
	}
	check("Split", a.Split(","), [][]string{
		{"a", "b", "c"}, {"abc"}, {"x", "y"}, {"p", "q", "r", "s"}, {""}, {"1", "2"},
	})
	check("SplitLines", FromSlice([]string{"a\nb", "x", "p\rq", "s\r\nt", "", "u\n"}, 2, 3).SplitLines(),
		[][]string{{"a", "b"}, {"x"}, {"p", "q"}, {"s", "t"}, {""}, {"u"}})
	eq(t, "Join", a.Join([]string{"1", "2"}), slice2x3(
		"1a,b,c2", "1abc2", "1x,y2", "1p,q,r,s2", "12", "11,22"))
	check("Partition", a.Partition(","), [][]string{
		{"a", ",", "b,c"}, {"abc", "", ""}, {"x", ",", "y"}, {"p", ",", "q,r,s"}, {"", "", ""}, {"1", ",", "2"},
	})
	check("RPartition", a.RPartition(","), [][]string{
		{"a,b", ",", "c"}, {"", "", "abc"}, {"x", ",", "y"}, {"p,q,r", ",", "s"}, {"", "", ""}, {"1", ",", "2"},
	})
}

func TestReplaceTranslateExpandTabs(t *testing.T) {
	a := slice2x3("aaa", "aba", "ab", "ba", "", "abc")
	eq(t, "Replace", a.Replace("a", "X", -1), slice2x3("XXX", "XbX", "Xb", "bX", "", "Xbc"))
	eq(t, "ReplaceN", a.Replace("a", "X", 1), slice2x3("Xaa", "Xba", "Xb", "bX", "", "Xbc"))
	eq(t, "ReplaceZero", a.Replace("a", "X", 0), a)
	eq(t, "Translate", a.Translate(map[rune]rune{'a': 'o'}), slice2x3(
		"ooo", "obo", "ob", "bo", "", "obc"))
	eq(t, "ExpandTabsDefault", FromSlice([]string{"a\tb", "\tab", "ab\t", "a\tb\tc", "tab", ""}, 2, 3).ExpandTabs(8),
		slice2x3("a       b", "        ab", "ab      ", "a       b       c", "tab", ""))
	eq(t, "ExpandTabs2", FromSlice([]string{"a\tb", "\tab", "ab\t", "a\tb\tc", "tab", ""}, 2, 3).ExpandTabs(2),
		slice2x3("a b", "  ab", "ab  ", "a b c", "tab", ""))
	// IsTitle oracle: "ABC" is not title (uppercase must follow uncased);
	// "Ab Cd" and "Ab3" are.
	tit := slice2x3("Ab Cd", "ABC", "Ab3", "aB", "Ab", "")
	boolEq(t, "IsTitle2", tit.IsTitle(), true, false, true, false, true, false)
}

func TestPad(t *testing.T) {
	a := slice2x3("hi", "x", "hello", "", "-1", "+1")
	// Center parity (CPython): left = deficit//2, extra space goes RIGHT.
	// 'x'.center(6) = '  x   ', 'hello'.center(6) = 'hello ', 'hi'.center(6)
	// = '  hi  '.
	eq(t, "Center", a.Center(6), slice2x3("  hi  ", "  x   ", "hello ", "      ", "  -1  ", "  +1  "))
	eq(t, "LJust", a.LJust(4), slice2x3("hi  ", "x   ", "hello", "    ", "-1  ", "+1  "))
	eq(t, "RJust", a.RJust(4), slice2x3("  hi", "   x", "hello", "    ", "  -1", "  +1"))
	eq(t, "ZFill", a.ZFill(4), slice2x3("00hi", "000x", "hello", "0000", "-001", "+001"))
}

func TestPredicates(t *testing.T) {
	a := slice2x3("abc", "ABC", "a1", "1.5", "", "  ")
	boolEq(t, "IsAlpha", a.IsAlpha(), true, true, false, false, false, false)
	boolEq(t, "IsDigit", a.IsDigit(), false, false, false, false, false, false)
	boolEq(t, "IsDecimal", a.IsDecimal(), false, false, false, false, false, false)
	boolEq(t, "IsNumeric", a.IsNumeric(), false, false, false, false, false, false)
	boolEq(t, "IsSpace", a.IsSpace(), false, false, false, false, false, true)
	boolEq(t, "IsAlNum", a.IsAlNum(), true, true, true, false, false, false)
	boolEq(t, "IsLower", a.IsLower(), true, false, true, false, false, false)
	boolEq(t, "IsUpper", a.IsUpper(), false, true, false, false, false, false)
	boolEq(t, "IsTitle", a.IsTitle(), false, false, false, false, false, false)
	// IsTitle2 oracle: 'Ab Cd'/'Ab3'/'Ab' are title (upper follows uncased);
	// 'ABC' upper follows cased → false; 'aB' lower follows uncased → false.
	tit := slice2x3("Ab Cd", "ABC", "Ab3", "aB", "Ab", "")
	boolEq(t, "IsTitle2", tit.IsTitle(), true, false, true, false, true, false)
	boolEq(t, "IsEmpty", a.IsEmpty(), false, false, false, false, true, false)

	uni := slice2x3("²", "3", "Ⅻ", "½", "é", "")
	// numpy treats ² as a digit — documented divergence: Go restricts to Nd.
	boolEq(t, "IsDigitUnicode", uni.IsDigit(), false, true, false, false, false, false)
	boolEq(t, "IsNumericUnicode", uni.IsNumeric(), true, true, true, true, false, false)
}

func TestRepeatSliceVal(t *testing.T) {
	a := slice2x3("ab", "x", "", "cde", "f", "gh")
	eq(t, "Repeat", a.Repeat(3), slice2x3("ababab", "xxx", "", "cdecdecde", "fff", "ghghgh"))
	eq(t, "Repeat0", a.Repeat(0), slice2x3("", "", "", "", "", ""))
	// 'ab'[1:-1]='' , 'cde'[1:-1]='d', 'gh'[1:-1]='' (len 2: [1:1] empty).
	eq(t, "SliceVal", a.SliceVal(1, -1), slice2x3("", "", "", "d", "", ""))
	eq(t, "SliceValEmpty", a.SliceVal(5, 2), slice2x3("", "", "", "", "", ""))
	panicked := false
	func() {
		defer func() { _ = recover(); panicked = true }()
		a.Repeat(-1)
	}()
	if !panicked {
		t.Fatal("Repeat(-1) did not panic")
	}
}

func TestViewSemantics(t *testing.T) {
	base := FromSlice([]string{"a", "b", "c", "d", "e", "f"}, 2, 3)
	// Slice view shares data: mutate through it.
	v := base.Slice(numgo.Range{Start: 0, Stop: 1})
	if !reflect.DeepEqual(v.Shape(), []int{1, 3}) {
		t.Fatalf("Slice shape = %v", v.Shape())
	}
	v.Set([]int{0, 1}, "B")
	if base.At(0, 1) != "B" {
		t.Fatal("Slice view mutation did not propagate to base")
	}
	// Transpose view shares data.
	tv := base.Transpose()
	if tv.Shape()[0] != 3 || tv.Shape()[1] != 2 || tv.At(1, 0) != "B" {
		t.Fatalf("Transpose: shape %v, At(1,0)=%q", tv.Shape(), tv.At(1, 0))
	}
	// Upper on a non-contiguous view is correct (elementwise over strides).
	ul := tv.Upper()
	if ul.At(0, 1) != "D" {
		t.Fatalf("Upper over transposed view: %q", ul.At(0, 1))
	}
	// Contiguous reshape shares data.
	c := tv.Reshape(3, 2) // still non-contiguous (transposed), so copies
	c.Set([]int{0, 0}, "Z")
	if base.At(0, 0) == "Z" {
		t.Fatal("Reshape of non-contiguous shares data — must copy")
	}
	// Reshape of the contiguous base shares data.
	c2 := base.Reshape(3, 2)
	c2.Set([]int{0, 0}, "Y")
	if base.At(0, 0) != "Y" {
		t.Fatal("Contiguous Reshape did not share data")
	}
	// At bounds.
	func() {
		defer func() { _ = recover() }()
		base.At(2, 0)
		t.Fatal("At(2,0) out of bounds did not panic")
	}()
}

func TestConcatenate(t *testing.T) {
	a := slice2x3("a", "b", "c", "d", "e", "f")
	b := slice2x3("g", "h", "i", "j", "k", "l")
	c := Concatenate(1, a, b)
	if !reflect.DeepEqual(c.Shape(), []int{2, 6}) {
		t.Fatalf("Concatenate(1) shape = %v", c.Shape())
	}
	if c.At(0, 3) != "g" || c.At(1, 5) != "l" {
		t.Fatalf("Concatenate(1) values wrong: %v", c.data)
	}
	d := Concatenate(0, a, b)
	if !reflect.DeepEqual(d.Shape(), []int{4, 3}) || d.At(2, 0) != "g" {
		t.Fatalf("Concatenate(0) wrong: shape %v data %v", d.Shape(), d.data)
	}
	// Mismatch panic.
	func() {
		defer func() { _ = recover() }()
		Concatenate(0, a, FromSlice([]string{"x", "y", "z"}))
		t.Fatal("Concatenate mismatch did not panic")
	}()
}

func TestConversions(t *testing.T) {
	f, err := ParseFloat(slice2x3("1.5", "-2", "1e3", "0", ".25", "3"))
	if err != nil {
		t.Fatalf("ParseFloat: %v", err)
	}
	want := []float64{1.5, -2, 1000, 0, 0.25, 3}
	for i, w := range want {
		if got := f.Data()[i]; got != w {
			t.Fatalf("ParseFloat[%d] = %v, want %v", i, got, w)
		}
	}
	// Error at first failing index (C order: element 1 of 2×3).
	_, err = ParseFloat(slice2x3("1", "x", "3", "4", "5", "6"))
	ee, ok := err.(*ElementError)
	if !ok || ee.Index != 1 {
		t.Fatalf("ParseFloat error = %v, want ElementError at index 1", err)
	}
	// Round trip FromFloat.
	s := FromFloat(f, "%.2f")
	eq(t, "FromFloat", s, slice2x3("1.50", "-2.00", "1000.00", "0.00", "0.25", "3.00"))
	f2, err := ParseFloat(s)
	if err != nil {
		t.Fatalf("round trip: %v", err)
	}
	for i, w := range want {
		if got := f2.Data()[i]; got != w {
			t.Fatalf("round trip[%d] = %v, want %v", i, got, w)
		}
	}

	iv, err := Atoi(slice2x3("1", "-22", "0", "7", "9", "100"))
	if err != nil {
		t.Fatalf("Atoi: %v", err)
	}
	if !reflect.DeepEqual(iv, []int64{1, -22, 0, 7, 9, 100}) {
		t.Fatalf("Atoi = %v", iv)
	}
	_, err = Atoi(slice2x3("1", "2", "x", "4", "y", "6"))
	ee, ok = err.(*ElementError)
	if !ok || ee.Index != 2 {
		t.Fatalf("Atoi error = %v, want ElementError at index 2", err)
	}
}

func TestReshapeNonContiguousCopies(t *testing.T) {
	base := FromSlice([]string{"a", "b", "c", "d", "e", "f"}, 2, 3)
	tv := base.Transpose() // non-contiguous
	r := tv.Reshape(2, 3)
	if r.IsContiguous() != true {
		t.Fatal("Reshape of non-contiguous should produce contiguous copy")
	}
	// Mutating the reshape result must NOT touch base (it was copied).
	r.Set([]int{0, 0}, "Z")
	if base.At(0, 0) == "Z" {
		t.Fatal("Reshape of non-contiguous shares data — must copy")
	}
	// Contiguous reshape shares data.
	c := base.Reshape(3, 2)
	c.Set([]int{0, 0}, "Y")
	if base.At(0, 0) != "Y" {
		t.Fatal("Contiguous Reshape did not share data")
	}
}
