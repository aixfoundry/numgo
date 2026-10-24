package strings

import (
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/aixfoundry/numgo/internal/nditer"
)

// Result-type rule: ops producing strings per element return *StrArray; ops
// producing any other element type (bool, [][]string, numbers) return plain
// slices in C order — the package stores strings only (same precedent as Atoi
// returning []int64).

// forEach walks a's elements in C order through nditer.Zip, calling f once per
// element. The dummy dst gives Zip a contiguous target of the same shape; its
// data slice is re-sliced per element on the strided path, so callbacks arrive
// in C order and concatenate to the full element sequence regardless of block
// size.
func forEach(a *StrArray, f func(string)) {
	if a.Size() == 0 {
		return
	}
	dst := &nditer.Layout{Shape: a.Shape(), Strides: nditer.COrderStrides(a.shape)}
	nditer.Zip(dst, make([]string, a.Size()), []*nditer.Layout{a.layout()}, [][]string{a.data},
		func(d []string, src [][]string) {
			for i := range d {
				f(src[0][i])
			}
		})
}

// elementwise maps f over a's elements in C order into a new compact array
// with the same shape.
func elementwise(a *StrArray, f func(string) string) *StrArray {
	dst := newStrArray(a.Shape(), nditer.COrderStrides(a.shape), 0, make([]string, a.Size()))
	i := 0
	forEach(a, func(v string) { dst.data[i] = f(v); i++ })
	return dst
}

// bools maps f over a's elements into a plain []bool in C order.
func bools(a *StrArray, f func(string) bool) []bool {
	out := make([]bool, a.Size())
	i := 0
	forEach(a, func(v string) { out[i] = f(v); i++ })
	return out
}

// Case ops. Upper/Lower/Swapcase use Go's Unicode mappings, matching numpy's
// per-character semantics (including ß→SS).

// Pure: Upper uppercases every element. Go's unicode simple case mapping keeps
// ß as ß, while numpy (full case mapping) yields "SS" — documented divergence;
// fixing it would need a special-case table, out of scope for the core.
func (a *StrArray) Upper() *StrArray { return elementwise(a, strings.ToUpper) }

// Pure: Lower lowercases every element.
func (a *StrArray) Lower() *StrArray { return elementwise(a, strings.ToLower) }

// Pure: Swapcase swaps the case of every character.
func (a *StrArray) Swapcase() *StrArray {
	return elementwise(a, func(s string) string {
		return strings.Map(func(r rune) rune {
			if unicode.IsUpper(r) {
				return unicode.ToLower(r)
			}
			if unicode.IsLower(r) {
				return unicode.ToUpper(r)
			}
			return r
		}, s)
	})
}

// Pure: Capitalize uppercases the first character and lowercases the rest of
// each element (numpy semantics).
func (a *StrArray) Capitalize() *StrArray {
	return elementwise(a, func(s string) string {
		r := []rune(s)
		if len(r) == 0 {
			return s
		}
		out := make([]rune, len(r))
		out[0] = unicode.ToUpper(r[0])
		for i := 1; i < len(r); i++ {
			out[i] = unicode.ToLower(r[i])
		}
		return string(out)
	})
}

// Pure: Title uppercases the first character of each word and lowercases the
// rest, matching numpy.strings.title: a character starts a word when the
// preceding character is uncased, so "don't do" → "Don'T Do". This equals Go's
// deprecated strings.Title and diverges from modern-English intuition; numpy
// wording is authoritative.
func (a *StrArray) Title() *StrArray {
	return elementwise(a, func(s string) string {
		prev := ' '
		r := []rune(s)
		out := make([]rune, len(r))
		for i, c := range r {
			if !unicode.IsLower(prev) && !unicode.IsUpper(prev) && !unicode.IsTitle(prev) {
				out[i] = unicode.ToUpper(c)
			} else {
				out[i] = unicode.ToLower(c)
			}
			prev = c
		}
		return string(out)
	})
}

// Search ops. Bounds follow Python slice semantics (negative counts from the
// end, clamped to [0, len]) but index BYTES, Go's convention — numpy indexes
// characters. The two agree on ASCII input.

// clampStart resolves a Python-style start bound: negative counts from the
// end, clamped to [0, len(s)].
func clampStart(s string, start int) int {
	n := len(s)
	if start < 0 {
		start += n
	}
	return max(min(start, n), 0)
}

// pythonStop resolves a Python-style exclusive stop bound: negative counts
// from the end, clamped to [0, len(s)].
func pythonStop(s string, stop int) int {
	n := len(s)
	if stop < 0 {
		stop += n
	}
	return max(min(stop, n), 0)
}

// Pure: Find returns the lowest index of sub at or after start in each element,
// or -1 when absent.
func (a *StrArray) Find(sub string, start int) *StrArray {
	return elementwise(a, func(s string) string {
		lo := clampStart(s, start)
		if i := strings.Index(s[lo:], sub); i >= 0 {
			return strconv.Itoa(i + lo)
		}
		return "-1"
	})
}

// Pure: RFind returns the highest index of sub before stop in each element, or
// -1 when absent. A stop of -1 searches the whole element.
func (a *StrArray) RFind(sub string, stop int) *StrArray {
	return elementwise(a, func(s string) string {
		hi := stop
		if hi < 0 {
			hi = len(s)
		}
		hi = max(min(hi, len(s)), 0)
		return strconv.Itoa(strings.LastIndex(s[:hi], sub))
	})
}

// Pure: Index is Find but panics "substring not found" (numpy ValueError
// wording) if any element lacks sub.
func (a *StrArray) Index(sub string, start int) *StrArray {
	res := a.Find(sub, start)
	for _, v := range res.data {
		if v == "-1" {
			panicNotFound()
		}
	}
	return res
}

// Pure: RIndex is RFind but panics "substring not found" if any element lacks
// sub.
func (a *StrArray) RIndex(sub string, stop int) *StrArray {
	res := a.RFind(sub, stop)
	for _, v := range res.data {
		if v == "-1" {
			panicNotFound()
		}
	}
	return res
}

// Pure: Count returns the number of non-overlapping occurrences of sub between
// start and stop in each element. Bounds are Python-style exclusive-stop when
// stop >= 0; a negative stop counts from the end (stop = -1 → whole element
// via the n+1 wrap matching Python's [start:-1] convention).
func (a *StrArray) Count(sub string, start, stop int) *StrArray {
	return elementwise(a, func(s string) string {
		lo := clampStart(s, start)
		hi := pythonStop(s, stop)
		return strconv.Itoa(strings.Count(s[lo:hi], sub))
	})
}

// Pure: StartsWith reports whether each element starts with prefix from start.
func (a *StrArray) StartsWith(prefix string, start int) []bool {
	return bools(a, func(s string) bool {
		return strings.HasPrefix(s[clampStart(s, start):], prefix)
	})
}

// Pure: EndsWith reports whether each element ends with suffix before stop.
// A stop of -1 means the whole element.
func (a *StrArray) EndsWith(suffix string, stop int) []bool {
	return bools(a, func(s string) bool {
		hi := stop
		if hi < 0 {
			hi = len(s)
		}
		hi = max(min(hi, len(s)), 0)
		return strings.HasSuffix(s[:hi], suffix)
	})
}

// Pure: Contains reports whether sub occurs in each element. A Go idiom numpy
// lacks (numpy composes it from find >= 0); documented addition.
func (a *StrArray) Contains(sub string) []bool {
	return bools(a, func(s string) bool { return strings.Contains(s, sub) })
}

// Trim ops. With an empty chars argument, whitespace is trimmed (numpy's
// default variants behave the same).

// strip implements Strip/LStrip/RStrip: cutset-rune trimming, or whitespace
// when chars is empty.
func strip(a *StrArray, chars string, left, right bool) *StrArray {
	var set map[rune]bool
	useSet := chars != ""
	if useSet {
		set = map[rune]bool{}
		for _, r := range chars {
			set[r] = true
		}
	}
	pred := func(r rune) bool {
		if useSet {
			return set[r]
		}
		return unicode.IsSpace(r)
	}
	return elementwise(a, func(s string) string {
		if left {
			s = strings.TrimLeftFunc(s, pred)
		}
		if right {
			s = strings.TrimRightFunc(s, pred)
		}
		return s
	})
}

// Pure: Strip removes leading and trailing chars (whitespace if empty).
func (a *StrArray) Strip(chars string) *StrArray { return strip(a, chars, true, true) }

// Pure: LStrip removes leading chars (whitespace if empty).
func (a *StrArray) LStrip(chars string) *StrArray { return strip(a, chars, true, false) }

// Pure: RStrip removes trailing chars (whitespace if empty).
func (a *StrArray) RStrip(chars string) *StrArray { return strip(a, chars, false, true) }

// Split ops.

// Pure: Split splits each element at sep into a [][]string in C order. numpy
// strings removed split from its namespace (numpy.char keeps it); provided for
// chararray parity.
func (a *StrArray) Split(sep string) [][]string {
	out := make([][]string, 0, a.Size())
	forEach(a, func(v string) { out = append(out, strings.Split(v, sep)) })
	return out
}

// Pure: SplitLines splits each element at line boundaries (\n, \r\n, \r) into
// a [][]string in C order, dropping the trailing empty piece a final newline
// would produce (Python str.splitlines semantics).
func (a *StrArray) SplitLines() [][]string {
	out := make([][]string, 0, a.Size())
	forEach(a, func(v string) {
		norm := strings.ReplaceAll(strings.ReplaceAll(v, "\r\n", "\n"), "\r", "\n")
		parts := strings.Split(norm, "\n")
		if len(parts) > 1 && parts[len(parts)-1] == "" {
			parts = parts[:len(parts)-1]
		}
		out = append(out, parts)
	})
	return out
}

// Pure: Join joins parts with each element as the separator (numpy.char.join
// semantics; numpy.strings removed join).
func (a *StrArray) Join(parts []string) *StrArray {
	return elementwise(a, func(sep string) string { return strings.Join(parts, sep) })
}

// Pure: Partition splits each element at the first sep into a
// []string{head, sep, tail}; a missing sep yields {elem, "", ""} (numpy
// semantics).
func (a *StrArray) Partition(sep string) [][]string {
	return partitionImpl(a, sep, false)
}

// Pure: RPartition splits each element at the last sep into
// []string{head, sep, tail}; a missing sep yields {"", "", elem}.
func (a *StrArray) RPartition(sep string) [][]string {
	return partitionImpl(a, sep, true)
}

func partitionImpl(a *StrArray, sep string, fromRight bool) [][]string {
	out := make([][]string, 0, a.Size())
	forEach(a, func(v string) {
		var head, found, tail string
		if fromRight {
			if i := strings.LastIndex(v, sep); i >= 0 {
				head, found, tail = v[:i], sep, v[i+len(sep):]
			} else {
				tail = v
			}
		} else if i := strings.Index(v, sep); i >= 0 {
			head, found, tail = v[:i], sep, v[i+len(sep):]
		} else {
			head = v
		}
		out = append(out, []string{head, found, tail})
	})
	return out
}

// Replace and friends.

// Pure: Replace replaces up to n occurrences of old with new in each element
// (n < 0 means all).
func (a *StrArray) Replace(old, new string, n int) *StrArray {
	return elementwise(a, func(s string) string { return strings.Replace(s, old, new, n) })
}

// Pure: Translate maps runes through the table (absent runes pass through).
// numpy's delete parameter has no map representation — delete via Replace
// with "". Documented simplification.
func (a *StrArray) Translate(table map[rune]rune) *StrArray {
	return elementwise(a, func(s string) string {
		return strings.Map(func(r rune) rune {
			if mapped, ok := table[r]; ok {
				return mapped
			}
			return r
		}, s)
	})
}

// Pure: ExpandTabs replaces each tab with spaces to the next tab stop of
// tabWidth runes (columns reset at newlines); tabWidth <= 0 removes tabs.
func (a *StrArray) ExpandTabs(tabWidth int) *StrArray {
	return elementwise(a, func(s string) string {
		if tabWidth <= 0 {
			return strings.ReplaceAll(s, "\t", "")
		}
		var b strings.Builder
		col := 0
		for _, r := range s {
			switch r {
			case '\t':
				n := tabWidth - col%tabWidth
				b.WriteString(strings.Repeat(" ", n))
				col += n
			case '\n', '\r':
				b.WriteRune(r)
				col = 0
			default:
				b.WriteRune(r)
				col++
			}
		}
		return b.String()
	})
}

// Pad ops. All four take the total target width in runes; shorter elements are
// padded, longer and non-positive widths pass through unchanged (numpy
// behavior).

// Pure: Center pads each element with spaces on both sides to width; the extra
// space goes to the right (CPython: left = deficit/2, rounded down).
func (a *StrArray) Center(width int) *StrArray {
	return pad(a, width, func(s string, n int) string {
		left := n / 2
		return strings.Repeat(" ", left) + s + strings.Repeat(" ", n-left)
	})
}

// Pure: LJust left-justifies each element in a field of width (pads right).
func (a *StrArray) LJust(width int) *StrArray {
	return pad(a, width, func(s string, n int) string { return s + strings.Repeat(" ", n) })
}

// Pure: RJust right-justifies each element in a field of width (pads left).
func (a *StrArray) RJust(width int) *StrArray {
	return pad(a, width, func(s string, n int) string { return strings.Repeat(" ", n) + s })
}

// Pure: ZFill pads each element with leading zeros to width, keeping a leading
// '+'/'-' sign ahead of the zeros.
func (a *StrArray) ZFill(width int) *StrArray {
	return pad(a, width, func(s string, n int) string {
		if strings.HasPrefix(s, "+") || strings.HasPrefix(s, "-") {
			return s[:1] + strings.Repeat("0", n) + s[1:]
		}
		return strings.Repeat("0", n) + s
	})
}

// pad applies f with the padding deficit only where the element is shorter
// than width.
func pad(a *StrArray, width int, f func(s string, n int) string) *StrArray {
	return elementwise(a, func(s string) string {
		n := width - utf8.RuneCountInString(s)
		if n <= 0 {
			return s
		}
		return f(s, n)
	})
}

// Pure: Repeat repeats each element count times (numpy.strings.multiply).
// Negative counts panic (numpy ValueError).
func (a *StrArray) Repeat(count int) *StrArray {
	if count < 0 {
		panicNegCount()
	}
	return elementwise(a, func(s string) string { return strings.Repeat(s, count) })
}

// Pure: SliceVal slices each element by Python-style [start:stop] byte bounds;
// negative values count from the end. Named SliceVal because StrArray.Slice is
// the view op (numpy.strings.slice).
func (a *StrArray) SliceVal(start, stop int) *StrArray {
	return elementwise(a, func(s string) string {
		lo := clampStart(s, start)
		hi := pythonStop(s, stop)
		if hi < lo {
			return ""
		}
		return s[lo:hi]
	})
}

// Predicates. Empty elements are false except for IsEmpty. IsDigit/IsDecimal
// both use Unicode's Nd class: numpy's isdigit additionally accepts
// digit-like No characters (²) — documented divergence. IsNumeric covers all
// Unicode numbers (N), matching Python str.isnumeric.

// allRunes reports whether pred holds for every rune (empty → false).
func allRunes(s string, pred func(rune) bool) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !pred(r) {
			return false
		}
	}
	return true
}

// Pure: IsAlpha reports whether every character is alphabetic.
func (a *StrArray) IsAlpha() []bool {
	return bools(a, func(s string) bool { return allRunes(s, unicode.IsLetter) })
}

// Pure: IsDigit reports whether every character is a Unicode decimal digit.
func (a *StrArray) IsDigit() []bool {
	return bools(a, func(s string) bool { return allRunes(s, unicode.IsDigit) })
}

// Pure: IsDecimal reports whether every character is a Unicode decimal digit
// (identical to IsDigit here; Python separates them only for non-Nd digits).
func (a *StrArray) IsDecimal() []bool {
	return bools(a, func(s string) bool { return allRunes(s, unicode.IsDigit) })
}

// Pure: IsNumeric reports whether every character is numeric (digits,
// fractions, Roman numerals — Python str.isnumeric semantics).
func (a *StrArray) IsNumeric() []bool {
	return bools(a, func(s string) bool { return allRunes(s, unicode.IsNumber) })
}

// Pure: IsSpace reports whether every character is whitespace.
func (a *StrArray) IsSpace() []bool {
	return bools(a, func(s string) bool { return allRunes(s, unicode.IsSpace) })
}

// Pure: IsAlNum reports whether every character is alphanumeric.
func (a *StrArray) IsAlNum() []bool {
	return bools(a, func(s string) bool {
		return allRunes(s, func(r rune) bool { return unicode.IsLetter(r) || unicode.IsNumber(r) })
	})
}

// casedCheck implements Python's str.islower/isupper: every cased rune matches
// pred and at least one cased rune exists.
func casedCheck(s string, pred func(rune) bool) bool {
	any := false
	for _, r := range s {
		if unicode.IsUpper(r) || unicode.IsLower(r) || unicode.IsTitle(r) {
			if !pred(r) {
				return false
			}
			any = true
		}
	}
	return any
}

// Pure: IsLower reports whether all cased characters are lowercase and at
// least one exists.
func (a *StrArray) IsLower() []bool {
	return bools(a, func(s string) bool { return casedCheck(s, unicode.IsLower) })
}

// Pure: IsUpper reports whether all cased characters are uppercase and at
// least one exists.
func (a *StrArray) IsUpper() []bool {
	return bools(a, func(s string) bool { return casedCheck(s, unicode.IsUpper) })
}

// Pure: IsTitle reports whether each element is title-cased: cased characters
// follow uncased ones and vice versa (Python str.istitle; empty → false).
func (a *StrArray) IsTitle() []bool {
	return bools(a, func(s string) bool {
		prevCased := false
		any := false
		for _, r := range s {
			if unicode.IsUpper(r) || unicode.IsLower(r) || unicode.IsTitle(r) {
				if unicode.IsUpper(r) == prevCased {
					return false
				}
				any = true
			}
			prevCased = unicode.IsUpper(r) || unicode.IsLower(r) || unicode.IsTitle(r)
		}
		return any
	})
}

// Pure: IsEmpty reports whether each element is the empty string. A Go idiom
// (Python has no str.isempty); documented addition.
func (a *StrArray) IsEmpty() []bool {
	return bools(a, func(s string) bool { return s == "" })
}
