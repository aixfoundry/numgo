package numgo

import (
	"fmt"
	"strconv"
	"strings"
)

func (x *Array) maxFormatWidth(verb rune, prec int) map[int]int {
	colsWidths := map[int]int{} // 列最大宽度
	for k, v := range x.All() {
		var f string
		if verb == 'v' {
			f = strconv.FormatFloat(v, 'g', prec, 64)
		} else {
			f = strconv.FormatFloat(v, byte(verb), prec, 64)
		}
		if len(f) > colsWidths[k[1]] {
			colsWidths[k[1]] = len(f)
		}
	}
	return colsWidths
}

// Format provides a custom string representation of the Array.
//
// Example:
//
//	fmt.Printf("%v", arr)
func (x *Array) Format(fs fmt.State, verb rune) {
	switch verb {
	case 'v', 'e', 'E', 'f', 'F', 'g', 'G':
	default:
		fmt.Fprintf(fs, "%%!%c(%T=Dims(%d, %d))", verb, x, x.r, x.c)
		return
	}
	width, _ := fs.Width()
	prec, ok := fs.Precision()
	if !ok {
		prec = -1
	}
	colsWidths := x.maxFormatWidth(verb, prec)
	for i := range x.r {
		var el string
		switch {
		case x.r == 1:
			fmt.Fprint(fs, "[")
			el = "]"
		case i == 0:
			fmt.Fprint(fs, "⎡")
			el = "⎤\n"
		case i < x.r-1:
			fmt.Fprint(fs, "⎢")
			el = "⎥\n"
		default:
			fmt.Fprint(fs, "⎣")
			el = "⎦"
		}
		for j := range x.c {
			v := x.At(i, j)
			var f string
			if verb == 'v' {
				f = strconv.FormatFloat(v, 'g', prec, 64)
			} else {
				f = strconv.FormatFloat(v, byte(verb), prec, 64)
			}
			if fs.Flag('-') {
				fs.Write([]byte(f))
				fs.Write([]byte(strings.Repeat(" ", max(width, colsWidths[j])-len(f))))
			} else {
				fs.Write([]byte(strings.Repeat(" ", max(width, colsWidths[j])-len(f))))
				fs.Write([]byte(f))
			}

			if j < x.c-1 {
				fs.Write([]byte(strings.Repeat(" ", 2)))
			}
		}
		fmt.Fprint(fs, el)
	}
}
