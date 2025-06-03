package numgo

import (
	"fmt"
	"strconv"
)

func (x *Array) format(prefix string, margin int, dot byte, squeeze bool, fs fmt.State, c rune) {
	rows, cols := x.r, x.c

	printed := margin
	if margin <= 0 {
		printed = max(cols, rows)
	}

	prec, ok := fs.Precision()
	if !ok {
		prec = -1
	}

	var (
		maxWidth int
		widths   widther
		buf, pad []byte
	)
	if squeeze {
		widths = make(columnWidth, cols)
	} else {
		widths = new(uniformWidth)
	}
	switch c {
	case 'v':
		buf, maxWidth = x.maxCellWidth('g', printed, prec, widths)
	case 'e', 'E', 'f', 'F', 'g', 'G':
		buf, maxWidth = x.maxCellWidth(c, printed, prec, widths)
	default:
		fmt.Fprintf(fs, "%%!%c(%T=Dims(%d, %d))", c, x, rows, cols)
		return
	}
	width, _ := fs.Width()
	width = max(width, maxWidth)
	pad = make([]byte, max(width, 2))
	for i := range pad {
		pad[i] = ' '
	}

	first := true
	if rows > 2*printed || cols > 2*printed {
		first = false
		fmt.Fprintf(fs, "Dims(%d, %d)\n", rows, cols)
	}

	skipZero := fs.Flag(' ')
	for i := 0; i < rows; i++ {
		if !first {
			fmt.Fprint(fs, prefix)
		}
		first = false
		var el string
		switch {
		case rows == 1:
			fmt.Fprint(fs, "[")
			el = "]"
		case i == 0:
			fmt.Fprint(fs, "⎡")
			el = "⎤\n"
		case i < rows-1:
			fmt.Fprint(fs, "⎢")
			el = "⎥\n"
		default:
			fmt.Fprint(fs, "⎣")
			el = "⎦"
		}

		for j := 0; j < cols; j++ {
			if j >= printed && j < cols-printed {
				j = cols - printed - 1
				if i == 0 || i == rows-1 {
					fmt.Fprint(fs, "...  ...  ")
				} else {
					fmt.Fprint(fs, "          ")
				}
				continue
			}

			v := x.data[i*x.c+j]
			if v == 0 && skipZero {
				buf = buf[:1]
				buf[0] = dot
			} else {
				if c == 'v' {
					buf = strconv.AppendFloat(buf[:0], v, 'g', prec, 64)
				} else {
					buf = strconv.AppendFloat(buf[:0], v, byte(c), prec, 64)
				}
			}
			padWidth := widths.width(j) - len(buf)
			if fs.Flag('-') {
				fs.Write(buf)
				fs.Write(pad[:padWidth])
			} else {
				fs.Write(pad[:padWidth])
				fs.Write(buf)
			}

			if j < cols-1 {
				fs.Write(pad[:2])
			}
		}

		fmt.Fprint(fs, el)

		if i >= printed-1 && i < rows-printed && 2*printed < rows {
			i = rows - printed - 1
			fmt.Fprintf(fs, "%s .\n%[1]s .\n%[1]s .\n", prefix)
			continue
		}
	}
}

func (x *Array) maxCellWidth(c rune, printed, prec int, w widther) ([]byte, int) {
	var (
		buf        = make([]byte, 0, 64)
		rows, cols = x.r, x.c
		max        int
	)
	for i := 0; i < rows; i++ {
		if i >= printed-1 && i < rows-printed && 2*printed < rows {
			i = rows - printed - 1
			continue
		}
		for j := range cols {
			if j >= printed && j < cols-printed {
				continue
			}

			buf = strconv.AppendFloat(buf, x.At(i, j), byte(c), prec, 64)
			if len(buf) > max {
				max = len(buf)
			}
			if len(buf) > w.width(j) {
				w.setWidth(j, len(buf))
			}
			buf = buf[:0]
		}
	}
	return buf, max
}

type widther interface {
	width(i int) int
	setWidth(i, w int)
}

type uniformWidth int

func (u *uniformWidth) width(_ int) int   { return int(*u) }
func (u *uniformWidth) setWidth(_, w int) { *u = uniformWidth(w) }

type columnWidth []int

func (c columnWidth) width(i int) int   { return c[i] }
func (c columnWidth) setWidth(i, w int) { c[i] = w }
