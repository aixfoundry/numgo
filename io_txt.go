// Text IO with missing-value handling (numpy genfromtxt, simplified to
// uniform float columns — no structured dtypes, converters, usecols or
// masking; missing/failed-parse fields become the filling value and jagged
// rows are errors, like LoadTxt).
package numgo

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// GenConfig configures GenFromTxt; zero values match numpy's defaults except
// FillingValues, which defaults to NaN (numpy's fill for float columns).
type GenConfig struct {
	Delimiter      string   // "" splits on any run of whitespace; otherwise a literal separator
	SkipHeader     int      // lines skipped at the start, before names detection
	SkipFooter     int      // data rows dropped from the end
	Comments       []string // lines starting with any entry are skipped (default "#")
	Names          []string // explicit column names to return
	UseHeaderNames bool     // the first non-skipped row supplies the names
	FillingValues  float64  // substituted for missing/failed-parse fields (default NaN)
	Missing        string   // a field equal to this string counts as missing ("" is always missing)
}

// GenFromTxt reads delimited float text with missing-value handling
// (numpy genfromtxt port, float columns only). It returns the data array and
// the column names — nil unless configured via Names or UseHeaderNames. A
// field equal to Missing (or empty) or one that fails float parsing becomes
// FillingValues; after that, a row with a different field count is an error.
func GenFromTxt(r io.Reader, cfg GenConfig) (*Array[float64], []string, error) {
	if len(cfg.Comments) == 0 {
		cfg.Comments = []string{"#"}
	}
	fill := cfg.FillingValues
	// Zero value = unset; an explicit 0 fill cannot be expressed
	// (shortcut: GenConfig has no set-bit — use NaN as the documented default).
	if fill == 0 {
		fill = math.NaN()
	}
	sc := bufio.NewScanner(r)
	var rows [][]float64
	var names []string
	skipped := 0
	for sc.Scan() {
		line := sc.Text()
		if skipped < cfg.SkipHeader {
			skipped++
			continue
		}
		if cfg.UseHeaderNames && names == nil {
			// numpy strips a leading comment marker from the names row
			// instead of dropping the line (_npyio_impl.genfromtxt).
			for _, m := range cfg.Comments {
				if strings.HasPrefix(line, m) {
					line = line[len(m):]
					break
				}
			}
			fields := splitFields(line, cfg.Delimiter)
			if len(fields) == 0 {
				continue
			}
			names = fields
			continue
		}
		if isComment(line, cfg.Comments) {
			continue
		}
		fields := splitFields(line, cfg.Delimiter)
		if len(fields) == 0 {
			continue
		}
		vals := make([]float64, len(fields))
		for i, f := range fields {
			f = strings.TrimSpace(f)
			if f == "" || f == cfg.Missing {
				vals[i] = fill
				continue
			}
			v, err := strconv.ParseFloat(f, 64)
			if err != nil {
				vals[i] = fill
				continue
			}
			vals[i] = v
		}
		rows = append(rows, vals)
	}
	if err := sc.Err(); err != nil {
		return nil, nil, err
	}
	if cfg.SkipFooter > 0 {
		if cfg.SkipFooter > len(rows) {
			return nil, nil, fmt.Errorf("numgo: GenFromTxt: skip_footer %d exceeds %d data rows", cfg.SkipFooter, len(rows))
		}
		rows = rows[:len(rows)-cfg.SkipFooter]
	}
	if len(rows) == 0 {
		return Zeros[float64](0), names, nil
	}
	cols := len(rows[0])
	var vals []float64
	for i, row := range rows {
		if len(row) != cols {
			return nil, nil, fmt.Errorf("numgo: GenFromTxt: inconsistent number of columns: line %d has %d fields, want %d", i+1, len(row), cols)
		}
		vals = append(vals, row...)
	}
	if cfg.Names != nil {
		names = cfg.Names
	}
	return FromSlice(vals, len(rows), cols), names, nil
}

func isComment(line string, markers []string) bool {
	for _, m := range markers {
		if strings.HasPrefix(line, m) {
			return true
		}
	}
	return false
}

// splitFields splits on the delimiter or, for "", on whitespace runs;
// whitespace mode drops empty fields, delimiter mode keeps interior empties
// (so ",," yields a missing field) but drops a trailing one, like numpy.
func splitFields(line, delim string) []string {
	if delim == "" {
		return strings.Fields(line)
	}
	fields := strings.Split(line, delim)
	if n := len(fields); n > 0 && fields[n-1] == "" {
		fields = fields[:n-1]
	}
	return fields
}
