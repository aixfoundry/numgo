// Package numgo is a numpy-compatible N-dimensional array library for
// float32/float64/complex64/complex128 element types. Binary IO covers .npy
// streams and .npz zip archives; PackBits/UnpackBits are pending integer
// dtype support.
package numgo

import (
	"archive/zip"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// npz IO (numpy savez/savez_compressed/np.load): a zip archive with one
// plain .npy stream per array. The member name is the mapping key + ".npy"
// on write; on read the suffix is stripped, like np.load's NpzFile.files.
// Unlike the npy API, npz failures are errors, not panics.

// SaveZ writes arrays as an uncompressed .npz zip archive (numpy savez): one
// "name.npy" member per map entry, each a plain .npy stream written by Save.
// Member order follows Go's randomized map order; numpy tolerates any order.
func SaveZ(w io.Writer, arrays map[string]AnyArray) error {
	return saveZ(w, false, arrays)
}

// SaveZCompressed writes arrays as a deflate-compressed .npz archive
// (numpy savez_compressed).
func SaveZCompressed(w io.Writer, arrays map[string]AnyArray) error {
	return saveZ(w, true, arrays)
}

// SaveZFile writes arrays to path as .npz, appending the suffix when missing
// (numpy savez); compressed selects savez_compressed over savez.
func SaveZFile(path string, compressed bool, arrays map[string]AnyArray) error {
	if !strings.HasSuffix(path, ".npz") {
		path += ".npz"
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := saveZ(f, compressed, arrays); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

func saveZ(w io.Writer, compressed bool, arrays map[string]AnyArray) error {
	method := zip.Store
	if compressed {
		method = zip.Deflate
	}
	zw := zip.NewWriter(w)
	for name, a := range arrays {
		// numpy appends .npy to the key unconditionally (_savez).
		entry, err := zw.CreateHeader(&zip.FileHeader{Name: name + ".npy", Method: method})
		if err != nil {
			return fmt.Errorf("numgo: SaveZ: %s: %w", name, err)
		}
		if err := npyWriteAny(entry, a); err != nil {
			return fmt.Errorf("numgo: SaveZ: %s: %w", name, err)
		}
	}
	return zw.Close()
}

// npyWriteAny writes the typed field of a as a .npy stream through Save; an
// AnyArray with no typed field set is an error, not a panic.
func npyWriteAny(w io.Writer, a AnyArray) error {
	switch {
	case a.F64 != nil:
		return Save(w, a.F64)
	case a.F32 != nil:
		return Save(w, a.F32)
	case a.C128 != nil:
		return Save(w, a.C128)
	case a.C64 != nil:
		return Save(w, a.C64)
	}
	return errors.New("numgo: SaveZ: array has no typed field set")
}

// AnyOf builds the tagged AnyArray for a so arrays of different dtypes can be
// mixed in one SaveZ map. Descr is the little-endian descriptor Save writes.
func AnyOf[T Number](a *Array[T]) AnyArray {
	if a == nil {
		panicNilArray("AnyOf")
	}
	out := AnyArray{Shape: a.Shape(), Descr: npyDescrFor[T]()}
	switch p := any(a).(type) {
	case *Array[float64]:
		out.F64 = p
	case *Array[float32]:
		out.F32 = p
	case *Array[complex128]:
		out.C128 = p
	case *Array[complex64]:
		out.C64 = p
	}
	return out
}

// LoadNpz reads every .npy member of an npz zip archive, keyed by member name
// with the .npy suffix stripped (np.load's NpzFile mapping). Non-.npy members
// are ignored; an archive without any .npy member is an error.
func LoadNpz(r io.Reader) (map[string]AnyArray, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	zr, err := zip.NewReader(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		return nil, fmt.Errorf("numgo: LoadNpz: %w", err)
	}
	out := make(map[string]AnyArray, len(zr.File))
	for _, f := range zr.File {
		if !strings.HasSuffix(f.Name, ".npy") {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("numgo: LoadNpz: %s: %w", f.Name, err)
		}
		a, err := Load(rc)
		_ = rc.Close()
		if err != nil {
			return nil, fmt.Errorf("numgo: LoadNpz: %s: %w", f.Name, err)
		}
		out[strings.TrimSuffix(f.Name, ".npy")] = a
	}
	if len(out) == 0 {
		return nil, errors.New("numgo: LoadNpz: archive contains no .npy members")
	}
	return out, nil
}

// LoadNpzFile reads path as an .npz archive.
func LoadNpzFile(path string) (map[string]AnyArray, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return LoadNpz(f)
}

// errNpz is returned by LoadFile for zip-magic files: archives need LoadNpzFile.
var errNpz = errors.New("numgo: npz archive: use LoadNpzFile to load .npz files")
