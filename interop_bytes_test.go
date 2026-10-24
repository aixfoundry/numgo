package numgo

import (
	"encoding/binary"
	"math"
)

// binaryLittleEndian appends v's little-endian float32 bytes to b.
func binaryLittleEndian(v float32, b []byte) []byte {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
	return append(b, buf[:]...)
}

// binaryDecodeFloat64 decodes b's first 8 bytes as a little-endian float64.
func binaryDecodeFloat64(b []byte) float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(b))
}
