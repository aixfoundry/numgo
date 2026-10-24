package random

// BitGenerator is the minimal interface the Generator consumes. This phase
// supports 64/32-bit raw draws only; float paths are derived in the Generator
// layer exactly as numpy's C distributions do. Implementations must preserve
// numpy's next_uint64/next_uint32 buffering contract (pcg64.h:382-396).
type BitGenerator interface {
	Next64() uint64
	Next32() uint32
	// Jumped returns an independent generator (numpy "jumped").
	Jumped() BitGenerator
	// Advance skips delta draws.
	Advance(delta uint64)
}
