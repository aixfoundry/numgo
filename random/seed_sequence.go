package random

// SeedSequence mixes sources of entropy reproducibly to derive BitGenerator
// initial state. Port of numpy/random/bit_generator.pyx SeedSequence
// (O'Neill's seed_seq alternative; pool hashing + generate_state).

const (
	defaultPoolSize        = 4
	initA           uint32 = 0x43b0d7e5
	multA           uint32 = 0x931e8875
	initB           uint32 = 0x8b51f9dd
	multB           uint32 = 0x58f38ded
	mixMultL        uint32 = 0xca01f9dd
	mixMultR        uint32 = 0x4973f715
	xshift                 = 16 // XSHIFT = 32 // 2
	mask32          uint32 = 0xFFFFFFFF
)

// hashmix: bit_generator.pyx:158.
func hashmix(value uint32, hashConst *uint32) uint32 {
	value ^= *hashConst
	*hashConst *= multA
	value *= *hashConst
	value ^= value >> xshift
	return value
}

// mix: bit_generator.pyx:166.
func mix(x, y uint32) uint32 {
	result := mixMultL*x - mixMultR*y
	return result ^ (result >> xshift)
}

// SeedSequence(entropy ...uint64) mirrors numpy.random.SeedSequence with the
// default pool_size=4 and spawn_key from Spawn. Entropy words are split into
// 32-bit chunks, lowest bits first (numpy _int_to_uint32_array).
type SeedSequence struct {
	entropy          []uint64
	spawnKey         []uint32
	poolSize         int
	nChildrenSpawned int
	pool             [defaultPoolSize]uint32
}

// NewSeedSequence builds a SeedSequence from non-negative entropy words.
func NewSeedSequence(entropy ...uint64) *SeedSequence {
	return newSeedSequence(entropy, nil, defaultPoolSize)
}

func newSeedSequence(entropy []uint64, spawnKey []uint32, poolSize int) *SeedSequence {
	if poolSize < defaultPoolSize {
		panicSeedSequenceSize()
	}
	for _, e := range entropy {
		if e>>63 != 0 {
			panicNegativeEntropy()
		}
	}
	ss := &SeedSequence{
		entropy:  entropy,
		spawnKey: spawnKey,
		poolSize: poolSize,
	}
	ss.mixEntropy(ss.assembledEntropy())
	return ss
}

// entropyWordsToU32 breaks each word into 32-bit chunks, lowest first;
// zero contributes a single 0 word (numpy _int_to_uint32_array).
func entropyWordsToU32(words []uint64) []uint32 {
	var out []uint32
	for _, n := range words {
		if n == 0 {
			out = append(out, 0)
			continue
		}
		for n > 0 {
			out = append(out, uint32(n&uint64(mask32)))
			n >>= 32
		}
	}
	return out
}

// assembledEntropy: get_assembled_entropy, bit_generator.pyx:386. When a
// spawn key is present and run entropy is shorter than the pool, zero-fill to
// the pool size first (numpy 1.19 spawn-key disambiguation, gh-16539).
func (ss *SeedSequence) assembledEntropy() []uint32 {
	run := entropyWordsToU32(ss.entropy)
	spawn := append([]uint32(nil), ss.spawnKey...)
	if len(spawn) > 0 && len(run) < ss.poolSize {
		pad := make([]uint32, ss.poolSize-len(run))
		run = append(run, pad...)
	}
	return append(run, spawn...)
}

// mixEntropy: bit_generator.pyx:351.
func (ss *SeedSequence) mixEntropy(entropyArray []uint32) {
	hashConst := initA
	for i := 0; i < ss.poolSize; i++ {
		if i < len(entropyArray) {
			ss.pool[i] = hashmix(entropyArray[i], &hashConst)
		} else {
			ss.pool[i] = hashmix(0, &hashConst)
		}
	}
	for iSrc := 0; iSrc < ss.poolSize; iSrc++ {
		for iDst := 0; iDst < ss.poolSize; iDst++ {
			if iSrc != iDst {
				ss.pool[iDst] = mix(ss.pool[iDst], hashmix(ss.pool[iSrc], &hashConst))
			}
		}
	}
	for iSrc := ss.poolSize; iSrc < len(entropyArray); iSrc++ {
		for iDst := 0; iDst < ss.poolSize; iDst++ {
			ss.pool[iDst] = mix(ss.pool[iDst], hashmix(entropyArray[iSrc], &hashConst))
		}
	}
}

// GenerateState returns nWords uint32 words for PRNG seeding
// (bit_generator.pyx:414, uint32 path).
func (ss *SeedSequence) GenerateState(nWords int) []uint32 {
	hashConst := initB
	state := make([]uint32, nWords)
	src := 0
	for i := 0; i < nWords; i++ {
		dataVal := ss.pool[src%ss.poolSize]
		src++
		dataVal ^= hashConst
		hashConst *= multB
		dataVal *= hashConst
		dataVal ^= dataVal >> xshift
		state[i] = dataVal
	}
	return state
}

// GenerateState64 returns nWords uint64 words: generate_state(n, uint64) is
// 2*n uint32 words viewed as little-endian uint64 pairs.
func (ss *SeedSequence) GenerateState64(nWords int) []uint64 {
	w := ss.GenerateState(2 * nWords)
	out := make([]uint64, nWords)
	for i := range out {
		out[i] = uint64(w[2*i]) | uint64(w[2*i+1])<<32
	}
	return out
}

// Spawn returns n child SeedSequences by extending the spawn key
// (bit_generator.pyx:462).
func (ss *SeedSequence) Spawn(n int) []*SeedSequence {
	if n < 0 {
		panicChildren()
	}
	seqs := make([]*SeedSequence, n)
	for i := 0; i < n; i++ {
		key := append(append([]uint32(nil), ss.spawnKey...), uint32(ss.nChildrenSpawned+i))
		seqs[i] = newSeedSequence(ss.entropy, key, ss.poolSize)
	}
	ss.nChildrenSpawned += n
	return seqs
}

// newSeedSequenceU32 builds from raw 32-bit entropy words (numpy accepts
// big ints split to u32 directly; our uint64 API loses interior zero words).
func newSeedSequenceU32(words []uint32, spawnKey []uint32, poolSize int) *SeedSequence {
	ss := &SeedSequence{spawnKey: spawnKey, poolSize: poolSize}
	ss.mixEntropy(append(append([]uint32(nil), words...), spawnKey...))
	return ss
}
