# Contributing

## Setup

- Go 1.27 or later; no other toolchain requirements (the module has zero third-party dependencies).
- `git clone` and you're ready — `go test ./...` needs nothing beyond the standard library.

## Building and testing

Run tests in both build modes; a PR is only done when both are green:

```
go test ./...
GOEXPERIMENT=simd go test ./...
```

The two builds can differ in the last ULP on float64 paths (FMA/lane-ordered accumulation under `GOEXPERIMENT=simd`); see [docs/architecture.md](docs/architecture.md) for the contracts that must hold in each. Race checks target the concurrency-sensitive packages:

```
go test -race ./fft ./internal/nditer ./random
```

(`-race` needs cgo for the TSan runtime; CI enables it only for this step.)

## Benchmarks

Before/after claims for performance changes should come from the same script used for the published numbers:

```
sh scripts/bench_compare.sh
```

It runs the Go benchmarks (`-count 3`, report the median) and the NumPy side (`scripts/bench_numpy.py`) for comparison. Don't chase noise: single-run improvements under ~5% are usually not real.

## Commits and pull requests

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) as in the existing history (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `chore:`).
- Keep PRs focused; include tests for behavior changes — non-square shapes catch most indexing bugs.
- PR checks: `gofmt`/`go vet` clean, tests green in both build modes, race checks passing.
- New operations go through `internal/nditer`; don't hand-write index algebra in operation code (see [docs/architecture.md](docs/architecture.md)).
- Panic messages come from `errors.go` and keep NumPy wording.
