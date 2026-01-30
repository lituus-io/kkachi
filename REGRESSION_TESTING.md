# Performance Regression Testing Guide

This guide explains how to use the benchmark infrastructure to detect performance regressions in the kkachi library.

## Quick Start

### Establish a Baseline

Before making changes, save the current performance as a baseline:

```bash
# Run all benchmarks and save as baseline
cargo bench -- --save-baseline main

# Or save specific benchmark suite
cargo bench --bench cache_benchmark -- --save-baseline main
cargo bench --bench memory_benchmark -- --save-baseline main
```

### Make Changes

After making code changes, compare against the baseline:

```bash
# Compare current performance to baseline
cargo bench -- --baseline main

# This will show percentage changes, e.g.:
# cache_hit/10    time:   [101.47 ns 101.87 ns 102.27 ns]
#                 change: [-2.5% +0.3% +3.1%] (p = 0.23 > 0.05)
```

## Interpreting Results

### Change Indicators

Criterion shows three values for change percentage:
- **Lower bound**: -2.5% (best case)
- **Point estimate**: +0.3% (most likely)
- **Upper bound**: +3.1% (worst case)

### Statistical Significance

- **p < 0.05**: Change is statistically significant (likely a real change)
- **p ≥ 0.05**: Change may be noise (not statistically significant)

### Regression Thresholds

Use these thresholds to determine if a change is acceptable:

| Change Magnitude | Assessment | Action |
|------------------|------------|--------|
| < 5% slower | Acceptable | Proceed |
| 5-20% slower | Warning | Review code, consider if trade-off is worth it |
| > 20% slower | Regression | Investigate and fix before merging |
| > 10% faster | Improvement | Document the optimization! |

## Example Workflow

### 1. Save Baseline Before Refactoring

```bash
# You're on main branch, about to refactor
cargo bench -- --save-baseline before-refactor
```

### 2. Make Changes

```bash
# Create feature branch
git checkout -b feature/my-optimization

# Make code changes
# ...edit files...
```

### 3. Compare Performance

```bash
# Run benchmarks and compare
cargo bench -- --baseline before-refactor
```

Example output:
```
cache_hit/10            time:   [95.234 ns 96.123 ns 97.045 ns]
                        change: [-8.1234% -5.6435% -3.2156%] (p = 0.00 < 0.05)
                        Performance has improved.

memory_search/10        time:   [7.1234 µs 7.2345 µs 7.3456 µs]
                        change: [+2.1234% +3.8765% +5.6234%] (p = 0.03 < 0.05)
                        Performance has regressed.
```

### 4. Investigate Regressions

If you see significant regressions:

```bash
# Run specific benchmark with more samples for accuracy
cargo bench --bench memory_benchmark -- --baseline before-refactor --sample-size 200

# Profile the code to find bottlenecks
cargo flamegraph --bench memory_benchmark
```

## Continuous Integration

### GitHub Actions Example

Add this to `.github/workflows/benchmark.yml`:

```yaml
name: Performance Benchmarks

on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Need full history for baseline comparison

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          profile: minimal

      - name: Checkout baseline (main)
        run: |
          git checkout main
          cargo bench --bench cache_benchmark -- --save-baseline main
          cargo bench --bench memory_benchmark -- --save-baseline main

      - name: Checkout PR branch
        run: git checkout ${{ github.head_ref }}

      - name: Run benchmarks and compare
        run: |
          cargo bench --bench cache_benchmark -- --baseline main
          cargo bench --bench memory_benchmark -- --baseline main

      - name: Check for regressions
        run: |
          # Parse criterion output and fail if >20% regression detected
          # (Add custom script here)
```

## Benchmark Organization

### Current Benchmark Suites

1. **cache_benchmark** (`benches/cache_benchmark.rs`)
   - Cache hit/miss performance
   - Cache eviction
   - Cache capacity impact
   - **Key metrics**: cache_hit/*, cache_miss/*, cache_eviction/*

2. **memory_benchmark** (`benches/memory_benchmark.rs`)
   - Document add/get/update
   - Search performance (k=1-20)
   - Search scaling (n=10-1000)
   - Recall construction (Arc optimization)
   - **Key metrics**: memory_search/*, memory_scaling/*, recall_construction/*

3. **llm_benchmarks** (`crates/kkachi/benches/llm_benchmarks.rs`)
   - High-level API benchmarks (refine, best_of, etc.)
   - **Key metrics**: refine_*, best_of_*, ensemble_*

### Adding New Benchmarks

When adding new benchmarks, follow this structure:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn my_benchmark(c: &mut Criterion) {
    c.bench_function("my_operation", |b| {
        b.iter(|| {
            // Code to benchmark
            black_box(my_operation())
        });
    });
}

criterion_group!(benches, my_benchmark);
criterion_main!(benches);
```

## Baseline Management

### List Saved Baselines

```bash
ls -la target/criterion/*/base/
```

### Delete Old Baselines

```bash
# Delete specific baseline
rm -rf target/criterion/*/before-refactor/

# Delete all baselines (keep only latest results)
find target/criterion -name 'base' -type d -exec rm -rf {} +
```

### Export/Import Baselines

```bash
# Export baseline for sharing
tar -czf baseline-v0.3.0.tar.gz target/criterion/*/main/

# Import baseline
tar -xzf baseline-v0.3.0.tar.gz
```

## Performance Monitoring

### Critical Metrics to Monitor

Based on `PERFORMANCE_BASELINE.md`, these are the key metrics:

| Metric | Baseline | Regression Threshold |
|--------|----------|---------------------|
| cache_hit (any size) | ~102ns | > 150ns |
| memory get_by_id | ~27ns | > 50ns |
| memory_search (k=5, n=100) | ~7µs | > 15µs |
| recall_construction (k=10) | ~1.6µs | > 3µs |

### Automated Checks

Create a script to check for regressions:

```bash
#!/bin/bash
# scripts/check-benchmarks.sh

THRESHOLD=20  # 20% regression threshold

# Run benchmarks and save output
cargo bench -- --baseline main 2>&1 | tee /tmp/bench-results.txt

# Check for regressions > threshold
if grep -E "change:.*\+${THRESHOLD}\.[0-9]+%" /tmp/bench-results.txt; then
    echo "ERROR: Performance regression detected (>${THRESHOLD}%)"
    exit 1
else
    echo "OK: No significant performance regressions"
    exit 0
fi
```

## Troubleshooting

### Noisy Results

If benchmark results are unstable:

```bash
# Increase sample size
cargo bench -- --sample-size 200

# Increase warm-up time
cargo bench -- --warm-up-time 5

# Run on isolated CPU
taskset -c 0 cargo bench
```

### Comparing Across Machines

Baselines are machine-specific. To compare across machines:

1. Run on same hardware for fair comparison
2. Use relative changes (%) rather than absolute times
3. Save machine specs with baselines:

```bash
echo "$(uname -a)" > target/criterion/machine-info.txt
echo "$(rustc --version)" >> target/criterion/machine-info.txt
```

## Best Practices

1. **Save baselines before major changes**
   - Before refactoring: `--save-baseline before-refactor`
   - Main branch: `--save-baseline main`

2. **Compare consistently**
   - Always compare feature branches against main baseline
   - Use same hardware and environment settings

3. **Investigate regressions early**
   - Don't wait until PR review to run benchmarks
   - Run locally during development

4. **Document improvements**
   - If benchmarks show >10% improvement, document it
   - Update `PERFORMANCE_BASELINE.md` with new baselines

5. **Run full suite before release**
   - Before tagging a release: `cargo bench -- --save-baseline v0.3.0`
   - Keep release baselines for future reference

## Additional Resources

- **Criterion.rs User Guide**: https://bheisler.github.io/criterion.rs/book/
- **Performance Baseline**: See `PERFORMANCE_BASELINE.md`
- **Optimization Guide**: See `OPTIMIZATION_GUIDE.md`
- **Cargo Bench Docs**: https://doc.rust-lang.org/cargo/commands/cargo-bench.html
