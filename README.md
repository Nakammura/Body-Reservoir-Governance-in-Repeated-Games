# Body-Reservoir Governance in Repeated Games

Code, data, and manuscript for:

> **Body-Reservoir Governance in Repeated Games:
> Embodied Decision-Making, Dynamic Sentinel Adaptation,
> and Complexity-Regularized Optimization**
>
> Yuki Nakamura, The Open University of Japan
>
> **Paper:** [arXiv:2602.20846](https://arxiv.org/abs/2602.20846) (cs.GT)
>
> **ORCID:** [0009-0001-7174-6737](https://orcid.org/0009-0001-7174-6737)

## Overview

Why does Tit-for-Tat dominate in theory yet unconditional cooperation
persist in practice? We propose that the answer lies in the
**physical cost of implementing strategies**.

We model game-theoretic agents as echo state networks (reservoir computers)
and define the **variational free energy** of a strategy as prediction error
plus complexity cost (KL divergence between driven and natural dynamics).
A three-layer **Body-Reservoir Governance (BRG)** architecture — body
(reservoir), cognitive filter, and metacognitive mixer — yields:

1. **Cost Structure**: Complexity cost decomposes into spectral and
   noise-floor terms, with unconditional strategies achieving lower cost.
2. **Optimal Receptivity**: Rich reservoirs favor body governance
   (α\* → 1); poor reservoirs favor cognitive control (α\* → 0).
3. **Dynamic Sentinel**: An adaptive α(t) mechanism detects environmental
   shifts and temporarily increases cognitive override, then recovers.
4. **Complexity-Regularized Optimization**: In the (d, τ\_env) parameter
   space, the sentinel advantage erodes gracefully with defection duration,
   with no sharp phase transition.

Ten numerical experiments on a continuous Prisoner's Dilemma confirm
all theoretical predictions using 20-seed Monte Carlo simulations
with zero external dependencies.

## Repository Structure

```
brg-repeated-games/
├── paper/
│   ├── main.tex                 # Manuscript (LaTeX)
│   ├── references.bib           # Bibliography
│   └── fig_*.pdf                # All 9 figures
├── simulation/
│   ├── Cargo.toml               # Rust project (zero dependencies)
│   └── src/
│       ├── esn.rs               # Core ESN library (RNG, linalg, KL)
│       ├── selfconsistent.rs    # Exp 1-2: Fixed point & KL landscape
│       ├── perturbation.rs      # Exp 3: Perturbation response
│       ├── habituation_alpha.rs # Exp 4: Habituation dynamics
│       ├── free_energy.rs       # Exp 5: Free energy landscape
│       ├── dynamic_alpha.rs     # Exp 6-7: Dynamic sentinel & sensitivity
│       ├── dimension_sweep.rs   # Exp 8: Reservoir dimension sweep
│       ├── phase_transition.rs  # Exp 9: (d, τ_env) phase diagram
│       └── ema_baseline.rs      # Exp 10: EMA-filtered TfT baseline
├── results/                     # Simulation output (CSV)
├── scripts/
│   └── generate_figures.py      # Figure generation (matplotlib)
├── LICENSE
└── README.md
```

## Reproducing Results

### Requirements

- **Rust** (stable, 1.70+) for simulations
- **Python 3.8+** with `matplotlib` and `numpy` for figures
- **LaTeX** distribution (e.g., TeX Live) for compiling the paper

### Running Simulations

All simulation code is self-contained with **zero external Rust dependencies**.
Each binary writes CSV data to stdout; redirect to the appropriate file
in `results/`.

```bash
cd simulation

# Exp 1-2: Self-consistent fixed point & KL landscape
cargo run --release --bin selfconsistent > ../results/selfconsistent.csv

# Exp 3: Perturbation response (with dynamic sentinel)
cargo run --release --bin perturbation > ../results/perturbation.csv

# Exp 4: Habituation dynamics (with dynamic sentinel)
cargo run --release --bin habituation_alpha > ../results/habituation_alpha.csv

# Exp 5: Free energy landscape FE(α, H)
cargo run --release --bin free_energy > ../results/free_energy.csv

# Exp 6-7: Dynamic sentinel response & parameter sensitivity
cargo run --release --bin dynamic_alpha > ../results/dynamic_alpha.csv

# Exp 8: Reservoir dimension sweep
cargo run --release --bin dimension_sweep > ../results/dimension_sweep.csv

# Exp 9: Phase transition grid (d × τ_env)
cargo run --release --bin phase_transition > ../results/phase_transition.csv

# Exp 10: EMA-filtered TfT baseline comparison
cargo run --release --bin ema_baseline > ../results/ema_baseline.csv
```

Progress messages are printed to stderr.

### Generating Figures

```bash
pip install matplotlib numpy
python scripts/generate_figures.py        # all 9 figures
python scripts/generate_figures.py 1 3 6  # specific figures only
```

Produces 9 PDF figures in `paper/`.

### Compiling the Paper

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Simulation Parameters

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Reservoir dimension | d | 30 (sweep: 5–100) |
| Spectral radius | ρ | 0.9 |
| Intrinsic noise | σ\_ξ | 0.15 |
| Input weights | W\_in | N(0, 0.5²) |
| Ridge regression | λ\_reg | 0.001 |
| Training samples | n\_train | 2000 |
| Sentinel baseline | α₀ | 0.85 |
| Sentinel threshold | θ | 0.1 |
| Seeds per experiment | — | 20 |

## Citation

```bibtex
@article{nakamura2026brg,
  author  = {Nakamura, Yuki},
  title   = {Body-Reservoir Governance in Repeated Games:
             Embodied Decision-Making, Dynamic Sentinel Adaptation,
             and Complexity-Regularized Optimization},
  year    = {2026},
  journal = {arXiv preprint arXiv:2602.20846},
}
```

## Author

Yuki Nakamura  
ORCID: [0009-0001-7174-6737](https://orcid.org/0009-0001-7174-6737)

## License

MIT License. See [LICENSE](LICENSE).
