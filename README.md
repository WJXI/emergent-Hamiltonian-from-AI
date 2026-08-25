# Emergent Hamiltonian from a diffusion model

This release is aligned with the revised manuscript. It contains the frozen
models, all processed arrays used in the reported evaluations, and the code
needed to regenerate every numerical result in the paper.

The two central changes relative to the original repository are:

1. the spin-glass Hamiltonian is read directly from the frozen model's internal
   pair matrix `W`; no OLS score-to-coupling reconstruction is used;
2. the molecular extension is physical 13-atom phenol in explicit water and
   includes Networks 1, 2, 3, and the sector-preserving constant projection
   Network 3-P. In this release, Network 3-P always means the blockwise
   projection defined in the paper.

No shared-law/residual decomposition is used.

## Contents

```text
code/
├── paper_results.json          exact values behind the rounded paper numbers
├── verify_release.py           fast integrity and number audit
├── environment.yml
├── spin_glass/
│   ├── checkpoints/            epoch-20 and final epoch-26 models
│   ├── data/test_2000/         independent 2,000-sequence OOD test set
│   ├── data/mcmc_same_sequences/
│   ├── data/identifiability_500/
│   ├── results/                direct-W, locality, mixing, and Gram results
│   └── src/                    training, evaluation, plotting, and REMC code
└── phenol/
    ├── configs/                MD and fixed-basis definitions
    ├── data/                   coordinate-only fit/selection data and parameters
    ├── frozen_models.json      hashes of every frozen paper artifact
    ├── prepare_data.py         trajectory preprocessing and OpenMM targets
    ├── evaluate_paper.py       one-command reproduction of the Phenol table
    ├── phenol_pairformer_arm1/ Network 1
    ├── phenol_dsm_controlled/  Network 2 and the physical holdout
    ├── phenol_pairformer_arm3/ Networks 3 and 3-P
    └── src/                    shared basis, MD, and force-target utilities
```

The large raw MD trajectories are not duplicated. The exact coordinate arrays
opened by DSM training/model selection and the complete 1,200-frame physical
holdout are included.

## Environment and first check

The reported calculations used the `ai4s` environment (Python 3.10.20,
PyTorch 2.11.0+cu130, NumPy 2.2.6, Matplotlib 3.10.9, and OpenMM 8.5.2).
A compatible environment can be created with

```bash
conda env create -f environment.yml
conda activate emergent-hamiltonian
python verify_release.py
```

`verify_release.py` authenticates all frozen checkpoints/artifacts, verifies
the data shapes and family identities, and checks the stored results against
`paper_results.json`. It is CPU-only.

The implementation tests can be run with

```bash
python -m pytest phenol
```

## Spin glass

### Final direct-W experiment

The final model is the epoch-26 checkpoint evaluated at `t=1e-4` on 2,000
unseen sequences, with five saved equilibrium configurations for each sequence:

```bash
python spin_glass/src/evaluate_direct_w.py --save-observations
python spin_glass/src/plot_paper_figures.py
```

The evaluator hooks the scalar pair head and reads the symmetrized internal
matrix exactly as it enters the score. The comparison scale is

```text
a_star = mean_c(<W>_c / J_c),
```

which gives the six pair-type/separation classes equal weight. It is not a fit
to individual observations. Expected values are Pearson correlation `0.9929`,
`a_star = 10.6244`, configuration-relative RMS `0.04045`, and `84.97%` of the
total absolute weight in `W` at separations `r=1,2`.

### Training and Fig. S1

The 40,000-sequence training archive is available from the original data
release:

- [SO3_TrainSet_40k.zip](https://github.com/WJXI/emergent-Hamiltonian-from-AI/releases/download/v1.0-data/SO3_TrainSet_40k.zip)

The verified archive is 155,913,997 bytes with SHA-256
`e82fb0b284dd63c4609574a8b9e8152b7248cb89b7d7c054a496be9915523f50`.
It contains 40 chunks and 40,000 unique sequences, with ten snapshots per
sequence.

Extract the archive, then move or rename its top-level `SO3_TrainSet_40k/`
directory to `spin_glass/data/train_40000/`; the `chunk_*.npz` files must be
directly inside that directory. The original 20 epochs and the
six-epoch low-noise continuation are reproduced by

```bash
python spin_glass/src/train_base.py
python spin_glass/src/train_low_noise.py --branch low_noise_mixture --epochs 6 \
  --start-checkpoint spin_glass/checkpoints/base_training/epoch_20.pt \
  --output-dir spin_glass/checkpoints/reproduced_low_noise_mixture
python spin_glass/src/plot_dataset_overview.py
```

The Fig. S1 script uses the training archive for panels (a,b) and the packaged
zero-overlap canonical test set for panels (c,d).

During the continuation, half of the diffusion times follow the original
uniform distribution on `[1e-4,1]`; half are log-uniform on `[1e-4,1e-2]`.

### REMC mixing and identifiability

The independent test-generation and aligned-overlap audit are

```bash
python spin_glass/src/generate_independent_test.py \
  --validation-sequences 0 --test-sequences 2000 --snapshots 5 \
  --burn-in 2000 --thinning 100 --sequence-seed 20260830 --test-forces

python spin_glass/src/mcmc_overlap_audit.py \
  --training-dir spin_glass/data/train_40000 \
  --independent-dir spin_glass/data/mcmc_same_sequences \
  --output spin_glass/results/aligned_overlap_audit.json

python spin_glass/src/identifiability_audit.py
```

After optimal common `O(3)` alignment, the 100-sweep overlap is `0.5095`
`[0.5062,0.5126]`, compared with the independently initialized baseline
`0.5081 [0.5051,0.5111]`. The six-class pooled score-Gram condition number is
`3.811` with six positive eigenvalues. This is identifiability within the
declared six-coupling physical family, not uniqueness of an arbitrary dense,
configuration-dependent `W`. The representation diagnostic uses the first 32
configurations of the included 500-sequence audit sample.

## Explicit-water phenol

### Data and physical system

All fitting and model selection use centered phenol coordinates only.

| Role | Independent water/velocity families | Production | Frames |
|---|---:|---:|---:|
| fit | 5 | 10 ns | 5,000 |
| selection | 1 | 2 ns | 1,000 |
| physical holdout | 8 new families | 24 ns | 1,200 |

The system is 13-atom phenol with no dummy atom, 512 rigid TIP3P waters,
FreeSolv GAFF 1.7/AM1-BCC solute parameters, PME electrostatics, a 2.5-nm
periodic box, and NVT dynamics at 298.15 K. The holdout is sampled every 20 ps.
No component energy, force, or force-field parameter enters training, model
selection, or the Network 3-P projection.

### Network definitions

- **Network 1:** graph-free scalar Pairformer using elements and all pair
  distances; it tests expressive capacity for the total PMF score.
- **Network 2:** 270 fixed, generic geometric carriers grouped into six physical
  sectors. Within-sector score-Gram truncation retains 240 directions, whose
  coefficients are configuration independent.
- **Network 3:** a scalar Pairformer acts on the same 240 retained carriers, so
  the effective coefficients may depend on the molecular configuration.
- **Network 3-P:** each of the six frozen Network 3 sector scores is projected
  separately onto constant coefficients. Each `lambda_c` is selected on the
  sixth family using only reconstruction of that frozen Network 3 sector.

The six sectors are deliberately aligned with the corresponding OpenMM
force-field components for the frozen post-training audit. Their decomposition
retains gauge redundancy, so the result supports a physically aligned PMF
readout and partial Hamiltonian emergence in the declared basis; it does not
claim a mathematically unique microscopic OpenMM decomposition.

### Reproduce the reported table

The complete table, including all four networks and all six sectors, is
recomputed directly from the released weights and holdout by

```bash
python phenol/evaluate_paper.py --device cuda
```

The result is written to `phenol/paper_evaluation.json`. Expected total-score
cosines are `0.961`, `0.992`, `0.967`, and `0.982` for Networks 1, 2, 3, and
3-P; Network 3-P gives `R_const^2 = 0.9669` against Network 3.

To refit only the sector-preserving projection from the frozen Network 3
models, without opening OpenMM labels, run

```bash
python phenol/phenol_pairformer_arm3/src/block_projection.py \
  --output-dir phenol/reproduced/network_3_P
```

### Retrain the three networks

```bash
python phenol/phenol_pairformer_arm1/src/pairformer_arm1.py \
  --output phenol/reproduced/network_1

python phenol/train_network2.py \
  --output-dir phenol/reproduced/network_2

python phenol/phenol_pairformer_arm3/src/pairformer_arm3.py \
  --output phenol/reproduced/network_3
```

All three use the same 5,000 fit frames, 1,000 selection frames, three DSM
noise levels, and three noise seeds.

### Regenerate the MD and processed arrays

Packmol 21.2.3 must be installed or specified with `PACKMOL_EXECUTABLE`.

```bash
# Six fit/selection families
python phenol/src/fresh_physical_families.py all \
  --config phenol/configs/fresh_physical_families.json
python phenol/prepare_data.py training

# Eight independent 3-ns holdout families and their frozen OpenMM targets
python phenol/phenol_dsm_controlled/src/final_holdout_md.py all
python phenol/prepare_data.py holdout
```

Full MD regeneration is substantially more expensive than verifying the
released processed arrays and frozen models.

## Provenance and license

The code is MIT licensed. The phenol parameter files retain their FreeSolv
provenance, recorded in
`phenol/data/public/freesolv_v0_52_parameter_source/selected/metadata.json`.

Repository: [WJXI/emergent-Hamiltonian-from-AI](https://github.com/WJXI/emergent-Hamiltonian-from-AI)
