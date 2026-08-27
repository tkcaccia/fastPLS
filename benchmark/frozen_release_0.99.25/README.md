# Frozen fastPLS 0.99.25 evidence rerun

This directory defines the release-level reproducibility contract requested for
the manuscript review. All central reruns must use the source archive recorded
in `frozen_release_manifest.tsv`; loading an arbitrary installed copy of
fastPLS is not permitted.

The frozen source is fastPLS 0.99.25 at Git commit
`7887401b09e25f54a546a253c255741cb1ab48e5`. The release scripts record:

- source-archive and input-data SHA-256 checksums;
- package, R, compiler, BLAS/LAPACK, CUDA, and GPU information;
- numerical solver controls, seeds, repetitions, and output contract;
- coefficient-path, prediction, route-selection, diagnostic, and warning
  regression checks;
- complete and failed benchmark rows without silent substitution.

The evidence classes are kept separate:

1. deterministic float64 SIMPLS validation against `pls::simpls.fit`;
2. approximate rSVD qualification across fixed seeds;
3. repeated selected CPU/CUDA workflow measurements;
4. NMR matched-solver and family-selected analyses;
5. exploratory ImageNet feasibility measurements.

Run `create_frozen_release.sh` from the repository root before executing the
benchmark workers. The script creates a clean source archive from the recorded
Git commit rather than from the working tree.
