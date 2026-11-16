# Fly egocentric visual input demo

**Question:** when a male fly's social experience is altered, what does his visual system receive during courtship?

**Pipeline** (`python fly_eye_view.py`, numpy/scipy/matplotlib only):
- Synthetic pose tracks (600 frames, 30 fps) with a chasing label, standing in for SLEAP. When not chasing, the male wanders independently (15.5 mm from her on average vs 7.5 mm chasing).
- Egocentric rendering of a 3-segment female body (head, thorax, abdomen ellipses, drawn as their union) on a 760-ommatidia hex lattice with Gaussian blur.
- Hassenstein-Reichardt motion detector, a minimal stand-in for T4/T5.

**Result:** frontal mean |EMD| was 0.0023 chasing vs 0.0006 not (3.75x); female angular width was 22.0 vs 8.9 deg. Much of this follows from the distance difference.

**Limitations:** synthetic tracks, a crude body (a first step toward fitted body models), no 3D geometry, and an EMD instead of the connectome-constrained model. A pipeline check, not a finding.

**Next:** real SLEAP tracks, a fitted body model, flyvis.

Refs: Lappalainen et al., Nature 2024, doi:10.1038/s41586-024-07939-3. Roemschied et al., Curr Biol 2026, doi:10.1016/j.cub.2026.02.003.
