"""Fly egocentric visual input demo: synthetic courtship -> hex-eye rendering -> HR motion response."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erf

rng = np.random.default_rng(0)
FPS, N = 30, 600
dt = 1 / FPS
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

# 1. Synthetic tracks (mm, rad). Stand-in for SLEAP pose tracks.
F = np.zeros((N, 2)); M = np.zeros((N, 2)); fh = np.zeros(N); mh = np.zeros(N)
chase = np.zeros(N, bool)
F[0] = [20, 20]; M[0] = [26, 20]; fh[0] = rng.uniform(-np.pi, np.pi); mh[0] = np.pi
state, left = True, 0
for t in range(1, N):
    if left <= 0:
        state = not state; left = rng.integers(60, 120)
    left -= 1; chase[t] = state
    fh[t] = fh[t-1] + rng.normal(0, 0.15)
    F[t] = np.clip(F[t-1] + 3 * dt * np.array([np.cos(fh[t]), np.sin(fh[t])]), 2, 38)
    if state:
        d = F[t-1] - M[t-1]; dist = np.hypot(*d)
        turn = np.clip(wrap(np.arctan2(d[1], d[0]) - mh[t-1]), -8*dt, 8*dt)
        mh[t] = mh[t-1] + turn
        sp = np.clip(4 * (dist - 4.5), 0, 25)
    else:
        # independent wander; soft ring keeps him 10-25 mm from her, heading away on entry
        d = F[t-1] - M[t-1]; dist = np.hypot(*d); to_f = np.arctan2(d[1], d[0])
        if chase[t-1]:  # just stopped chasing: turn to head away from her
            mh[t-1] = to_f + np.pi + rng.normal(0, 0.5)
        away = wrap(to_f + np.pi - mh[t-1]); toward = wrap(to_f - mh[t-1])
        steer = 0.0
        if dist < 10: steer = np.clip(away, -6*dt, 6*dt)
        elif dist > 25: steer = np.clip(toward, -6*dt, 6*dt)
        mh[t] = mh[t-1] + steer + rng.normal(0, 0.15); sp = 9
    M[t] = np.clip(M[t-1] + sp * dt * np.array([np.cos(mh[t]), np.sin(mh[t])]), 2, 38)
np.savez("tracks.npz", female=F, male=M, female_heading=fh, male_heading=mh, chasing=chase)

# 2. Hex lattice (azimuth +-135, elevation +-60); spacing chosen to give ~721 points
S = 7.2
pts = []
for r, el in enumerate(np.arange(-60, 60.01, S * np.sqrt(3) / 2)):
    off = (S / 2) * (r % 2)
    for az in np.arange(-135 + off, 135.01, S):
        pts.append((az, el, r))
P = np.array(pts); AZ, EL, ROW = P[:, 0], P[:, 1], P[:, 2].astype(int)
n_om = len(P); SIG = 4.0 / 2.355 * 1.0 + 1.5  # acceptance-angle sigma (deg)
# 3-segment female body along her heading axis: (centre offset from body centre, semi-length, semi-width, height) in mm
SEG = np.array([(1.2, 0.3, 0.35, 0.8),   # head ~0.6 mm
                (0.4, 0.5, 0.50, 1.0),   # thorax ~1.0 mm
                (-0.8, 0.7, 0.45, 0.9)]) # abdomen ~1.4 mm

# per-segment geometry in the male frame, shape (N, 3)
sx = F[:, 0:1] + SEG[:, 0] * np.cos(fh)[:, None]; sy = F[:, 1:2] + SEG[:, 0] * np.sin(fh)[:, None]
sdx, sdy = sx - M[:, 0:1], sy - M[:, 1:2]
sdist = np.hypot(sdx, sdy)
saz = np.degrees(wrap(np.arctan2(sdy, sdx) - mh[:, None]))        # segment azimuth in male frame
sphi = wrap(fh[:, None] - np.arctan2(sdy, sdx))                    # heading vs line of sight
swid = 2 * np.sqrt((SEG[:, 1] * np.sin(sphi))**2 + (SEG[:, 2] * np.cos(sphi))**2)
shw = np.degrees(np.arctan(swid / 2 / sdist))                      # half-width (deg)
shh = np.degrees(np.arctan(SEG[:, 3] / 2 / sdist))                 # half-height (deg)
dx, dy = F[:, 0] - M[:, 0], F[:, 1] - M[:, 1]
dist = np.hypot(dx, dy)
az_f = np.degrees(wrap(np.arctan2(dy, dx) - mh))                  # female body-centre azimuth
rel = (saz - az_f[:, None] + 180) % 360 - 180
ang_w = (rel + shw).max(1) - (rel - shw).min(1)                    # azimuthal extent of the union

def render(t):
    cov = np.zeros(n_om)
    for j in range(3):
        hw, hh = max(shw[t, j], 0.5), max(shh[t, j], 0.5)
        daz = (AZ - saz[t, j] + 180) % 360 - 180
        r = np.hypot(daz / hw, EL / hh)
        cov = np.maximum(cov, 0.5 * (1 - erf((r - 1) * min(hw, hh) / (np.sqrt(2) * SIG))))
    return 1 - cov  # bright background (1), dark female (union of segments)

L = np.array([render(t) for t in range(N)])  # (frames, ommatidia)

# 3. Hassenstein-Reichardt EMD between horizontal neighbours (stand-in for T4/T5)
left, right = [], []
for r in np.unique(ROW):
    idx = np.where(ROW == r)[0]; idx = idx[np.argsort(AZ[idx])]
    left += list(idx[:-1]); right += list(idx[1:])
left, right = np.array(left), np.array(right)
tau = 0.05; k = dt / (tau + dt)
def lowpass(x):
    y = np.zeros_like(x); y[0] = x[0]
    for t in range(1, len(x)): y[t] = y[t-1] + k * (x[t] - y[t-1])
    return y
a, b = L[:, left], L[:, right]
emd = lowpass(a) * b - a * lowpass(b)   # >0: rightward (increasing azimuth)
emd_az = (AZ[left] + AZ[right]) / 2
front = np.abs(emd_az) <= 30
resp = np.abs(emd[:, front]).mean(1)
rc, rn = resp[chase].mean(), resp[~chase].mean()

# Fig 1
tc = np.where(chase)[0]; tn = np.where(~chase)[0]
snaps = [("chasing, near", tc[np.argmin(dist[tc])]), ("chasing, far", tc[np.argmax(dist[tc])]),
         ("not chasing", tn[np.argmin(np.abs(dist[tn] - np.median(dist[tn])))])]
fig = plt.figure(figsize=(11, 6.5)); gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
for i, (name, t) in enumerate(snaps):
    ax = fig.add_subplot(gs[0, i])
    sc = ax.scatter(AZ, EL, c=L[t], cmap="gray", vmin=0, vmax=1, s=22, edgecolor="0.7", linewidth=0.3)
    ax.set_aspect("equal"); ax.set_title(f"{name}\nt={t/FPS:.1f}s, d={dist[t]:.1f} mm, size={ang_w[t]:.0f} deg", fontsize=9)
    ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("elevation (deg)") if i == 0 else None
ax = fig.add_subplot(gs[1, :]); T = np.arange(N) / FPS
for t in range(N):
    if chase[t]: ax.axvspan(t/FPS, (t+1)/FPS, color="tab:orange", alpha=0.15, lw=0)
ax.plot(T, ang_w, color="k", label="female angular width (deg)")
ax.plot(T, az_f, color="tab:blue", label="female azimuth (deg)")
for _, t in snaps: ax.axvline(t/FPS, color="r", ls=":", lw=1)
ax.set_xlabel("time (s)"); ax.legend(loc="upper right", fontsize=8); ax.set_title("Shaded = chasing (synthetic labels)", fontsize=9)
fig.tight_layout(); fig.savefig("fig_eye_view.png", dpi=130); plt.close(fig)

# Fig 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 2.5]})
a1.bar(["chasing", "not chasing"], [rc, rn], color=["tab:orange", "0.6"]); a1.set_ylabel("mean |EMD| (a.u.), frontal +-30 deg")
for t in range(N):
    if chase[t]: a2.axvspan(t/FPS, (t+1)/FPS, color="tab:orange", alpha=0.15, lw=0)
a2.plot(T, resp, color="k", lw=1); a2.set_xlabel("time (s)"); a2.set_ylabel("mean |EMD|")
fig.tight_layout(); fig.savefig("fig_motion_response.png", dpi=130); plt.close(fig)

print(f"ommatidia={n_om} chase_frac={chase.mean():.2f} dist chase={dist[chase].mean():.1f}mm non={dist[~chase].mean():.1f}mm")
print(f"mean|EMD| frontal: chasing={rc:.4f} not={rn:.4f} ratio={rc/rn:.2f}")
print(f"angular width: chasing mean={ang_w[chase].mean():.1f} deg, not={ang_w[~chase].mean():.1f} deg")
