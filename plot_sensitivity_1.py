"""Plot the sensitivity study of sensitivity_sweep_v8.m from its two CSV files.

Reads   sensitivity_severity.csv   (fault, severity, tripped, trip_delay_ms, detection_delay_ms)
        sensitivity_noise.csv      (case, seed, max_cusum, max_residual, tripped, detection_delay_ms)

Every curve is its own panel. --layout chooses how the panels are saved:
    rows    two curves per row on ONE figure, an odd last curve centred (default)
            e.g. 5 curves -> 3 rows: 2 + 2 + 1 centred
    single  one PNG per curve
    pair    two curves side by side per PNG; an odd last curve is centred on its own PNG
    grid    all curves of a figure on one PNG (the original 2 x 3 / 1 x 2 layout)

Usage:  python plot_sensitivity.py --dir fault_analysis
        python plot_sensitivity.py --dir fault_analysis --layout pair --outdir figs
Needs:  pandas, numpy, matplotlib
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, RED, GREY, INK = '#2a78d6', '#e34948', '#8a8982', '#1f1f1f'
# per fault: axis label, log x axis?, healthy-side value (the "mild" end is the one nearest this value)
SWEEPS = [
    ('LG_A',           'LG fault: fault resistance (ohm)',          True,  np.inf),
    ('Overcurrent_A',  'Overcurrent: load resistance (ohm)',        False, 5.0),
    ('Overvoltage_A',  'Overvoltage: voltage multiplier',           False, 1.0),
    ('Undervoltage_A', 'Undervoltage: voltage multiplier',          False, 1.0),
    ('OpenCircuit_A',  'Open circuit: series resistance (ohm)',     True,  5.0),
]

def mildest_detected(sev, tripped, healthy):
    """Detected severity closest to the healthy value (the weakest fault still caught)."""
    s = sev[tripped]
    if s.size == 0: return np.nan
    if np.isinf(healthy): return s.max()
    return s[np.argmin(np.abs(s - healthy))]

# ---------------- panels: each draws one curve on a given axis ----------------
def severity_panel(ax, ts, name, label, logx, healthy):
    d = ts[ts.fault == name].sort_values('severity')
    sev = d.severity.values; trip = d.tripped.values > 0; dly = d.detection_delay_ms.values
    ax.plot(sev[trip], dly[trip], 'o-', color=BLUE, lw=1.6, ms=5, label='detected')
    if (~trip).any():
        top = np.nanmax(dly) * 1.25 if np.isfinite(dly).any() else 1.0
        ax.plot(sev[~trip], np.full((~trip).sum(), top), 'x', color=RED, ms=9, mew=2, label='not detected')
    if logx: ax.set_xscale('log')
    mild = mildest_detected(sev, trip, healthy)
    ax.set_xlabel(label); ax.set_ylabel('detection delay (ms)'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='best')
    ax.set_title(f'{name.replace("_", " ")}: mildest detected {mild:.4g}, missed {int((~trip).sum())}/{len(d)}', fontsize=10)

def healthy_panel(ax, tn, h):
    H = tn[tn['case'] == 'Healthy'].sort_values('seed')
    ax.plot(H.seed, H.max_cusum, 'o-', color=BLUE, lw=1.4, ms=4)
    ax.axhline(h, color=INK, ls='--', lw=1); ax.text(H.seed.min(), h * 1.03, f'threshold H = {h:g}', fontsize=8)
    ax.set_ylim(0, max(h * 1.25, H.max_cusum.max() * 1.15)); ax.grid(alpha=.3)
    ax.set_xlabel('noise seed'); ax.set_ylabel('largest CUSUM value in the healthy run')
    ax.set_title(f'Healthy run: {int(H.tripped.sum())} false trips in {len(H)} seeds, worst CUSUM {H.max_cusum.max():.3g}', fontsize=10)

def mild_panel(ax, tn, case):
    M = tn[tn['case'] == case].sort_values('seed'); ok = M.tripped.values > 0
    ax.plot(M.seed[ok], M.detection_delay_ms[ok], 's-', color=RED, ms=4, lw=1.3, label=case.replace('_', ' '))
    if ok.any():
        v = M.detection_delay_ms[ok]
        ax.axhline(v.mean(), color=GREY, ls=':', lw=1)
        ax.set_title(f'{case.replace("_", " ")}: delay {v.mean():.2f} ms mean, {v.std():.2f} std, min {v.min():.1f}, max {v.max():.1f}; missed {int((~ok).sum())}/{len(M)}', fontsize=9)
    if (~ok).any(): ax.plot(M.seed[~ok], np.zeros((~ok).sum()), 'x', color=INK, ms=9, mew=2, label='not detected')
    ax.set_xlabel('noise seed'); ax.set_ylabel('detection delay (ms)'); ax.grid(alpha=.3); ax.legend(fontsize=8)

# ---------------- layout: save a list of (file stem, draw function) ----------------
def save_panels(panels, layout, outdir, grid_name, grid_shape):
    if layout == 'rows':
        n = len(panels); nr = (n + 1) // 2
        fig = plt.figure(figsize=(12.4, 4.6 * nr))
        gs = fig.add_gridspec(nr, 4)                      # 4 columns: a pair uses 2 + 2, a lone last panel uses the middle 2
        for i, (stem, fn) in enumerate(panels):
            row, pos = divmod(i, 2)
            if i == n - 1 and n % 2 == 1: cols = slice(1, 3)      # last odd panel centred
            else: cols = slice(2 * pos, 2 * pos + 2)
            fn(fig.add_subplot(gs[row, cols]))
        fig.tight_layout()
        path = os.path.join(outdir, grid_name + '_rows.png'); fig.savefig(path, dpi=170); plt.close(fig); print('wrote', path)
        return
    if layout == 'single':
        groups = [[p] for p in panels]
    elif layout == 'pair':
        groups = [panels[i:i + 2] for i in range(0, len(panels), 2)]
    else:
        groups = [panels]
    for gi, grp in enumerate(groups):
        if layout == 'pair' and len(grp) == 1 and len(panels) > 1:
            # lone last curve of an odd set: same canvas as a pair, curve centred
            fig = plt.figure(figsize=(12.4, 4.6)); gs = fig.add_gridspec(1, 4)
            grp[0][1](fig.add_subplot(gs[0, 1:3])); fig.tight_layout()
            path = os.path.join(outdir, f'{grid_name}_pair{gi + 1}.png'); fig.savefig(path, dpi=170); plt.close(fig); print('wrote', path)
            continue
        if layout == 'grid': nr, nc = grid_shape
        else: nr, nc = 1, len(grp)
        fig, axs = plt.subplots(nr, nc, figsize=(6.2 * nc, 4.6 * nr), squeeze=False)
        axs = axs.ravel()
        for ax, (stem, fn) in zip(axs, grp): fn(ax)
        for ax in axs[len(grp):]: ax.set_visible(False)
        fig.tight_layout()
        stem = grid_name if layout == 'grid' else (grp[0][0] if layout == 'single' else f'{grid_name}_pair{gi + 1}')
        path = os.path.join(outdir, stem + '.png'); fig.savefig(path, dpi=170); plt.close(fig); print('wrote', path)

def summary_text(ts):
    out = []
    for name, label, logx, healthy in SWEEPS:
        d = ts[ts.fault == name]
        if d.empty: continue
        trip = d.tripped.values > 0
        out.append(f'{name:16s} mildest detected {mildest_detected(d.severity.values, trip, healthy):.4g}  '
                   f'delay {d.detection_delay_ms.min():.1f} to {d.detection_delay_ms.max():.1f} ms  missed {int((~trip).sum())}/{len(d)}')
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='fault_analysis', help='folder holding the two sensitivity CSV files')
    ap.add_argument('--outdir', default=None, help='where to write the PNGs (default: same as --dir)')
    ap.add_argument('--layout', choices=('rows', 'single', 'pair', 'grid'), default='rows')
    ap.add_argument('--h', type=float, default=0.5, help='CUSUM threshold H of the current channels')
    a = ap.parse_args(); out = a.outdir or a.dir; os.makedirs(out, exist_ok=True)
    ts = pd.read_csv(os.path.join(a.dir, 'sensitivity_severity.csv'))
    tn = pd.read_csv(os.path.join(a.dir, 'sensitivity_noise.csv'))

    sev_panels = [(f'sev_{n}', (lambda ax, n=n, l=l, lg=lg, hh=hh: severity_panel(ax, ts, n, l, lg, hh)))
                  for n, l, lg, hh in SWEEPS if not ts[ts.fault == n].empty]
    save_panels(sev_panels, a.layout, out, 'fig_5_21', (2, 3))

    noise_panels = [('noise_healthy', lambda ax: healthy_panel(ax, tn, a.h))]
    noise_panels += [(f'noise_{c}', (lambda ax, c=c: mild_panel(ax, tn, c))) for c in tn['case'].unique() if c != 'Healthy']
    save_panels(noise_panels, a.layout, out, 'fig_5_22', (1, 2))

    print('\n'.join(summary_text(ts)))
