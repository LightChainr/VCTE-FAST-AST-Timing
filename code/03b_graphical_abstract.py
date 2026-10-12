"""Graphical abstract for Clinica Chimica Acta (optional; landscape, >= 1328 px wide, readable at 13 cm).
Study numbers are read from the result files; the FAST positions on the scale and the report card are an illustrative example."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Wedge, Circle

H = Path(__file__).resolve().parent
R = H/'results'; F = H/'figures'


def rows(name, key):
    with (R/name).open() as f:
        return {tuple(x[k] for k in key): x for x in csv.DictReader(f)}


RCV = json.loads((R/'rcv_summary.json').read_text())
A = rows('03_observed_anchor_reclassification.csv', ('dataset', 'direction'))
Q = rows('23_rcv_threshold_flag.csv', ('analysis', 'direction', 'label'))
S = rows('14_actual_reclassification_strata.csv', ('stratum', 'level'))
C = rows('25_clinical_action_changes.csv', ('label', 'direction'))
G = rows('06_subgroup_support.csv', ('subgroup',))
early, late = A[('linked', 'earlier')], A[('linked', 'later')]
ff, nf = Q[('primary', 'all', 'reclassified_if_flagged')], Q[('primary', 'all', 'reclassified_if_not_flagged')]
sens = Q[('primary', 'all', 'sensitivity')]
m_e, m_l = S[('etiology_earlier', 'MASLD_compatible')], S[('etiology_later', 'MASLD_compatible')]
lag7 = S[('lag_band', '1-7')]
fr = C[('false_reassurance_among_sameday_assess', 'later')]
W = RCV['workload']
n_cmp = int(early['comparisons']) + int(late['comparisons'])
n_scan, n_pat = int(G[('all',)]['linked_scans']), int(G[('all',)]['linked_patients'])
pc = lambda v, d=1: f'{100*float(v):.{d}f}%'
share = lambda x, k: int(x[k])/int(x['discordant'])

NAVY, EARLY, LATE, INK, MUTED = '#1F3B5C', '#D55E00', '#0072B2', '#222222', '#6B6B6B'
TINT = '#EEF3F8'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 6.5, 'pdf.fonttype': 42})
WCM, HCM = 13.0, 8.0
fig = plt.figure(figsize=(WCM/2.54, HCM/2.54)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WCM); ax.set_ylim(0, HCM); ax.axis('off')
T = lambda x, y, s, **k: ax.text(x, y, s, **{'ha': 'center', 'va': 'center', 'color': INK, **k})


def tube(x, y, col, h=0.5, w=0.17):
    """Blood-collection tube with its base at (x, y)."""
    ax.add_patch(FancyBboxPatch((x - w/2, y), w, h, boxstyle=f'round,pad=0,rounding_size={w/2}', fc='white', ec=MUTED, lw=0.5, zorder=5))
    ax.add_patch(FancyBboxPatch((x - w/2, y), w, h*0.55, boxstyle=f'round,pad=0,rounding_size={w/2}', fc=col, ec='none', zorder=6))
    ax.add_patch(Rectangle((x - w/2 - 0.03, y + h), w + 0.06, 0.11, fc=col, ec=MUTED, lw=0.4, zorder=6))


def probe(x, y):
    """Elastography probe with its tip at (x, y)."""
    ax.add_patch(Polygon([(x - 0.08, y), (x + 0.08, y), (x + 0.13, y + 0.52), (x - 0.13, y + 0.52)], closed=True, fc=NAVY, ec='none', zorder=5))
    ax.add_patch(Rectangle((x - 0.11, y - 0.06), 0.22, 0.07, fc='#9FB4C9', ec='none', zorder=5))
    ax.plot([x, x + 0.05, x + 0.28], [y + 0.52, y + 0.7, y + 0.77], color=NAVY, lw=0.9, zorder=4)


def flag(x, y, s=1.0, col=EARLY):
    ax.plot([x, x], [y, y + 0.42*s], color=col, lw=0.9*s, zorder=9, solid_capstyle='round')
    ax.add_patch(Polygon([(x, y + 0.42*s), (x + 0.27*s, y + 0.33*s), (x, y + 0.24*s)], closed=True, fc=col, ec='none', zorder=9))


def curve(p0, p1, col, rad, lw=0.9, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=7,
                                 lw=lw, color=col, ls=ls, zorder=7, shrinkA=0, shrinkB=2))


def donut(x, y, p, col, label, r=0.62):
    ax.add_patch(Wedge((x, y), r, 0, 360, width=0.17, fc='#DCE3EA', ec='none'))
    ax.add_patch(Wedge((x, y), r, 90 - 360*float(p), 90, width=0.17, fc=col, ec='none'))
    T(x, y + 0.03, pc(p), fontsize=8.2, fontweight='bold', color=col)
    T(x, y - r - 0.22, label, fontsize=6.2, fontweight='bold', color=INK)


# Title
T(WCM/2, 7.6, 'AST from another date changes the FibroScan-AST (FAST) category of the same scan',
  fontsize=8.4, fontweight='bold', color=NAVY)
T(WCM/2, 7.2, f'{n_scan:,} elastography scans · {n_pat:,} patients · {n_cmp} same-scan comparisons with same-day AST',
  fontsize=6.2, color=MUTED)
ax.plot([0.3, WCM - 0.3], [6.95, 6.95], color='#C9D2DD', lw=0.6)

# Left: one scan, three AST dates, three FAST categories (illustrative positions)
ty = 6.0
ax.plot([0.55, 7.15], [ty, ty], color='#C9D2DD', lw=2.2, solid_capstyle='round', zorder=2)
for xt, lab in ((0.55, '−90 d'), (7.15, '+90 d')):
    T(xt, ty + 0.3, lab, fontsize=5.6, color=MUTED)
x_e, x_0, x_l = 1.9, 3.85, 5.95
probe(x_0, ty + 0.06)
T(x_0 + 0.75, ty + 0.55, 'Scan:\nLSM + CAP', fontsize=5.8, fontweight='bold', color=NAVY, ha='left', linespacing=1.0)
tube(x_0, ty - 0.68, '#8C8C8C', h=0.42)
T(x_0 + 0.5, ty - 0.5, 'same-day AST\n(reference)', fontsize=5.6, color=MUTED, ha='left', linespacing=1.0)
for x, col, lab in ((x_e, EARLY, 'earlier AST'), (x_l, LATE, 'later AST')):
    tube(x, ty + 0.06, col, h=0.42)
    T(x, ty - 0.2, lab, fontsize=6.0, fontweight='bold', color=col)

sx0, sx1, sy = 0.55, 7.15, 4.2
X = lambda f: sx0 + f*(sx1 - sx0)
cmap = LinearSegmentedColormap.from_list('fast', ['#7FBF7B', '#F3D36B', '#E8806E'])
grad = np.linspace(0, 1, 400)[None, :]
ax.imshow(grad, extent=(sx0, sx1, sy - 0.16, sy + 0.16), aspect='auto', cmap=cmap, zorder=3, interpolation='bicubic')
for c in (0.35, 0.67):
    ax.plot([X(c), X(c)], [sy - 0.24, sy + 0.24], color='white', lw=1.6, zorder=4)
    T(X(c), sy - 0.38, f'{c:.2f}', fontsize=5.4, color=MUTED)
for a_, b_, lab in ((0, 0.35, 'Rule-out'), (0.35, 0.67, 'Indeterminate'), (0.67, 1, 'Rule-in')):
    T(X((a_ + b_)/2), sy - 0.38, lab, fontsize=5.9, fontweight='bold', color='#3F3F3F')
T(sx0 - 0.02, sy + 0.32, 'FAST', fontsize=5.8, fontweight='bold', color=MUTED, ha='left')
f_e, f_0, f_l = 0.74, 0.47, 0.24
for f, col in ((f_e, EARLY), (f_0, '#5A5A5A'), (f_l, LATE)):
    ax.add_patch(Circle((X(f), sy), 0.12, fc=col, ec='white', lw=0.8, zorder=8))
ax.annotate('', xy=(X(f_0), sy + 0.14), xytext=(x_0, ty - 0.74), arrowprops=dict(arrowstyle='-|>', lw=0.7, ls=(0, (2, 1.5)), color='#5A5A5A', mutation_scale=7), zorder=7)
curve((X(f_0) + 0.12, sy + 0.1), (X(f_e) - 0.1, sy + 0.12), EARLY, -0.55, lw=1.1)
curve((X(f_0) - 0.12, sy + 0.1), (X(f_l) + 0.1, sy + 0.12), LATE, 0.55, lw=1.1)
T((X(f_0) + X(f_e))/2 + 0.15, sy + 0.62, 'with earlier AST', fontsize=5.8, fontweight='bold', color=EARLY)
T((X(f_0) + X(f_l))/2 - 0.15, sy + 0.62, 'with later AST', fontsize=5.8, fontweight='bold', color=LATE)

for k, (d, col, nm, word, key) in enumerate(((early, EARLY, 'earlier AST', 'up', 'upward'), (late, LATE, 'later AST', 'down', 'downward'))):
    xc = 2.1 + 3.3*k
    T(xc, 3.08, pc(d['proportion']), fontsize=13, fontweight='bold', color=col)
    T(xc, 2.62, f'changed with {nm}', fontsize=5.9, fontweight='bold', color=col)
    T(xc, 2.38, f'{pc(share(d, key), 0)} of changes moved {word}', fontsize=5.8)
T(3.85, 1.9, f'Later AST gave false reassurance in {pc(fr["proportion"])}\n'
  f'MASLD-compatible: {pc(m_e["proportion"])} / {pc(m_l["proportion"])}  ·  AST within 7 days: {pc(lag7["proportion"])}',
  fontsize=5.8, color=INK, linespacing=1.35)

# Right: the laboratory fix
ax.add_patch(FancyBboxPatch((7.6, 0.2), 5.15, 6.6, boxstyle='round,pad=0,rounding_size=0.25', fc=TINT, ec='none', zorder=0))
ax.text(7.85, 6.5, 'Laboratory fix: an RCV flag', ha='left', va='center', fontsize=7.6, fontweight='bold', color=NAVY)
ax.text(7.85, 6.17, f'Flag when the AST needed to cross 0.35 or 0.67 lies\nwithin the RCV (−{100*RCV["rcv_down"]:.1f}% to +{100*RCV["rcv_up"]:.1f}%) of the AST used',
        ha='left', va='top', fontsize=5.9, color=INK, linespacing=1.2)

# report card (example)
cx, cy, cw, ch = 7.95, 3.45, 4.45, 1.9
ax.add_patch(FancyBboxPatch((cx + 0.05, cy - 0.05), cw, ch, boxstyle='round,pad=0,rounding_size=0.1', fc='#C9D2DD', ec='none', zorder=1))
ax.add_patch(FancyBboxPatch((cx, cy), cw, ch, boxstyle='round,pad=0,rounding_size=0.1', fc='white', ec='#B8C4D1', lw=0.5, zorder=2))
ax.add_patch(FancyBboxPatch((cx, cy + ch - 0.36), cw, 0.36, boxstyle='round,pad=0,rounding_size=0.1', fc=NAVY, ec='none', zorder=3))
ax.text(cx + 0.15, cy + ch - 0.18, 'FAST report (example)', ha='left', va='center', fontsize=6.0, fontweight='bold', color='white', zorder=4)
ax.text(cx + 0.15, cy + 1.22, 'FAST 0.41  ·  Indeterminate', ha='left', va='center', fontsize=6.3, fontweight='bold', zorder=4)
ax.text(cx + 0.15, cy + 0.9, 'AST 52 U/L, measured 35 days before the scan', ha='left', va='center', fontsize=5.9, color=MUTED, zorder=4)
ax.add_patch(FancyBboxPatch((cx + 0.1, cy + 0.12), cw - 0.2, 0.5, boxstyle='round,pad=0,rounding_size=0.08', fc='#FDEBDD', ec='none', zorder=3))
flag(cx + 0.27, cy + 0.17, s=0.8)
ax.text(cx + 0.6, cy + 0.37, 'Threshold within the RCV of this AST:\nsame-day AST advised', ha='left', va='center',
        fontsize=5.8, fontweight='bold', color='#8A3B00', linespacing=1.1, zorder=4)

T(10.18, 3.05, 'Category changed with same-day AST', fontsize=6.0, color=MUTED)
donut(9.15, 2.1, ff['proportion'], EARLY, 'Flagged')
donut(11.2, 2.1, nf['proportion'], '#7A93AD', 'Not flagged')
T(10.18, 0.62, f'Captured {pc(sens["proportion"])} of reclassifications\nSame-day AST advised for {pc(W["nonconcurrent_flagged"]/W["scored_scans"])} of scans',
  fontsize=5.9, color=INK, linespacing=1.25)

# Bottom-left take-home
ax.add_patch(FancyBboxPatch((0.4, 0.3), 6.9, 0.85, boxstyle='round,pad=0,rounding_size=0.15', fc='#FFF4E5', ec='none'))
T(3.85, 0.72, 'Report the AST date with every FAST result', fontsize=7.0, fontweight='bold', color='#8A3B00')

for ext, kw in (('png', dict(dpi=600)), ('tif', dict(dpi=600, pil_kwargs={'compression': 'tiff_lzw'})), ('pdf', {})):
    fig.savefig(F/f'Graphical_Abstract.{ext}', **kw)
print('Graphical abstract written')
