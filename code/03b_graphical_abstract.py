"""Graphical abstract for Clinical Biochemistry (optional; banner 5 x 13 cm, >= 531 x 1328 px, readable at that size).
Three panels read left to right: the problem, the score comparison, the laboratory fix. Study numbers are read from the result files."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Wedge

H = Path(__file__).resolve().parent
R = H/'results'; F = H/'figures'


def rows(name, key):
    with (R/name).open() as f:
        return {tuple(x[k] for k in key): x for x in csv.DictReader(f)}


RCV = json.loads((R/'rcv_summary.json').read_text())
A = rows('03_observed_anchor_reclassification.csv', ('dataset', 'direction'))
Q = rows('23_rcv_threshold_flag.csv', ('analysis', 'direction', 'label'))
G = rows('06_subgroup_support.csv', ('subgroup',))
SC = rows('29_reclassification_by_score.csv', ('set', 'score'))
PH = rows('29_posthoc_ratio_cancellation.csv', ('score',))
early, late = A[('linked', 'earlier')], A[('linked', 'later')]
ff, nf = Q[('primary', 'all', 'reclassified_if_flagged')], Q[('primary', 'all', 'reclassified_if_not_flagged')]
n_cmp = int(early['comparisons']) + int(late['comparisons'])
n_scan, n_pat = int(G[('all',)]['linked_scans']), int(G[('all',)]['linked_patients'])
n_sc = int(SC[('primary_all', 'FAST')]['comparisons'])
pc = lambda v, d=1: f'{100*float(v):.{d}f}%'

NAVY, EARLY, LATE, INK, MUTED = '#1F3B5C', '#D55E00', '#0072B2', '#222222', '#6B6B6B'
TINT, LINE, WARM = '#EEF3F8', '#C9D2DD', '#FFF4E5'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 6.0, 'pdf.fonttype': 42})
WCM, HCM = 13.0, 5.0
fig = plt.figure(figsize=(WCM/2.54, HCM/2.54)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WCM); ax.set_ylim(0, HCM); ax.axis('off')
T = lambda x, y, s, **k: ax.text(x, y, s, **{'ha': 'center', 'va': 'center', 'color': INK, **k})


def tube(x, y, col, h=0.36, w=0.14):
    """Blood-collection tube with its base at (x, y)."""
    ax.add_patch(FancyBboxPatch((x - w/2, y), w, h, boxstyle=f'round,pad=0,rounding_size={w/2}', fc='white', ec=MUTED, lw=0.45, zorder=5))
    ax.add_patch(FancyBboxPatch((x - w/2, y), w, h*0.55, boxstyle=f'round,pad=0,rounding_size={w/2}', fc=col, ec='none', zorder=6))
    ax.add_patch(Rectangle((x - w/2 - 0.025, y + h), w + 0.05, 0.09, fc=col, ec=MUTED, lw=0.35, zorder=6))


def probe(x, y, s=0.8):
    """Elastography probe with its tip at (x, y)."""
    ax.add_patch(Polygon([(x - 0.07*s, y), (x + 0.07*s, y), (x + 0.12*s, y + 0.5*s), (x - 0.12*s, y + 0.5*s)], closed=True, fc=NAVY, ec='none', zorder=5))
    ax.add_patch(Rectangle((x - 0.1*s, y - 0.05*s), 0.2*s, 0.06*s, fc='#9FB4C9', ec='none', zorder=5))


def flag(x, y, s=1.0, col=EARLY):
    ax.plot([x, x], [y, y + 0.36*s], color=col, lw=0.8*s, zorder=9, solid_capstyle='round')
    ax.add_patch(Polygon([(x, y + 0.36*s), (x + 0.24*s, y + 0.28*s), (x, y + 0.2*s)], closed=True, fc=col, ec='none', zorder=9))


def donut(x, y, p, col, label, r=0.44):
    ax.add_patch(Wedge((x, y), r, 0, 360, width=0.09, fc='#DCE3EA', ec='none'))
    ax.add_patch(Wedge((x, y), r, 90 - 360*float(p), 90, width=0.09, fc=col, ec='none'))
    T(x, y, pc(p), fontsize=6.3, fontweight='bold', color=col)
    T(x, y - r - 0.15, label, fontsize=5.8, fontweight='bold')


def header(x, y, num, s, col=NAVY):
    ax.add_patch(plt.Circle((x + 0.13, y), 0.13, fc=col, ec='none'))
    T(x + 0.13, y - 0.005, num, fontsize=6.2, fontweight='bold', color='white')
    ax.text(x + 0.36, y, s, ha='left', va='center', fontsize=6.6, fontweight='bold', color=col)


# Title
T(WCM/2, 4.68, 'AST from another date changes FibroScan-AST (FAST) more often than Agile',
  fontsize=8.2, fontweight='bold', color=NAVY)
T(WCM/2, 4.36, f'{n_scan:,} elastography scans · {n_pat:,} patients · {n_cmp} same-scan comparisons with same-day AST as reference',
  fontsize=5.8, color=MUTED)

# Panel 1: one scan, AST from another date
header(0.25, 3.92, '1', 'Same scan, AST from another date'.replace('Same scan, ', ''))
ty = 3.05
ax.plot([0.45, 4.25], [ty, ty], color=LINE, lw=1.8, solid_capstyle='round', zorder=2)
x_e, x_0, x_l = 1.0, 2.35, 3.7
probe(x_0, ty + 0.04)
T(x_0, ty - 0.17, 'scan', fontsize=5.6, color=NAVY, fontweight='bold')
for x, col, lab in ((x_e, EARLY, 'AST before'), (x_l, LATE, 'AST after')):
    tube(x, ty + 0.04, col)
    T(x, ty - 0.17, lab, fontsize=5.6, fontweight='bold', color=col)
for k, (d, col, nm, arrow, key) in enumerate(((early, EARLY, 'earlier', '▲', 'upward'), (late, LATE, 'later', '▼', 'downward'))):
    xc = 1.3 + 2.1*k
    T(xc, 2.2, pc(d['proportion']), fontsize=13, fontweight='bold', color=col)
    T(xc, 1.68, f'FAST changed\nwith {nm} AST', fontsize=5.8, linespacing=1.1)
    T(xc, 1.2, f'{arrow} mostly {key}', fontsize=5.8, fontweight='bold', color=col)
T(2.35, 0.82, 'Frequent even within 7 days', fontsize=5.6, color=MUTED)

# Panel 2: score comparison on the same comparisons
ax.add_patch(FancyBboxPatch((4.55, 0.7), 4.0, 3.5, boxstyle='round,pad=0,rounding_size=0.2', fc=TINT, ec='none', zorder=0))
header(4.7, 3.92, '2', 'Compared with other scores')
T(6.55, 3.55, f'Categories changed, same {n_sc} comparisons', fontsize=5.6, color=MUTED)
bx, bw, top = 5.65, 2.25, 3.12
for i, (k, nm, col) in enumerate((('FAST', 'FAST', NAVY), ('FIB4', 'FIB-4', '#7A93AD'), ('Agile4', 'Agile 4', '#7A93AD'), ('Agile3', 'Agile 3+', '#7A93AD'))):
    p = float(SC[('primary_all', k)]['proportion']); y = top - 0.42*i
    ax.text(bx - 0.1, y, nm, ha='right', va='center', fontsize=6.0, fontweight='bold', color=col if k == 'FAST' else INK)
    ax.add_patch(Rectangle((bx, y - 0.13), bw, 0.26, fc='white', ec='none', zorder=1))
    ax.add_patch(Rectangle((bx, y - 0.13), bw*p/0.25, 0.26, fc=col, ec='none', zorder=2))
    ax.text(bx + bw*p/0.25 + 0.06, y, pc(p), ha='left', va='center', fontsize=6.0, fontweight='bold', color=col if k == 'FAST' else INK)
T(6.55, 1.32, 'Agile changed about half as often as FAST', fontsize=5.8, fontweight='bold', color=NAVY)
fib = PH[('FIB4',)]
T(6.55, 0.96, f'FIB-4 rose to {pc(fib["AST_only_changed"])} when AST and ALT\ncame from different specimens', fontsize=5.6, color=MUTED, linespacing=1.1)

# Panel 3: the laboratory fix
header(8.75, 3.92, '3', 'Laboratory fix: an RCV flag', col='#8A3B00')
ax.text(8.8, 3.52, f'Flag FAST if the AST that would cross a cutoff\nlies within the reference change value\n(−{100*RCV["rcv_down"]:.1f}% to +{100*RCV["rcv_up"]:.1f}%) of the AST used',
        ha='left', va='top', fontsize=5.7, color=INK, linespacing=1.15)
flag(12.42, 3.76, s=0.8)
T(10.8, 2.56, 'Category changed with same-day AST', fontsize=5.6, color=MUTED)
donut(10.05, 1.84, ff['proportion'], EARLY, 'Flagged')
donut(11.55, 1.84, nf['proportion'], '#7A93AD', 'Not flagged')
T(10.8, 0.86, 'Same data the laboratory already holds', fontsize=5.6, color=MUTED)

# Take-home strip
ax.add_patch(FancyBboxPatch((0.25, 0.12), WCM - 0.5, 0.48, boxstyle='round,pad=0,rounding_size=0.12', fc=WARM, ec='none'))
T(WCM/2, 0.36, 'Report the AST date   ·   Take AST and ALT from one specimen   ·   Flag FAST results near a cutoff',
  fontsize=6.6, fontweight='bold', color='#8A3B00')

for ext, kw in (('png', dict(dpi=600)), ('tif', dict(dpi=600, pil_kwargs={'compression': 'tiff_lzw'})), ('pdf', {})):
    fig.savefig(F/f'Graphical_Abstract.{ext}', **kw)
print('Graphical abstract written')
