"""Threshold proximity relative to the reference change value (RCV) of AST (v14, CCLM).

The flag uses only what is available when FAST is calculated: LSM, CAP and the AST in hand. A result is
flagged when the AST needed to reach a reachable FAST threshold (0.35 or 0.67) lies inside the 95%
log-normal RCV interval of that AST. The paired comparisons then show how often flagged and
unflagged results were reclassified by same-day AST.
"""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from scipy.special import logit

H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m", H/"01_rebuild_and_analyze.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

CVI = 9.5            # within-subject biological variation of AST, EuBIVAS (Carobene et al., Clin Chem 2017)
CVA_PRIMARY = 4.75   # desirable analytical imprecision = 0.5 x CVI
CVA_SENSITIVITY = 3.0
# Added at v15 (post hoc, after reading Au et al., J Hepatol 2026): AST within-subject CV of 16.3% over a median of
# 15 days in patients screened for a MASLD trial (central laboratory; includes analytical variation). Holding CVA at
# 4.75%, the biological component is sqrt(16.3^2 - 4.75^2) = 15.6%.
WCV_MASLD = 16.3
CVI_MASLD = float(np.sqrt(WCV_MASLD**2 - CVA_PRIMARY**2))
Z = 1.96


def rcv(cva, cvi=CVI):
    """Two-sided 95% log-normal RCV (Fokkema et al., Clin Chem 2006)."""
    sigma = np.sqrt(np.log(1 + (np.hypot(cva, cvi)/100)**2))
    return np.exp(Z*np.sqrt(2)*sigma), np.exp(-Z*np.sqrt(2)*sigma)


def margins(lsm, cap, ast):
    b = -1.65 + 1.07*np.log(lsm) + 2.66e-8*np.asarray(cap)**3
    out = []
    for c in (0.35, 0.67):
        reach = b > logit(c)
        ac = np.where(reach, 63.3/np.where(reach, b - logit(c), 1.0), np.nan)
        out.append(np.log(ac/np.asarray(ast)))
    return np.column_stack(out)


def flag(d, up, dn):
    D = margins(d.lsm.to_numpy(), d.cap.to_numpy(), d.AST.to_numpy())
    with np.errstate(invalid="ignore"):
        return ((D > np.log(dn)) & (D < np.log(up))).any(axis=1)


def prop(d, col, label, **extra):
    lo, hi = m.cluster_interval(d, col)
    return dict(label=label, n=len(d), patients=d.research_token.nunique(), events=int(d[col].sum()),
                proportion=float(d[col].mean()), ci_low=lo, ci_high=hi, **extra)


a = pd.read_parquet(m.DATA/"same_day_anchor_comparisons.parquet")
a["reclassified"] = a.discordant.astype(bool)
rows = []
for cva, cvi, tag in [(CVA_PRIMARY, CVI, "primary"), (CVA_SENSITIVITY, CVI, "cva3"), (CVA_PRIMARY, CVI_MASLD, "masld_wcv")]:
    up, dn = rcv(cva, cvi)
    a["flag"] = flag(a, up, dn)
    for direction, d in [("all", a), ("earlier", a[a.direction.eq("earlier")]), ("later", a[a.direction.eq("later")])]:
        meta = dict(analysis=tag, direction=direction, cva=cva, rcv_up=up-1, rcv_down=1-dn)
        f, u = d[d.flag], d[~d.flag]
        r = d[d.reclassified]
        rows += [prop(d.assign(x=d.flag), "x", "flagged_share", **meta),
                 prop(f, "reclassified", "reclassified_if_flagged", **meta),
                 prop(u, "reclassified", "reclassified_if_not_flagged", **meta),
                 prop(r.assign(x=r.flag), "x", "sensitivity", **meta),
                 prop(u.assign(x=~u.reclassified), "x", "npv", **meta)]
    if tag == "primary":
        a["outside_rcv"] = (a.AST/a.reference_AST > up) | (a.AST/a.reference_AST < dn)
        rows.append(prop(a, "outside_rcv", "AST_change_beyond_RCV", analysis=tag, direction="all", cva=cva, rcv_up=up-1, rcv_down=1-dn))
        # Decomposition added at review: a threshold crossed by an AST change within the RCV is flagged by construction.
        rc = a[a.reclassified]
        rows.append(prop(rc.assign(x=~rc.outside_rcv), "x", "reclassified_with_change_within_RCV", analysis=tag, direction="all", cva=cva, rcv_up=up-1, rcv_down=1-dn))
        assert rc[~rc.outside_rcv].flag.all()
        rb = rc[rc.outside_rcv]
        rows.append(prop(rb.assign(x=rb.flag), "x", "sensitivity_among_reclassified_beyond_RCV", analysis=tag, direction="all", cva=cva, rcv_up=up-1, rcv_down=1-dn))
        primary_flag = a.flag.copy()
res = pd.DataFrame(rows)
res.to_csv(m.TABLES/"23_rcv_threshold_flag.csv", index=False)

# Workload: linked scans scored under the symmetric 90-day rule.
up, dn = rcv(CVA_PRIMARY)
w = pd.read_parquet(m.DATA/"wide_window_assessments.parquet")
w["flag"] = flag(w, up, dn)
w["nonconcurrent"] = w.lag.ne(0)
workload = {"scored_scans": int(len(w)), "flagged": int(w.flag.sum()),
            "nonconcurrent_scans": int(w.nonconcurrent.sum()), "nonconcurrent_flagged": int((w.flag & w.nonconcurrent).sum()),
            "flagged_by_category": w[w.flag].category.value_counts().to_dict(),
            "scans_by_category": w.category.value_counts().to_dict()}
# Same-day categories stable under the RCV (analogue of the fixed ±25% check).
sd = w[w.lag.eq(0)].copy()
stable = []
for cat, g in sd.groupby("category"):
    keep = (m.category(m.fast(g.lsm, g.cap, g.AST*up)) == cat) & (m.category(m.fast(g.lsm, g.cap, g.AST*dn)) == cat)
    g = g.assign(stable=keep)
    stable.append(prop(g, "stable", "same_day_stable_within_RCV", category=cat))
pd.DataFrame(stable).to_csv(m.TABLES/"24_same_day_stability_rcv.csv", index=False)
um, dm = rcv(CVA_PRIMARY, CVI_MASLD)
fm = flag(w, um, dm)
workload_masld = {"flagged": int(fm.sum()), "nonconcurrent_flagged": int((fm & w.nonconcurrent).sum())}
summary = {"CVI": CVI, "CVA_primary": CVA_PRIMARY, "rcv_up": up-1, "rcv_down": 1-dn, "workload": workload,
           "masld_wcv": {"total_wCV": WCV_MASLD, "CVI": round(CVI_MASLD, 2), "rcv_up": um-1, "rcv_down": 1-dm, "workload": workload_masld}}
(m.TABLES/"rcv_summary.json").write_text(json.dumps(summary, indent=2))
print(res.round(3).to_string()); print(pd.DataFrame(stable).round(3).to_string()); print(json.dumps(summary, indent=2))
