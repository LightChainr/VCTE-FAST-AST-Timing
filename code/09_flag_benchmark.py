"""Post hoc checks after external review (contract amendment 30): flag benchmark at equal workload, chemistry line,
selection into paired comparisons, and the prospective/retrospective reporting scenario. Reads the frozen dataset only."""
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
WAREHOUSE = Path(os.environ["VCTE_FAST_WAREHOUSE"])  # institutional research data repository (not public)
DATA = WAREHOUSE/"03_冻结分析数据/DS-VCTE-FAST-V13-20261003"
OUT = HERE/"results"
SEED = 20261024
BOOT = 2000
CUTS = (0.35, 0.67)
RCV = json.loads((OUT/"rcv_summary.json").read_text())
UP, DN = 1 + RCV["rcv_up"], 1 - RCV["rcv_down"]   # primary RCV limits (02b_rcv_threshold.py)


def eta(lsm, cap, ast):
    return -1.65 + 1.07*np.log(lsm) + 2.66e-8*np.asarray(cap)**3 - 63.3/np.asarray(ast)


def log_margins(lsm, cap, ast):
    """log(required AST / AST in hand) for 0.35 and 0.67; NaN where the threshold is unreachable (as in 02b)."""
    b = -1.65 + 1.07*np.log(lsm) + 2.66e-8*np.asarray(cap)**3
    out = []
    for c in CUTS:
        reach = b > logit(c)
        ac = np.where(reach, 63.3/np.where(reach, b - logit(c), 1.0), np.nan)
        out.append(np.log(ac/np.asarray(ast)))
    return np.column_stack(out)


def groups(tokens):
    return pd.Series(np.arange(len(tokens))).groupby(np.asarray(tokens)).apply(np.array).tolist()


def resamples(tokens):
    g = groups(tokens); rng = np.random.default_rng(SEED)
    return [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]) for _ in range(BOOT)]


def metrics(f, y):
    f, y = np.asarray(f, bool), np.asarray(y, bool)
    return {"flagged": int(f.sum()), "flagged_share": f.mean(), "sensitivity": f[y].mean(), "specificity": (~f[~y]).mean(),
            "ppv": y[f].mean(), "npv": (~y[~f]).mean(), "tp": int((f & y).sum()), "fp": int((f & ~y).sum()),
            "fn": int((~f & y).sum()), "tn": int((~f & ~y).sum())}


def matched(score, k):
    """Flag the k comparisons with the smallest score (ties broken by position, which is fixed by the sorted input)."""
    order = np.argsort(np.asarray(score, float), kind="stable")
    f = np.zeros(len(order), bool); f[order[:k]] = True
    return f


def main():
    a = pd.read_parquet(DATA/"same_day_anchor_comparisons.parquet").sort_values(["scan_token", "direction"]).reset_index(drop=True)
    y = a.discordant.astype(bool).to_numpy()
    D = log_margins(a.lsm.to_numpy(), a.cap.to_numpy(), a.AST.to_numpy())
    with np.errstate(invalid="ignore"):
        rcv_flag = ((D > np.log(DN)) & (D < np.log(UP))).any(axis=1)
    k = int(rcv_flag.sum()); assert k == 421, k
    e = eta(a.lsm.to_numpy(), a.cap.to_numpy(), a.AST.to_numpy())
    rules = {
        "RCV flag (primary)": rcv_flag,
        "FAST distance, probability scale": matched(np.min(np.abs(expit(e)[:, None] - np.array(CUTS)), axis=1), k),
        "FAST distance, logit scale": matched(np.min(np.abs(e[:, None] - logit(np.array(CUTS))), axis=1), k),
        "Fixed symmetric AST percentage": matched(np.nanmin(np.where(np.isnan(D), np.inf, np.abs(D)), axis=1), k),
        "Nonconcurrent indeterminate": (a.category == "indeterminate").to_numpy(),
    }
    idx = resamples(a.research_token)
    rows = []
    for name, f in rules.items():
        r = metrics(f, y)
        sens = np.array([f[i][y[i]].mean() for i in idx]); diff = np.array([rcv_flag[i][y[i]].mean() - f[i][y[i]].mean() for i in idx])
        r.update(rule=name, sens_low=np.quantile(sens, .025), sens_high=np.quantile(sens, .975),
                 rcv_minus_sens=rcv_flag[y].mean() - r["sensitivity"],
                 rcv_minus_low=np.quantile(diff, .025), rcv_minus_high=np.quantile(diff, .975),
                 agreement_with_rcv=float((f == rcv_flag).mean()))
        rows.append(r)
    bench = pd.DataFrame(rows)[["rule", "flagged", "flagged_share", "tp", "fp", "fn", "tn", "sensitivity", "sens_low", "sens_high",
                                "specificity", "ppv", "npv", "rcv_minus_sens", "rcv_minus_low", "rcv_minus_high", "agreement_with_rcv"]]
    bench.to_csv(OUT/"30_flag_benchmark.csv", index=False)

    # (b) chemistry line of each AST: identical patient token and report time in the liver LIS export.
    lis = pd.read_parquet(DATA/"liver_lis_events.parquet", columns=["research_token", "analyte", "report_timestamp", "instrument"])
    lis = lis[lis.analyte.eq("AST")]; lis["ts"] = pd.to_datetime(lis.report_timestamp)
    line = lis.dropna(subset=["instrument"]).groupby(["research_token", "ts"]).instrument.first()
    ev = pd.read_parquet(DATA/"ast_events.parquet")
    ref = (a[["scan_token", "research_token", "scan_timestamp", "reference_AST"]].drop_duplicates("scan_token")
           .merge(ev[["research_token", "available_timestamp", "available_date", "AST", "observation_token"]],
                  left_on=["research_token", "reference_AST"], right_on=["research_token", "AST"]))
    ref = ref[ref.available_date.eq(pd.to_datetime(ref.scan_timestamp).dt.normalize())]
    ref["gap"] = (ref.available_timestamp - pd.to_datetime(ref.scan_timestamp)).abs()
    ref = ref.sort_values(["scan_token", "gap", "available_timestamp", "observation_token"]).drop_duplicates("scan_token").set_index("scan_token")
    a["ref_line"] = line.reindex(list(zip(a.research_token, ref.available_timestamp.reindex(a.scan_token)))).to_numpy()
    a["cmp_line"] = line.reindex(list(zip(a.research_token, a.available_timestamp))).to_numpy()
    b = a.dropna(subset=["ref_line", "cmp_line"]).copy()
    b["same_line"] = b.ref_line.eq(b.cmp_line)
    lrows = []
    for lab, g in [("same line", b[b.same_line]), ("different lines", b[~b.same_line])]:
        gi = resamples(g.research_token); v = g.discordant.to_numpy(float)
        ci = np.quantile([v[i].mean() for i in gi], [.025, .975])
        lrows.append({"group": lab, "comparisons": len(g), "patients": g.research_token.nunique(), "reclassified": int(v.sum()),
                      "proportion": v.mean(), "ci_low": ci[0], "ci_high": ci[1],
                      "median_abs_log_AST_ratio": float(np.median(np.abs(np.log(g.AST/g.reference_AST))))})
    pd.DataFrame(lrows).to_csv(OUT/"30_chemistry_line.csv", index=False)

    # (c) selection: scans with same-day AST, in or not in a paired comparison.
    w = pd.read_parquet(DATA/"wide_window_assessments.parquet")
    sd = w[w.lag.eq(0)].copy(); sd["paired"] = sd.scan_token.isin(set(a.scan_token))
    srows = []
    for lab, g in [("paired", sd[sd.paired]), ("not paired", sd[~sd.paired])]:
        q = lambda c: f"{g[c].median():.1f} ({g[c].quantile(.25):.1f}–{g[c].quantile(.75):.1f})"
        srows.append({"group": lab, "scans": len(g), "patients": g.research_token.nunique(), "same_day_AST": q("AST"), "LSM": q("lsm"), "CAP": q("cap"),
                      "AST_gt40": float((g.AST > 40).mean()), "MASLD_compatible": float(g.MASLD_compatible.astype(bool).mean()),
                      "diabetes": float(g.diabetes_any.astype(bool).mean()), "HBsAg_positive": float(g.HBsAg_positive.astype(bool).mean()),
                      **{f"category_{c}": float(g.category.eq(c).mean()) for c in ("rule_out", "indeterminate", "rule_in")}})
    pd.DataFrame(srows).to_csv(OUT/"30_selection_paired.csv", index=False)

    # (d) reporting scenario under the +/-90-day rule.
    W = log_margins(w.lsm.to_numpy(), w.cap.to_numpy(), w.AST.to_numpy())
    with np.errstate(invalid="ignore"):
        wf = ((W > np.log(DN)) & (W < np.log(UP))).any(axis=1)
    assert int((wf & w.lag.ne(0)).sum()) == RCV["workload"]["nonconcurrent_flagged"]
    scen = {"scored_scans": int(len(w)), "nonconcurrent_scans": int(w.lag.ne(0).sum()),
            "earlier_scored": int(w.lag.lt(0).sum()), "earlier_flagged": int((wf & w.lag.lt(0)).sum()),
            "later_scored": int(w.lag.gt(0).sum()), "later_flagged": int((wf & w.lag.gt(0)).sum())}
    (OUT/"30_summary.json").write_text(json.dumps({"benchmark_k": k, "scenario": scen,
        "chemistry_line_known": int(len(b)), "chemistry_line_total": int(len(a))}, indent=2))
    print(bench.round(3).to_string(index=False)); print(pd.DataFrame(lrows).round(3).to_string(index=False))
    print(pd.DataFrame(srows).to_string(index=False)); print(json.dumps(scen, indent=2))


if __name__ == "__main__":
    main()
