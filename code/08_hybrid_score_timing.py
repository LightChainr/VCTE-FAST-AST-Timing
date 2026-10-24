"""AST/ALT timing in Agile 3+, Agile 4 and FIB-4 compared with FAST (amendment 29, exploratory).

Reuses the 879 same-day-anchor FAST comparisons of the main analysis. ALT is taken from the same panel as each
AST (identical report timestamp); same-day platelets are held fixed in the primary analysis. Reads the frozen
dataset only.
"""
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
SCORES = ["FAST", "Agile3", "Agile4", "FIB4"]
CUTS = {"FAST": (.35, .67), "Agile3": (.451, .679), "Agile4": (.251, .565), "FIB4": (1.30, 2.67)}


def fast(lsm, cap, ast):
    return expit(-1.65+1.07*np.log(lsm)+2.66e-8*cap**3-63.3/ast)


def agile3(lsm, plt, ast, alt, dm, male, age):
    return expit(-3.92368+2.29714*np.log(lsm)-0.00902*plt-0.98633*(alt/ast)+1.08636*dm-0.38581*male+0.03018*age)


def agile4(lsm, plt, ast, alt, dm, male):
    return expit(7.50139-15.42498/np.sqrt(lsm)-0.01378*plt-1.41149*(alt/ast)-0.53281*male+0.41741*dm)


def fib4(age, ast, alt, plt):
    return age*ast/(plt*np.sqrt(alt))


def category(score, values, age=None):
    low, high = CUTS[score]
    v = np.asarray(values)
    if age is not None:  # age-adapted FIB-4 rule-out at >=65 years
        low = np.where(np.asarray(age) >= 65, 2.0, low)
    # FAST uses <=0.35 for rule-out; Agile and FIB-4 use strict "<" rule-out cut-offs
    lower = v <= low if score == "FAST" else v < low
    return np.select([lower, v >= high], [0, 2], default=1)


def scores(d, ast, alt, plt):
    return {"FAST": fast(d.lsm, d.cap, ast),
            "Agile3": agile3(d.lsm, plt, ast, alt, d.dm, d.male, d.age_at_scan),
            "Agile4": agile4(d.lsm, plt, ast, alt, d.dm, d.male),
            "FIB4": fib4(d.age_at_scan, ast, alt, plt)}


def boot_index(tokens, rng):
    groups = pd.Series(np.arange(len(tokens))).groupby(np.asarray(tokens)).apply(np.array).tolist()
    for _ in range(BOOT):
        yield np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def summarize(d, label):
    """Reclassification per score with patient-cluster CIs; paired FAST-minus differences share the resamples."""
    cols = [f"disc_{s}" for s in SCORES] + ["disc_FIB4_age"]
    x = d[cols].to_numpy(float)
    draws = np.array([x[i].mean(0) for i in boot_index(d.research_token, np.random.default_rng(SEED))])
    est = x.mean(0)
    rows = []
    for j, c in enumerate(cols):
        s = c[5:]
        lo, hi = np.quantile(draws[:, j], [.025, .975])
        diff = draws[:, 0]-draws[:, j]
        dlo, dhi = np.quantile(diff, [.025, .975])
        rows.append({"set": label, "score": s, "comparisons": len(d), "patients": d.research_token.nunique(),
                     "reclassified": int(x[:, j].sum()), "proportion": est[j], "ci_low": lo, "ci_high": hi,
                     "FAST_minus": est[0]-est[j], "FAST_minus_low": dlo, "FAST_minus_high": dhi})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    a = pd.read_parquet(DATA/"same_day_anchor_comparisons.parquet")
    ev = pd.read_parquet(DATA/"ast_events.parquet")
    lis = pd.read_parquet(DATA/"liver_lis_events.parquet")
    lis["ts"] = pd.to_datetime(lis.report_timestamp)
    lis["value"] = pd.to_numeric(lis.value, errors="coerce")
    assert len(a) == 879 and a.research_token.nunique() == 558

    alt = lis[lis.analyte.eq("ALT") & lis.value.gt(0)].groupby(["research_token", "ts"]).value.first()
    plt = lis[lis.analyte.eq("PLT") & lis.value.gt(0)][["research_token", "ts", "value"]]

    # reference (same-day) draw: the anchor AST is recovered from the frozen events by token, scan and value
    anchor = a[["scan_token", "research_token", "scan_timestamp", "reference_AST"]].drop_duplicates("scan_token")
    cand = anchor.merge(ev[["research_token", "available_timestamp", "available_date", "AST", "observation_token"]],
                        left_on=["research_token", "reference_AST"], right_on=["research_token", "AST"])
    cand = cand[cand.available_date.eq(pd.to_datetime(cand.scan_timestamp).dt.normalize())]
    cand["gap"] = (cand.available_timestamp-pd.to_datetime(cand.scan_timestamp)).abs()
    ref = cand.sort_values(["scan_token", "gap", "available_timestamp", "observation_token"]).drop_duplicates("scan_token")
    ref = ref.set_index("scan_token").available_timestamp.rename("ref_ts")
    assert ref.index.nunique() == anchor.scan_token.nunique()

    d = a.join(ref, on="scan_token")
    d["ref_ALT"] = alt.reindex(list(zip(d.research_token, d.ref_ts))).to_numpy()
    d["cmp_ALT"] = alt.reindex(list(zip(d.research_token, d.available_timestamp))).to_numpy()

    def nearest_plt(tokens, times):
        q = pd.DataFrame({"research_token": tokens.to_numpy(), "t": pd.to_datetime(times).to_numpy(), "k": range(len(tokens))})
        m = q.merge(plt, on="research_token")
        m = m[m.ts.dt.normalize().eq(m.t.dt.normalize())]
        m["gap"] = (m.ts-m.t).abs()
        return m.sort_values(["k", "gap", "ts"]).drop_duplicates("k").set_index("k").value.reindex(range(len(tokens))).to_numpy()

    d["ref_PLT"] = nearest_plt(d.research_token, d.scan_timestamp)
    d["cmp_PLT"] = nearest_plt(d.research_token, d.available_timestamp)
    d["dm"] = d.diabetes_any.astype(float)
    d["male"] = d.sex.astype(str).str.lower().isin(["male", "m", "男", "1"]).astype(float)
    assert d.male.between(0, 1).all() and 0 < d.male.mean() < 1

    flow = {"anchor_comparisons": len(a), "with_reference_ALT": int(d.ref_ALT.notna().sum()),
            "with_both_ALT": int((d.ref_ALT.notna() & d.cmp_ALT.notna()).sum())}
    s = d[d.ref_ALT.notna() & d.cmp_ALT.notna() & d.ref_PLT.notna()].copy()
    flow["analysis_set"] = len(s)
    flow["analysis_patients"] = int(s.research_token.nunique())
    flow["secondary_with_comparator_PLT"] = int(s.cmp_PLT.notna().sum())

    R = scores(s, s.reference_AST, s.ref_ALT, s.ref_PLT)
    C = scores(s, s.AST, s.cmp_ALT, s.ref_PLT)
    assert np.allclose(R["FAST"], s.reference_FAST) and np.allclose(C["FAST"], s.FAST)
    for k in SCORES:
        s[f"ref_{k}"], s[f"cmp_{k}"] = R[k], C[k]
        s[f"disc_{k}"] = (category(k, R[k]) != category(k, C[k])).astype(int)
    s["disc_FIB4_age"] = (category("FIB4", R["FIB4"], s.age_at_scan) != category("FIB4", C["FIB4"], s.age_at_scan)).astype(int)

    rows = summarize(s, "primary_all")
    for direction, g in s.groupby("direction"):
        rows += summarize(g, f"primary_{direction}")
    rows += summarize(s[s.MASLD_compatible.astype(bool)], "primary_MASLD_compatible")

    sec = s[s.cmp_PLT.notna()].copy()
    C2 = scores(sec, sec.AST, sec.cmp_ALT, sec.cmp_PLT)
    for k in SCORES:
        sec[f"disc_{k}"] = (category(k, sec[f"ref_{k}"]) != category(k, C2[k])).astype(int)
    sec["disc_FIB4_age"] = (category("FIB4", sec.ref_FIB4, sec.age_at_scan) != category("FIB4", C2["FIB4"], sec.age_at_scan)).astype(int)
    rows += summarize(sec, "secondary_all_labs_nonconcurrent")
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"29_reclassification_by_score.csv", index=False)

    dl = pd.DataFrame({"AST": np.log(s.AST/s.reference_AST), "ALT": np.log(s.cmp_ALT/s.ref_ALT)})
    dl["AST_over_ALT"] = dl.AST-dl.ALT
    dl["AST_over_sqrtALT"] = dl.AST-dl.ALT/2
    mech = {"sd_dlog": dl.std().round(4).to_dict(), "corr_dlogAST_dlogALT": round(float(dl.AST.corr(dl.ALT)), 4),
            "median_abs_change": {k: float(np.median(np.abs((np.log(s[f"cmp_{k}"]/s[f"ref_{k}"]) if k == "FIB4"
                                   else logit(s[f"cmp_{k}"])-logit(s[f"ref_{k}"]))))) for k in SCORES}}

    trans = []
    for k in ["FAST", "Agile3", "Agile4", "FIB4"]:
        rc, cc = category(k, s[f"ref_{k}"]), category(k, s[f"cmp_{k}"])
        not_out = rc > 0
        trans.append({"score": k, "reference_not_rule_out": int(not_out.sum()),
                      "became_rule_out": int((not_out & (cc == 0)).sum()),
                      "proportion": float((cc[not_out] == 0).mean()) if not_out.any() else np.nan,
                      **{f"ref_{n}": int((rc == i).sum()) for i, n in enumerate(["out", "indet", "in"])}})
    pd.DataFrame(trans).to_csv(OUT/"29_false_reassurance_by_score.csv", index=False)

    # Post hoc (added after the primary results were seen): is the lower Agile reclassification due to AST/ALT
    # cancellation or to fewer Agile results lying near its cut-offs? Same score, same comparisons, but only AST
    # taken from the comparator draw while ALT stays at the same-day value.
    C3 = scores(s, s.AST, s.ref_ALT, s.ref_PLT)
    rng = np.random.default_rng(SEED)
    idx = list(boot_index(s.research_token, rng))
    posthoc = []
    for k in ["Agile3", "Agile4", "FIB4"]:
        only = (category(k, s[f"ref_{k}"]) != category(k, C3[k])).astype(int)
        ref = s[f"ref_{k}"].to_numpy()
        both = s[f"disc_{k}"].to_numpy()
        diff = np.array([only[i].mean()-both[i].mean() for i in idx])
        dlo, dhi = np.quantile(diff, [.025, .975])
        width = np.diff(logit(np.array(CUTS[k]))) if k != "FIB4" else np.diff(np.log(CUTS[k]))
        near = np.abs((logit(ref) if k != "FIB4" else np.log(ref))[:, None]
                      - (logit(np.array(CUTS[k])) if k != "FIB4" else np.log(CUTS[k]))[None, :]).min(1)
        posthoc.append({"score": k, "AST_only_changed": only.mean(), "AST_and_ALT_changed": both.mean(),
                        "difference": only.mean()-both.mean(), "diff_low": dlo, "diff_high": dhi,
                        "zone_width_logit": float(width[0]),
                        "median_abs_change_over_zone_width": mech["median_abs_change"][k]/float(width[0]),
                        "reference_within_0.25_zone_of_cutoff": float((near < .25*width[0]).mean())})
    width = float(np.diff(logit(np.array(CUTS["FAST"])))[0])
    near = np.abs(logit(s.ref_FAST.to_numpy())[:, None]-logit(np.array(CUTS["FAST"]))[None, :]).min(1)
    posthoc.append({"score": "FAST", "AST_only_changed": s.disc_FAST.mean(), "AST_and_ALT_changed": np.nan,
                    "zone_width_logit": width, "median_abs_change_over_zone_width": mech["median_abs_change"]["FAST"]/width,
                    "reference_within_0.25_zone_of_cutoff": float((near < .25*width).mean())})
    pd.DataFrame(posthoc).to_csv(OUT/"29_posthoc_ratio_cancellation.csv", index=False)
    print(pd.DataFrame(posthoc).round(3).to_string(index=False))

    p = table.query("set=='primary_all' and score=='Agile3'").iloc[0]
    primary = bool(p.FAST_minus >= .10 and p.FAST_minus_low > 0)
    summary = {"flow": flow, "mechanism": mech, "decision_rule": {"FAST_minus_Agile3": p.FAST_minus,
               "ci": [p.FAST_minus_low, p.FAST_minus_high], "promote_to_main_result": primary},
               "case_mix": {"MASLD_compatible": int(s.MASLD_compatible.astype(bool).sum()),
                            "HBsAg_positive": int(s.HBsAg_positive.astype(bool).sum()),
                            "diabetes": int(s.dm.sum()), "male": int(s.male.sum()),
                            "median_PLT": float(s.ref_PLT.median()), "median_LSM": float(s.lsm.median())}}
    (OUT/"29_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(table.round(3).to_string(index=False))
    print(pd.DataFrame(trans).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
