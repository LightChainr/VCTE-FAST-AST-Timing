"""Link VCTE patients to two laboratory sources and freeze identifier-free laboratory events (v13).

v13: before mid-2024 the elastography device issued a new 11-digit record number at every visit, so one
person can hold several record numbers. A laboratory identity may therefore link to more than one record
number when those records share birth date and sex and their recorded heights differ by at most 3 cm.
Record numbers linked to the same laboratory person form one analysis person (person_token = the
research token of that person's earliest record; no new identifier or hash is created).

Source 1: the existing warehouse LIS layer (RT1 tokens), linked as in v11.
Source 2: the 2026-10-03 liver-related LIS export (direct identifiers, controlled area only).
  - Patients already linked to source 1 are joined to source 2 deterministically by national ID.
  - Other patients are linked by exact birth date + sex + a same-department laboratory encounter
    within ±1 day of a scan, unique in both directions.
Only events re-keyed to existing VCTE research tokens leave the controlled area. No identifier, barcode
or new patient hash is written; the identity layer is read only.
"""
from pathlib import Path
import json
import re
import sqlite3

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WAREHOUSE = Path(__import__("os").environ["VCTE_FAST_WAREHOUSE"])  # institutional research data repository (not public)
DB = WAREHOUSE / "04_分析就绪数据/unified_analysis.sqlite"
IDENTITY = WAREHOUSE / "02_受控身份数据/identity_master.sqlite"
V8 = WAREHOUSE / "03_冻结分析数据/DS-SHARED-VCTE-LIS-V8-20260721/vcte_v8.parquet"
NEW_RAW = WAREHOUSE / "01_院内原始数据/DS-SHARED-LIS-LIVER-MIXED-20261003/staging_all_rows.parquet"
DATA = WAREHOUSE / "03_冻结分析数据/DS-VCTE-FAST-V13-20261003"
TABLES = HERE / "results"
LAB_END = pd.Timestamp("2026-08-13 23:59:59")
NULL_SHIFTS = (1, 7, -3, 30, -60)
HEIGHT_TOLERANCE_CM = 3.0
DEPT_KEYS = ["消化", "感染", "内分泌", "肝胆", "全科", "体检", "心", "肾", "风湿", "呼吸", "中医", "急诊",
             "妇", "老年", "血液", "肿瘤", "普外", "外科", "神经", "康复", "营养", "减重", "体重"]
ANALYTE = {"AST-J": "AST", "ALT-J": "ALT", "GGT-J": "GGT", "ALP": "ALP", "TBIL-7600": "TBIL", "ALB-J": "ALB",
           "PLT": "PLT", "HBsAg1": "HBsAg", "HBeAg1": "HBeAg", "HBcAb1": "anti_HBc", "HBV-DNA": "HBV_DNA",
           "HCV-AB": "anti_HCV", "GLUZ": "fasting_glucose", "HbA1c": "HbA1c", "TG-J": "TG", "HDL-J": "HDL",
           "LDL-J": "LDL", "TCHOL-J": "TC", "CK": "CK", "CK_MB": "CK_MB", "UA-J": "UA", "CREA-J": "CREA"}


def ro(path):
    return sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)


def dept_core(s):
    if not isinstance(s, str) or not s.strip():
        return None
    return next((k for k in DEPT_KEYS if k in s), "other")


def parse_value(text):
    s = str(text).strip().replace("，", ",")
    m = re.fullmatch(r"([<>≤≥]=?)?\s*([0-9]*\.?[0-9]+(?:[Ee][+-]?\d+)?)", s)
    if not m:
        return np.nan, "text"
    q = m.group(1)
    return float(m.group(2)), ("exact" if not q else ("below_limit" if q[0] in "<≤" else "above_limit"))


def height_consistent(heights):
    h = pd.Series(heights, dtype=float)
    return len(h) <= 1 or (h.notna().all() and h.max() - h.min() <= HEIGHT_TOLERANCE_CM)


def link(v, enc, key, tolerance=1, department=True):
    m = v[["patient_id", "scan_date", "bd", "sex", "dc"]].merge(enc[[key, "date", "bd", "sex", "dc"]], on=["bd", "sex"], suffixes=("_v", "_l"))
    m = m[(m.date - m.scan_date).dt.days.abs() <= tolerance]
    if department:
        m = m[m.dc_v.notna() & (m.dc_v == m.dc_l)]
    ev = m.groupby(["patient_id", key]).date.nunique().rename("evidence_days").reset_index()
    a = ev.groupby("patient_id")[key].nunique()
    ev = ev[ev.patient_id.map(a).eq(1)].copy()          # each record number: one laboratory identity
    ev["h"] = ev.patient_id.map(v.groupby("patient_id").h.median())
    ok = ev.groupby(key).h.agg(lambda x: height_consistent(x.tolist()))  # one identity: records of one person
    return ev[ev[key].map(ok)].drop(columns="h")


def quality(v, enc, key, gold, tolerance, department):
    ev = link(v, enc, key, tolerance, department)
    g = ev[ev.patient_id.isin(gold)]
    nulls = []
    for shift in NULL_SHIFTS:
        w = v.copy(); w["bd"] = w.bd + pd.Timedelta(days=shift)
        nulls.append(len(link(w, enc, key, tolerance, department)))
    agree = [p in gold[i] if isinstance(gold[i], set) else p == gold[i] for i, p in zip(g.patient_id, g[key])]
    return ev, {"tolerance_days": tolerance, "department_match": department, "linked_patients": len(ev),
                "reference_patients": len(gold), "reference_recovered": len(g),
                "reference_agreement": float(np.mean(agree)) if agree else np.nan,
                "null_links_mean": float(np.mean(nulls)), "estimated_false_link_fraction": float(np.mean(nulls) / len(ev))}


def main():
    TABLES.mkdir(exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    # VCTE side
    v = pd.read_parquet(V8)
    db = ro(DB)
    s = pd.read_sql_query("SELECT scan_token,research_token,scan_timestamp,lsm,cap FROM vcte_scans", db)
    s["scan_timestamp"] = pd.to_datetime(s.scan_timestamp); v["scan_timestamp"] = pd.to_datetime(v.scan_timestamp)
    for d, l, c in [(s, "lsm", "cap"), (v, "LSM", "CAP")]:
        d["k"] = d.scan_timestamp.astype(str) + "|" + d[l].round(4).astype(str) + "|" + d[c].round(3).astype(str)
    v = v.merge(s[["k", "scan_token", "research_token"]], on="k", validate="one_to_one")
    v = v[(v.qc_pass == 1) & (v.adult_at_scan == 1)].copy()
    assert len(v) == 9562 and v.research_token.nunique() == 8079
    v["bd"] = pd.to_datetime(v.birth_date).dt.normalize(); v["scan_date"] = pd.to_datetime(v.scan_date).dt.normalize()
    v["dc"] = v.department.map(dept_core)
    v["h"] = pd.to_numeric(v.height_cm, errors="coerce")
    H = v.groupby("patient_id").h.median()
    # Source 1: warehouse LIS identities
    enc1 = pd.read_sql_query("SELECT DISTINCT research_token lab_token,event_date,department FROM lis_observations", db)
    db.close()
    enc1["date"] = pd.to_datetime(enc1.event_date, errors="coerce").dt.normalize()
    enc1 = enc1.dropna(subset=["date"]); enc1["dc"] = enc1.department.map(dept_core)
    idc = ro(IDENTITY)
    mp = pd.read_sql_query("SELECT master_person_uid,research_token lab_token,canonical_birth_date bd,canonical_sex sex FROM master_persons", idc)
    gold = pd.read_sql_query("""SELECT b.source_key_value patient_id,p.research_token lab_token FROM legacy_key_bridge b
        JOIN master_persons p USING(master_person_uid) WHERE b.source_system LIKE '%VCTE%' AND b.linkage_method='high_confidence_lis_fingerprint'""", idc)
    nids = pd.read_sql_query("SELECT identifier_value nid,master_person_uid FROM identifiers WHERE identifier_type='national_id'", idc)
    idc.close()
    mp["bd"] = pd.to_datetime(mp.bd, errors="coerce").dt.normalize()
    enc1 = enc1.merge(mp[["lab_token", "bd", "sex"]], on="lab_token")
    gold = dict(zip(gold[gold.patient_id.isin(v.patient_id)].patient_id, gold[gold.patient_id.isin(v.patient_id)].lab_token))
    rows = []
    maps1 = {}
    for tol in (0, 1, 3):
        ev, r = quality(v, enc1, "lab_token", gold, tol, True); r["source"] = "warehouse_LIS"; rows.append(r); maps1[tol] = ev
    patients = v.drop_duplicates("patient_id")[["patient_id", "research_token"]].copy()
    p1 = dict(zip(maps1[1].patient_id, maps1[1].lab_token))
    p1.update(gold)
    gold_holders = {}
    for p, t in gold.items():
        gold_holders.setdefault(t, []).append(p)
    p1 = {p: t for p, t in p1.items() if p in gold or t not in gold_holders
          or height_consistent([H.get(p)] + [H.get(g) for g in gold_holders[t]])}
    patients["lab_token"] = patients.patient_id.map(p1)
    patients["source1_method"] = np.where(patients.patient_id.isin(gold), "laboratory_fingerprint",
                                          np.where(patients.lab_token.notna(), "dob_sex_department_encounter", "unlinked"))
    patients["source1_strict_agrees"] = patients.patient_id.map(dict(zip(maps1[0].patient_id, maps1[0].lab_token))).eq(patients.lab_token)
    # Source 2: new liver-related LIS export (controlled area)
    raw = pd.read_parquet(NEW_RAW)
    raw["ts"] = pd.to_datetime(raw["审核时间"], errors="coerce")
    raw = raw[raw.ts.notna() & raw.ts.le(LAB_END)].copy()
    nid = raw["身份证号"].str.strip().str.upper()
    nid = nid.where(nid.str.fullmatch(r"\d{17}[\dX]"))
    raw["bd"] = pd.to_datetime(raw["出生日期"].str[:10], errors="coerce").dt.normalize()
    raw["sex"] = raw["性别"].map({"男": "M", "女": "F"})
    raw["nid"] = nid
    raw["pkey"] = nid.fillna("N:" + raw["姓名"].str.strip() + "|" + raw["出生日期"].str[:10] + "|" + raw["性别"])
    raw["date"] = raw.ts.dt.normalize(); raw["dc"] = raw["申请科室名称"].map(dept_core)
    enc2 = raw[["pkey", "bd", "sex", "date", "dc"]].drop_duplicates()
    # Reference for source-2 validation: national IDs of source-1 links.
    tok_nid = nids.merge(mp[["master_person_uid", "lab_token"]], on="master_person_uid").groupby("lab_token").nid.agg(set)
    ref = {p: tok_nid.get(t) for p, t in p1.items() if isinstance(tok_nid.get(t), set)}
    maps2 = {}
    for tol in (0, 1, 3):
        ev, r = quality(v, enc2, "pkey", ref, tol, True); r["source"] = "liver_LIS_20261003"; rows.append(r); maps2[tol] = ev
    pd.DataFrame(rows).to_csv(TABLES / "00_linkage_quality.csv", index=False)
    # Deterministic bridge for source-1 patients, then rule-based links for the rest.
    raw_nids = set(raw.nid.dropna())
    bridge = {p: next(iter(s & raw_nids)) for p, s in ref.items() if len(s & raw_nids) == 1}
    rule = dict(zip(maps2[1].patient_id, maps2[1].pkey))
    conflict = {p for p, k in rule.items() if p in ref and not k.startswith("N:") and k not in ref[p]}
    p2 = dict(bridge)
    used = {}
    for p, k in bridge.items():
        used.setdefault(k, []).append(p)
    for p, k in rule.items():
        if p not in p2 and p not in conflict and height_consistent([H.get(p)] + [H.get(q) for q in used.get(k, [])]):
            p2[p] = k; used.setdefault(k, []).append(p)
    patients["source2_method"] = np.where(patients.patient_id.isin(bridge), "national_id_bridge",
                                          np.where(patients.patient_id.isin(p2), "dob_sex_department_encounter", "unlinked"))
    patients["source2_strict_agrees"] = patients.patient_id.map(dict(zip(maps2[0].patient_id, maps2[0].pkey))).eq(patients.patient_id.map(p2))
    patients["linked"] = patients.lab_token.notna() | patients.patient_id.isin(p2)
    # Persons: record numbers sharing a laboratory identity in either source (union-find).
    parent = {p: p for p in patients.patient_id}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for mapping in (p1, p2):
        first = {}
        for p, k in mapping.items():
            if k in first:
                parent[find(p)] = find(first[k])
            else:
                first[k] = p
    first_scan = v.groupby("patient_id").scan_timestamp.min()
    root = patients.patient_id.map(find)
    earliest = patients.assign(root=root, t=patients.patient_id.map(first_scan)).sort_values("t").drop_duplicates("root").set_index("root").research_token
    patients["person_token"] = root.map(earliest)
    patients["records_in_person"] = patients.groupby("person_token").patient_id.transform("size")
    # Identifier-free events from source 2.
    pid_token = dict(zip(patients.patient_id, patients.research_token))
    pid_person = dict(zip(patients.patient_id, patients.person_token))
    key_to_vcte = {k: pid_person[p] for p, k in p2.items()}
    ev2 = raw[raw.pkey.isin(key_to_vcte)].copy()
    ev2["research_token"] = ev2.pkey.map(key_to_vcte)
    ev2["test_code"] = ev2["项目代码"].str.strip()
    ev2["analyte"] = ev2.test_code.map(ANALYTE)
    ev2 = ev2[ev2.analyte.notna()].copy()
    parsed = ev2["测定结果"].map(parse_value)
    ev2["value"] = [x[0] for x in parsed]; ev2["qualifier"] = [x[1] for x in parsed]
    # A re-run on the same specimen is the same measurement; keep the final reported row per barcode and test.
    ev2 = ev2.sort_values("ts").drop_duplicates(["research_token", "条码号", "test_code"], keep="last")
    ev2 = ev2.drop_duplicates(["research_token", "test_code", "ts", "value"])
    out_ev = pd.DataFrame({"research_token": ev2.research_token, "analyte": ev2.analyte, "test_code": ev2.test_code,
                           "value": ev2.value, "qualifier": ev2.qualifier, "report_timestamp": ev2.ts,
                           "department": ev2["申请科室名称"], "instrument": ev2["报告仪器"], "source": "liver_LIS_20261003"})
    out_ev = out_ev.sort_values(["research_token", "analyte", "report_timestamp"]).reset_index(drop=True)
    assert not out_ev.astype(str).apply(lambda c: c.str.contains(r"\d{17}[\dXx]", regex=True)).any().any()
    path = DATA / "liver_lis_events.parquet"; out_ev.to_parquet(path, index=False); path.chmod(0o600)
    final = patients.drop(columns="patient_id").rename(columns={"research_token": "vcte_token"})
    path = DATA / "vcte_lis_linkage.parquet"; final.to_parquet(path, index=False); path.chmod(0o600)
    q = pd.DataFrame(rows)
    pr1 = q[(q.source == "warehouse_LIS") & (q.tolerance_days == 1)].iloc[0]
    pr2 = q[(q.source == "liver_LIS_20261003") & (q.tolerance_days == 1)].iloc[0]
    summary = {"eligible_patients": int(len(final)), "linked_patients": int(final.linked.sum()),
               "source1_methods": final.source1_method.value_counts().to_dict(),
               "source2_methods": final.source2_method.value_counts().to_dict(),
               "linked_persons": int(final.loc[final.linked, "person_token"].nunique()),
               "linked_records_in_multi_record_persons": int((final.linked & final.records_in_person.gt(1)).sum()),
               "persons_with_multiple_records": int(final.loc[final.linked & final.records_in_person.gt(1), "person_token"].nunique()),
               "both_sources": int((final.lab_token.notna() & final.source2_method.ne("unlinked")).sum()),
               "source2_only": int((final.lab_token.isna() & final.source2_method.ne("unlinked")).sum()),
               "source2_rule_conflicts_with_source1": len(conflict),
               "source1_rule_reference_agreement": float(pr1.reference_agreement),
               "source1_rule_false_link_fraction": float(pr1.estimated_false_link_fraction),
               "source2_rule_reference_agreement": float(pr2.reference_agreement),
               "source2_rule_reference_recovered": int(pr2.reference_recovered),
               "source2_rule_false_link_fraction": float(pr2.estimated_false_link_fraction),
               "new_export_rows_through_lab_end": int(len(raw)), "new_export_persons": int(raw.pkey.nunique()),
               "new_export_national_id_coverage": float(raw.nid.notna().mean()),
               "new_events_frozen": int(len(out_ev)), "new_AST_events": int((out_ev.analyte == "AST").sum()),
               "laboratory_end": str(LAB_END.date())}
    (TABLES / "00_linkage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(q.round(4).to_string())


if __name__ == "__main__":
    main()
