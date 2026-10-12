"""FAST with dated AST in the laboratory-linked VCTE population (v13: warehouse LIS plus the 2026-10-03 liver LIS export).

Differences from v10: laboratory data are drawn for every VCTE patient linked by 00_link_vcte_lis.py rather
than only for the 372 patients with laboratory-fingerprint links. Laboratory etiology markers (HBsAg,
anti-HCV) and metabolic laboratory values are added. Availability is described inside the linked
population only; unlinked patients are not treated as untested.
"""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
WAREHOUSE = Path(__import__("os").environ["VCTE_FAST_WAREHOUSE"])  # institutional research data repository (not public)
DB = WAREHOUSE / "04_分析就绪数据/unified_analysis.sqlite"
REGISTRY = WAREHOUSE / "00_目录数据库/unified_research_data.sqlite"
DATASET = "DS-VCTE-FAST-V13-20261003"
DATA = WAREHOUSE / "03_冻结分析数据" / DATASET
TABLES = HERE / "results"
SEED = 20261003
BOOT = 2000
LAB_END = pd.Timestamp("2026-08-13")
SCAN_END = pd.Timestamp("2026-07-09")
LABELS = ["rule_out", "indeterminate", "rule_in"]
HBSAG_POSITIVE = 0.05   # IU/mL, quantitative chemiluminescent assay lower reporting cutoff
HCV_POSITIVE = 1.0      # signal-to-cutoff ratio
ETIOLOGY = ["HBsAg_positive", "anti_HCV_positive", "MASLD_compatible", "other_or_undetermined"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def ro(path):
    return sqlite3.connect(Path(path).as_uri()+"?mode=ro", uri=True)


def fast(l, c, a):
    return expit(-1.65+1.07*np.log(l)+2.66e-8*np.asarray(c)**3-63.3/np.asarray(a))


def category(f):
    return np.select([np.asarray(f)<=.35, np.asarray(f)>=.67], LABELS[::2], default=LABELS[1])


def cluster_interval(d, column, metric="mean", seed=SEED):
    if d.empty:
        return np.nan, np.nan
    values = [g[column].to_numpy(float) for _,g in d.groupby("research_token", sort=True)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOT):
        arr = np.concatenate([values[i] for i in rng.integers(0, len(values), len(values))])
        draws.append(np.mean(arr) if metric=="mean" else np.median(arr))
    return tuple(np.quantile(draws,[.025,.975]))


def freeze(df, path):
    df = df.reset_index(drop=True)
    if path.exists():
        pd.testing.assert_frame_equal(df,pd.read_parquet(path),check_dtype=False)
    else:
        df.to_parquet(path,index=False)
        path.chmod(0o600)


def history_flags(text):
    text = str(text or "")
    if any(x in text for x in ("待查","排除","疑似","筛查","?","？")):
        return {}
    patterns = {
        "fatty_history":r"脂肪肝|脂肪性肝|MASLD|NAFLD|MASH|NASH",
        "diabetes_history":r"糖尿病|diabetes",
        "viral_history":r"乙肝|乙型.*肝炎|丙肝|丙型.*肝炎|病毒性肝炎",
        "alcohol_history":r"酒精性|酒精相关",
        "autoimmune_history":r"自身免疫性肝炎|原发性胆汁|原发性硬化性胆管",
        "drug_history":r"药物性肝|药物.*肝损",
        "nonspecific_liver_history":r"肝功能|肝损|慢性肝病|肝纤维化|肝硬化",
    }
    return {k:bool(re.search(v,text,re.I)) for k,v in patterns.items()}


def load_linkage():
    link = pd.read_parquet(DATA/"vcte_lis_linkage.parquet")
    assert link.vcte_token.is_unique and link.person_token.notna().all()
    return link


def lab_markers(db, lab_to_vcte):
    """Return per-patient viral markers and dated metabolic results, keyed by the VCTE research token."""
    codes = ("HBsAg1","HCV-AB","HbA1c","HBA1C","GLUZ","TG-J","HDL-J","dHDL")
    q = f"""SELECT o.research_token lab_token,o.test_code,o.result_numeric,o.result_qualifier,
            COALESCE(o.report_timestamp,o.event_timestamp) ts FROM lis_observations o JOIN cohort c ON c.research_token=o.research_token
            WHERE o.test_code IN {codes} AND o.result_numeric IS NOT NULL"""
    m = pd.read_sql_query(q, db)
    m["research_token"] = m.lab_token.map(lab_to_vcte)
    m["date"] = pd.to_datetime(m.ts, errors="coerce").dt.normalize()
    m["analyte"] = m.test_code.map({"HBsAg1":"HBsAg","HCV-AB":"anti_HCV","HbA1c":"HbA1c","HBA1C":"HbA1c",
                                     "GLUZ":"fasting_glucose","TG-J":"TG","HDL-J":"HDL","dHDL":"HDL"})
    return m.dropna(subset=["research_token","date"])


def attach_markers(scans, markers):
    """Viral markers: any result in the record period (chronic-state markers). Metabolic: −365 to +90 days."""
    viral = markers[markers.analyte.isin(["HBsAg","anti_HCV"])].groupby(["research_token","analyte"]).result_numeric.max().unstack()
    viral = viral.reindex(columns=["HBsAg","anti_HCV"])
    s = scans.merge(viral, left_on="research_token", right_index=True, how="left")
    s["HBsAg_tested"] = s.HBsAg.notna()
    s["HBsAg_positive"] = s.HBsAg.ge(HBSAG_POSITIVE)
    s["HBsAg_positive_ge1"] = s.HBsAg.ge(1)
    s["anti_HCV_tested"] = s.anti_HCV.notna()
    s["anti_HCV_positive"] = s.anti_HCV.ge(HCV_POSITIVE)
    met = markers[~markers.analyte.isin(["HBsAg","anti_HCV"])].merge(s[["scan_token","research_token","scan_date"]], on="research_token")
    met = met[(met.date-met.scan_date).dt.days.between(-365,90)]
    agg = met.groupby(["scan_token","analyte"]).result_numeric.agg(["max","min"]).unstack()
    def col(stat, a):
        return agg[(stat,a)] if (stat,a) in agg.columns else pd.Series(dtype=float)
    s = s.merge(pd.DataFrame({"HbA1c_max":col("max","HbA1c"),"FPG_max":col("max","fasting_glucose"),
                              "TG_max":col("max","TG"),"HDL_min":col("min","HDL")}), left_on="scan_token", right_index=True, how="left")
    female = s.sex.isin(["F","female","女"])
    s["lab_diabetes"] = s.HbA1c_max.ge(6.5) | s.FPG_max.ge(7.0)
    s["diabetes_any"] = s.lab_diabetes | s.diabetes_history
    glyc = s.HbA1c_max.ge(5.7) | s.FPG_max.ge(5.6) | s.diabetes_history
    lipid = s.TG_max.ge(1.7) | (s.HDL_min.lt(1.0) & ~female) | (s.HDL_min.lt(1.3) & female)
    s["cardiometabolic_criterion"] = s.bmi.ge(23) | glyc | lipid
    s["MASLD_compatible"] = s.cap.ge(275) & s.cardiometabolic_criterion & ~s.HBsAg_positive & ~s.anti_HCV_positive
    s["etiology_group"] = np.select([s.HBsAg_positive, s.anti_HCV_positive, s.MASLD_compatible],
                                    ETIOLOGY[:3], default=ETIOLOGY[3])
    s["viral_markers_negative"] = s.HBsAg_tested & ~s.HBsAg_positive & ~s.anti_HCV_positive
    return s


def construct():
    db=ro(DB)
    scans=pd.read_sql_query("SELECT * FROM vcte_scans WHERE qc_pass=1 AND adult_at_scan=1",db)
    assert len(scans)==9562 and scans.research_token.nunique()==8079
    for col in ("scan_timestamp","scan_date"):
        scans[col]=pd.to_datetime(scans[col])
    link=load_linkage()
    scans=scans.merge(link.rename(columns={"vcte_token":"research_token"}),on="research_token",how="left",validate="many_to_one")
    # Analysis unit is the person: record numbers that share a laboratory identity are one person.
    scans["record_token"]=scans.research_token
    scans["research_token"]=scans.person_token
    s1_strict=scans.source1_method.eq("laboratory_fingerprint")|(scans.lab_token.notna()&scans.source1_strict_agrees.fillna(False).astype(bool))
    s2_strict=(scans.source2_method.eq("national_id_bridge")&s1_strict)|(scans.source2_method.eq("dob_sex_department_encounter")&scans.source2_strict_agrees.fillna(False).astype(bool))
    scans["strict_link"]=(s1_strict|s2_strict)&scans.linked
    scans["method"]=np.select([scans.source1_method.eq("laboratory_fingerprint"),scans.source2_method.eq("national_id_bridge"),
                               scans.lab_token.notna()|scans.source2_method.ne("unlinked")],
                              ["laboratory_fingerprint","rule_plus_national_id","birth_date_sex_encounter"],default="unlinked")
    lab_to_vcte=dict(zip(link.lab_token.dropna(),link.loc[link.lab_token.notna(),"person_token"]))
    assert link.dropna(subset=["lab_token"]).groupby("lab_token").person_token.nunique().eq(1).all()
    newlis=pd.read_parquet(DATA/"liver_lis_events.parquet")
    db.execute("CREATE TEMP TABLE cohort(research_token TEXT PRIMARY KEY)")
    db.executemany("INSERT INTO cohort VALUES (?)",[(x,) for x in lab_to_vcte])
    unit_counts=pd.read_sql_query("SELECT test_code,unit,COUNT(*) n FROM lis_observations WHERE test_code IN ('AST','AST-J') GROUP BY 1,2",db)
    nonblank=unit_counts[unit_counts.unit.fillna("").str.strip().ne("")]
    assert set(nonblank.unit.str.upper().str.strip())=={"U/L"}
    unit_counts.to_csv(TABLES/"unit_code_inventory.csv",index=False)
    new=pd.read_sql_query("""SELECT o.observation_token,o.research_token,o.event_timestamp,
      o.collection_timestamp,o.report_timestamp,o.test_code,o.result_numeric,o.result_qualifier,
      o.unit,o.diagnosis,o.department,o.specimen_token,o.instrument
      FROM lis_observations o JOIN cohort c USING(research_token)
      WHERE o.test_code IN ('AST','AST-J') AND o.result_numeric>0 AND o.result_qualifier='exact'""",db)
    old=pd.read_sql_query("""SELECT l.linked_lis_event_token observation_token,l.research_token,
      l.event_timestamp,l.collection_timestamp,l.value_numeric result_numeric,l.unit,l.diagnosis,
      l.department FROM linked_lis_events l JOIN cohort c USING(research_token)
      WHERE l.analyte='AST' AND l.value_numeric>0 AND l.value_qualifier='exact'""",db)
    for d in (old,new):
        for col in ("event_timestamp","collection_timestamp"):
            d[col]=pd.to_datetime(d[col],errors="coerce")
        d["vkey"]=d.result_numeric.round(6)
    new["report_timestamp"]=pd.to_datetime(new.report_timestamp,errors="coerce")
    new["available_timestamp"]=new.report_timestamp.fillna(new.event_timestamp)
    old["available_timestamp"]=old.event_timestamp
    old["unit_status"]="frozen_explicit"
    old["source_kind"]="linked"
    new["source_kind"]="expanded"
    matched=new.merge(old[["research_token","available_timestamp","vkey"]].drop_duplicates(),
                      on=["research_token","available_timestamp","vkey"],how="left",indicator=True)
    missing=matched.unit.fillna("").str.strip().eq("")
    matched["unit_status"]=np.select([~missing,missing & matched._merge.eq("both")],
                                      ["direct","exact_overlap_recovered"],default="code_harmonized")
    overlaps=int((missing & matched._merge.eq("both")).sum())
    events=pd.concat([old,matched.drop(columns="_merge")],ignore_index=True)
    events["research_token"]=events.research_token.map(lab_to_vcte)
    na=newlis[newlis.analyte.eq("AST")&newlis.qualifier.eq("exact")&newlis.value.gt(0)].copy()
    na=pd.DataFrame({"observation_token":"LIV-"+pd.Series(range(len(na)),index=na.index).astype(str),"research_token":na.research_token,
                     "event_timestamp":na.report_timestamp,"available_timestamp":na.report_timestamp,"collection_timestamp":pd.NaT,
                     "result_numeric":na.value,"vkey":na.value.round(6),"unit":None,"department":na.department,
                     "unit_status":"export_without_unit","source_kind":"liver_export_20261003"})
    events=pd.concat([events,na],ignore_index=True)
    events["minute"]=pd.to_datetime(events.available_timestamp).dt.floor("min")
    events=events[events.available_timestamp.notna() & events.available_timestamp.le(LAB_END+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1))].copy()
    events["unit_rank"]=events.unit_status.map({"frozen_explicit":0,"direct":1,"exact_overlap_recovered":2,"code_harmonized":3,"export_without_unit":4})
    events=events.sort_values(["unit_rank","observation_token"])
    before=len(events)
    events=events.drop_duplicates(["research_token","minute","vkey"])
    assert events.research_token.notna().all()
    events["AST"]=events.result_numeric
    events["available_date"]=events.available_timestamp.dt.normalize()
    events["collection_date"]=events.collection_timestamp.dt.normalize()
    events["strict_unit"]=events.unit_status.isin(["frozen_explicit","direct","exact_overlap_recovered"])
    dx_old=pd.read_sql_query("SELECT l.research_token,l.event_timestamp diagnosis_timestamp,l.diagnosis FROM linked_lis_events l JOIN cohort c USING(research_token) WHERE l.diagnosis IS NOT NULL",db)
    dx_new=pd.read_sql_query("""SELECT o.research_token,COALESCE(o.report_timestamp,o.event_timestamp) diagnosis_timestamp,o.diagnosis
      FROM lis_observations o JOIN cohort c USING(research_token) WHERE NULLIF(TRIM(o.diagnosis),'') IS NOT NULL""",db)
    dx=pd.concat([dx_old,dx_new],ignore_index=True).drop_duplicates()
    dx["research_token"]=dx.research_token.map(lab_to_vcte)
    dx["diagnosis_date"]=pd.to_datetime(dx.diagnosis_timestamp,errors="coerce").dt.normalize()
    flag_data=dx.diagnosis.map(history_flags).apply(pd.Series).fillna(False)
    flag_cols=list(flag_data.columns)
    dx=pd.concat([dx.reset_index(drop=True),flag_data.reset_index(drop=True)],axis=1)
    dx_by={p:g for p,g in dx.groupby("research_token")}
    histories=[]
    for s in scans.itertuples():
        g=dx_by.get(s.research_token)
        g=g[g.diagnosis_date.le(s.scan_date)] if g is not None else pd.DataFrame()
        histories.append({"scan_token":s.scan_token,"diagnosis_record_available":not g.empty,
                          **{k:bool(g[k].any()) if not g.empty else False for k in flag_cols}})
    scans=scans.merge(pd.DataFrame(histories),on="scan_token",validate="one_to_one")
    markers=lab_markers(db,lab_to_vcte)
    nm=newlis[newlis.analyte.isin(["HBsAg","anti_HCV","HbA1c","fasting_glucose","TG","HDL"])&newlis.value.notna()]
    nm=pd.DataFrame({"research_token":nm.research_token,"analyte":nm.analyte,"result_numeric":nm.value,
                     "date":pd.to_datetime(nm.report_timestamp).dt.normalize()})
    nm=nm[nm.date.le(LAB_END)]
    markers=pd.concat([markers[["research_token","analyte","result_numeric","date"]],nm],ignore_index=True)
    # Interval from specimen collection to report sign-off, from results that record both (all patients).
    ct=pd.read_sql_query("""SELECT collection_timestamp c,report_timestamp r FROM lis_observations
        WHERE test_code IN ('AST','AST-J') AND collection_timestamp IS NOT NULL AND report_timestamp IS NOT NULL""",db)
    ct=ct.assign(c=pd.to_datetime(ct.c,errors="coerce"),r=pd.to_datetime(ct.r,errors="coerce")).dropna()
    hrs=(ct.r-ct.c).dt.total_seconds()/3600
    timing_check={"AST_results_with_both_times":int(len(ct)),"median_hours":float(hrs.median()),
                  "p25_hours":float(hrs.quantile(.25)),"p75_hours":float(hrs.quantile(.75)),"p95_hours":float(hrs.quantile(.95)),
                  "within_24h":float(hrs.le(24).mean()),"same_calendar_day":float((ct.r.dt.normalize()==ct.c.dt.normalize()).mean()),
                  "negative":int(hrs.lt(0).sum())}
    pd.DataFrame({"hours_bin_start":np.arange(0,48,1)}).assign(
        n=np.histogram(hrs.clip(upper=47.99),bins=np.arange(0,49,1))[0]).to_csv(TABLES/"21_collection_to_report_hours.csv",index=False)
    inst=pd.read_parquet(DATA/"liver_lis_events.parquet",columns=["analyte","instrument"]).query("analyte=='AST'").instrument.value_counts()
    db.close()
    scans["metabolic_steatosis"]=(scans.cap>=275)&(scans.bmi>=23)
    scans["metabolic_steatosis_248"]=(scans.cap>=248)&(scans.bmi>=23)
    scans["BMI_ge25"]=scans.bmi>=25
    scans=attach_markers(scans,markers)
    for col in ("HBsAg_positive","anti_HCV_positive","MASLD_compatible","viral_markers_negative","lab_diabetes","diabetes_any"):
        scans.loc[~scans.linked,col]=False
    scans.loc[~scans.linked,"etiology_group"]="not_linked"
    first=scans.sort_values(["scan_timestamp","scan_token"]).drop_duplicates("research_token")
    fl=first[first.linked]
    audit={"recorded_unit_AST":int(nonblank.loc[nonblank.test_code.eq('AST'),'n'].sum()),
           "recorded_unit_AST_J":int(nonblank.loc[nonblank.test_code.eq('AST-J'),'n'].sum()),
           "overlap_matched_missing_unit_AST_in_scanned_patients":overlaps,
           "cross_export_copies_removed":before-len(events),
           "retained_AST_events":len(events),"AST_patients":int(events.research_token.nunique()),
           "unit_status_counts":events.unit_status.value_counts().to_dict(),
           "collection_timestamp_events":int(events.collection_timestamp.notna().sum()),
           "report_date_definition":"result availability (report timestamp; event timestamp when absent)",
           "unit_definition":"U/L direct, exact-overlap recovery, or code-harmonized within the same LIS AST test codes; explicit-unit sensitivity provided",
           "scan_end":"2026-07-09","laboratory_end":"2026-08-13",
           "linked_patients":int(fl.research_token.nunique()),"linked_scans":int(scans.linked.sum()),
           "linkage_methods":first.method.value_counts().to_dict(),"source_counts":events.source_kind.value_counts().to_dict(),
           "first_patient_recorded_history":{k:int(first[k].sum()) for k in ["diagnosis_record_available",*flag_cols]},
           "linked_first_scan_markers":{"HBsAg_tested":int(fl.HBsAg_tested.sum()),"HBsAg_positive":int(fl.HBsAg_positive.sum()),
                "HBsAg_positive_ge1":int(fl.HBsAg_positive_ge1.sum()),"anti_HCV_tested":int(fl.anti_HCV_tested.sum()),
                "anti_HCV_positive":int(fl.anti_HCV_positive.sum()),"lab_diabetes":int(fl.lab_diabetes.sum()),
                "diabetes_any":int(fl.diabetes_any.sum()),"MASLD_compatible":int(fl.MASLD_compatible.sum()),
                "viral_markers_negative":int(fl.viral_markers_negative.sum()),
                "etiology_groups":fl.etiology_group.value_counts().to_dict()},
           "eligible_scans":len(scans),"eligible_patients":int(scans.research_token.nunique()),
           "eligible_record_numbers":int(scans.record_token.nunique()),
           "linked_record_numbers":int(scans.loc[scans.linked,"record_token"].nunique()),
           "linked_persons_with_multiple_records":int(scans[scans.linked].groupby("research_token").record_token.nunique().gt(1).sum()),
           "collection_to_report":timing_check,"liver_export_AST_instrument":inst.to_dict()}
    DATA.mkdir(parents=True,exist_ok=True)
    keep=[c for c in scans.columns if c not in ("lab_token",)]
    freeze(scans[keep],DATA/"scans.parquet")
    evcols=["observation_token","research_token","available_timestamp","available_date","collection_timestamp",
            "collection_date","AST","unit_status","strict_unit","source_kind"]
    freeze(events[evcols].sort_values(["research_token","available_timestamp","observation_token"]),DATA/"ast_events.parquet")
    return scans[keep],events[evcols],audit


def pairs(scans,events,date_col="available_date"):
    e=events.dropna(subset=[date_col])
    d=scans.merge(e,on="research_token",how="inner",validate="many_to_many")
    d["lag"]=(d[date_col]-d.scan_date).dt.days
    d=d[d.lag.between(-90,90)].copy()
    d["abs_lag"]=d.lag.abs()
    d["hours"]=(d.available_timestamp-d.scan_timestamp).dt.total_seconds()/3600
    d["abs_hours"]=d.hours.abs()
    d["tie"]=np.select([d.lag.eq(0),d.lag.lt(0)],[0,1],default=2)
    d["FAST"]=fast(d.lsm,d.cap,d.AST)
    d["category"]=category(d.FAST)
    return d


def select(d,mask,same=False):
    order=["scan_token","abs_hours","available_timestamp","observation_token"] if same else ["scan_token","abs_lag","tie","available_timestamp","observation_token"]
    return d[mask].sort_values(order).drop_duplicates("scan_token").copy()


def rules(d):
    out={"prior_90":select(d,d.lag.lt(0)),"same_day":select(d,d.lag.eq(0),True),
         "later_90":select(d,d.lag.gt(0)),"available_at_scan":select(d,d.hours.le(0))}
    for w in (1,3,7,14,30,60,90):
        out[f"symmetric_{w}"]=select(d,d.abs_lag.le(w))
    return out


def support(selections,label,denominator):
    rows=[]
    for r,d in selections.items():
        rows.append({"dataset":label,"rule":r,"scans":len(d),"denominator_scans":denominator,"patients":d.research_token.nunique(),
          "unique_AST":d.observation_token.nunique(),"later_n":int(d.lag.gt(0).sum()),
          "reused_n":int(d.observation_token.duplicated(False).sum()),"median_lag":d.abs_lag.median(),
          **{c:int(d.category.eq(c).sum()) for c in LABELS}})
    return rows


def anchor_comparisons(sel):
    anchor=sel["same_day"].set_index("scan_token")
    out=[]
    for direction,r in [("earlier","prior_90"),("later","later_90")]:
        remote=sel[r].set_index("scan_token")
        ids=anchor.index.intersection(remote.index)
        d=remote.loc[ids].copy()
        d["direction"]=direction
        d["reference_AST"]=anchor.loc[ids,"AST"]
        d["reference_FAST"]=anchor.loc[ids,"FAST"]
        d["reference_category"]=anchor.loc[ids,"category"]
        d["delta_log_AST"]=np.log(d.AST/d.reference_AST)
        d["delta_FAST"]=d.FAST-d.reference_FAST
        d["discordant"]=d.category.ne(d.reference_category).astype(int)
        codes={k:i for i,k in enumerate(LABELS)}
        d["upward"]=d.category.map(codes).gt(d.reference_category.map(codes)).astype(int)
        d["downward"]=d.category.map(codes).lt(d.reference_category.map(codes)).astype(int)
        d["boundary_distance"]=np.minimum(np.abs(d.FAST-.35),np.abs(d.FAST-.67))
        d["reference_boundary_distance"]=np.minimum(np.abs(d.reference_FAST-.35),np.abs(d.reference_FAST-.67))
        out.append(d.reset_index())
    return pd.concat(out,ignore_index=True)


def comparison_summary(d,label):
    rows=[]
    for direction,g in d.groupby("direction"):
        lo,hi=cluster_interval(g,"discordant")
        mlo,mhi=cluster_interval(g,"delta_FAST","median")
        rlo,rhi=cluster_interval(g,"delta_log_AST","median")
        rows.append({"dataset":label,"direction":direction,"comparisons":len(g),"scans":g.scan_token.nunique(),"patients":g.research_token.nunique(),
          "discordant":int(g.discordant.sum()),"proportion":g.discordant.mean(),"ci_low":lo,"ci_high":hi,
          "upward":int(g.upward.sum()),"downward":int(g.downward.sum()),"median_lag":g.abs_lag.median(),
          "median_delta_FAST":g.delta_FAST.median(),"median_delta_FAST_low":mlo,"median_delta_FAST_high":mhi,
          "median_AST_ratio":np.exp(g.delta_log_AST.median()),"median_AST_ratio_low":np.exp(rlo),"median_AST_ratio_high":np.exp(rhi)})
    return rows


def composition(scans,sel,label):
    pre=set(sel["prior_90"].scan_token); wide=set(sel["symmetric_90"].scan_token)
    d=scans.copy()
    d["availability_group"]=np.select([d.scan_token.isin(pre),d.scan_token.isin(wide)],["prior","wide_only"],default="unavailable")
    rows=[]
    for name,g in d.groupby("availability_group"):
        row={"dataset":label,"group":name,"scans":len(g),"patients":g.research_token.nunique(),
             "female_pct":100*g.sex.isin(["F","female","女"]).mean(),"LSM_ge9_5_pct":100*g.lsm.ge(9.5).mean(),
             "BMI_ge25_pct":100*g.bmi.ge(25).mean(),"metabolic_steatosis_pct":100*g.metabolic_steatosis.mean()}
        for col in ("age_at_scan","bmi","lsm","cap"):
            row.update({f"{col}_{q}":g[col].quantile(p) for q,p in [("p25",.25),("median",.5),("p75",.75)]})
        rows.append(row)
    return d,rows


def model_incremental(d):
    rows=[]
    status={"comparisons":len(d),"patients":int(d.research_token.nunique()),"discordances":int(d.discordant.sum())}
    if d.discordant.sum()<50 or d.research_token.nunique()<100:
        status["status"]="descriptive_only_prespecified_sample_gate"
        return pd.DataFrame(),status
    x=d.copy()
    x["log_AST"]=np.log(x.AST)
    x["log_lag"]=np.log1p(x.abs_lag)
    x["later"]=x.direction.eq("later").astype(int)
    y=x.discordant.to_numpy(); groups=x.research_token.to_numpy()
    predictions={}
    features={"boundary_only":["boundary_distance","log_AST"],
              "boundary_plus_timing":["boundary_distance","log_AST","log_lag","later","bmi"]}
    for name,cols in features.items():
        pred=np.zeros(len(x))
        for seed in range(10):
            cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED+seed)
            for train,test in cv.split(x,y,groups):
                assert not set(groups[train])&set(groups[test])
                pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
                pipe.fit(x.iloc[train][cols],y[train])
                pred[test]+=pipe.predict_proba(x.iloc[test][cols])[:,1]/10
        predictions[name]=pred
        cal=sm.GLM(y,sm.add_constant(logit(np.clip(pred,1e-6,1-1e-6))),family=sm.families.Binomial()).fit()
        rows.append({"model":name,"Brier":brier_score_loss(y,pred),"AUC":roc_auc_score(y,pred),
                     "calibration_intercept":cal.params[0],"calibration_slope":cal.params[1]})
    x["brier_gain"]=(y-predictions["boundary_only"])**2-(y-predictions["boundary_plus_timing"])**2
    lo,hi=cluster_interval(x,"brier_gain")
    status.update(status="patient_grouped_OOF_exploratory",brier_improvement=float(x.brier_gain.mean()),
                  brier_improvement_ci_low=lo,brier_improvement_ci_high=hi,
                  interval_scope="patient bootstrap of fixed cross-fitted losses; no full refit uncertainty")
    freeze(pd.DataFrame({"research_token":groups,"discordant":y,**predictions}),DATA/"model_oof_predictions.parquet")
    return pd.DataFrame(rows),status


def register(files):
    now=datetime.now(timezone.utc).isoformat()
    c=sqlite3.connect(REGISTRY)
    c.execute("PRAGMA foreign_keys=ON")
    with c:
        if not c.execute("SELECT 1 FROM datasets WHERE dataset_id=?",(DATASET,)).fetchone():
            c.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(
              DATASET,"VCTE FAST v13 person-level laboratory-linked analysis","local unified warehouse","clinical",
              "VCTE 2023-03-01 to 2026-07-09; LIS support through 2026-08-13","linked-laboratory reanalysis after JCTH rejection",
              "S3","pseudonymized research_token; exact event dates","analysis_frozen",
              "DS-VCTE-FAST-V12-20261003",len(files),"8079 record numbers; 9562 scans; person-level merge via shared laboratory identity; warehouse LIS plus 2026-10-03 liver LIS export",
              "local authorized research; no public patient-level release",
              "VCTE-LIS token map built read-only from identity layer; identity layer not modified; AST code harmonization flagged",now,now))
        for p in files:
            aid="AS-VCTE-FAST-V13-"+hashlib.sha256(p.name.encode()).hexdigest()[:12]
            existing=c.execute("SELECT sha256 FROM assets WHERE stored_path=?",(str(p),)).fetchone()
            if existing:
                if existing[0]!=sha(p):
                    c.execute("UPDATE assets SET sha256=?,size_bytes=? WHERE stored_path=?",(sha(p),p.stat().st_size,str(p)))
            else:
                c.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?)",(aid,DATASET,str(p),p.name,"Parquet" if p.suffix==".parquet" else "JSON",p.stat().st_size,sha(p),"verified",None,now))
    c.close()


def main():
    TABLES.mkdir(parents=True,exist_ok=True)
    oldfiles=[PROJECT/"04_manuscript/VCTE_FAST_Clinical_Manuscript_v9r3.md",
              WAREHOUSE/"03_冻结分析数据/DS-SHARED-VCTE-LIS-V8-20260721/vcte_v8.parquet",
              WAREHOUSE/"03_冻结分析数据/DS-SHARED-VCTE-LIS-V8-20260721/lis_v8.parquet"]
    hashes={str(p):sha(p) for p in oldfiles}
    scans_all,events,audit=construct()
    scans=scans_all[scans_all.linked].copy()
    n_link=len(scans)
    p=pairs(scans,events); sel=rules(p)
    strict=rules(pairs(scans,events[events.strict_unit]))
    strict_link_ids=set(scans.loc[scans.strict_link,"scan_token"])
    sl_scans=scans[scans.scan_token.isin(strict_link_ids)]
    strict_link=rules(pairs(sl_scans,events))
    summary=support(sel,"linked",n_link)+support(strict,"strict_unit",n_link)+support(strict_link,"strict_linkage",len(sl_scans))
    db=ro(DB)
    old=pd.read_sql_query("SELECT linked_lis_event_token observation_token,research_token,event_timestamp,value_numeric AST FROM linked_lis_events WHERE analyte='AST' AND value_numeric>0",db)
    db.close()
    old["available_timestamp"]=pd.to_datetime(old.event_timestamp)
    old["available_date"]=old.available_timestamp.dt.normalize()
    oldsel=rules(pairs(scans_all.assign(research_token=scans_all.record_token),old))  # submitted data are keyed by record number
    summary+=support(oldsel,"submitted_baseline",len(scans_all))
    assert len(oldsel['same_day'])==210 and len(oldsel['symmetric_90'])==626
    pd.DataFrame(summary).to_csv(TABLES/"01_timing_support.csv",index=False)
    freeze(sel["symmetric_90"].drop(columns=["event_timestamp"],errors="ignore"),DATA/"wide_window_assessments.parquet")
    comp,comp_rows=composition(scans,sel,"linked")
    pd.DataFrame(comp_rows).to_csv(TABLES/"02_case_mix.csv",index=False)
    a=anchor_comparisons(sel); sa=anchor_comparisons(strict); sla=anchor_comparisons(strict_link)
    freeze(a,DATA/"same_day_anchor_comparisons.parquet")
    rows=(comparison_summary(a,"linked")+comparison_summary(sa,"strict_unit")+comparison_summary(sla,"strict_linkage")
          +comparison_summary(anchor_comparisons(oldsel),"submitted_baseline"))
    pd.DataFrame(rows).to_csv(TABLES/"03_observed_anchor_reclassification.csv",index=False)
    models,model_status=model_incremental(a)
    models.to_csv(TABLES/"04_cross_fitted_model_comparison.csv",index=False)
    brackets=sel["prior_90"].merge(sel["later_90"][["scan_token","AST","FAST","category","lag"]],on="scan_token",suffixes=("_earlier","_later"))
    brackets["delta_FAST"]=brackets.FAST_later-brackets.FAST_earlier
    brackets["discordant"]=brackets.category_later.ne(brackets.category_earlier).astype(int)
    lo,hi=cluster_interval(brackets,"discordant")
    bracket_summary={"scans":len(brackets),"patients":int(brackets.research_token.nunique()),
                     "discordant":int(brackets.discordant.sum()),"proportion":float(brackets.discordant.mean()),"ci_low":lo,"ci_high":hi,
                     "median_delta_FAST":float(brackets.delta_FAST.median())}
    freeze(brackets,DATA/"bracket_comparisons.parquet")
    pd.crosstab(a.reference_category,a.category).reindex(index=LABELS,columns=LABELS,fill_value=0).to_csv(TABLES/"05_observed_transition_matrix.csv")
    subrows=[]
    for name,mask in [("all",np.ones(len(scans),bool)),("HBsAg_positive",scans.HBsAg_positive),
                       ("viral_markers_negative",scans.viral_markers_negative),("MASLD_compatible",scans.MASLD_compatible),
                       ("diabetes_any",scans.diabetes_any),("BMI_ge25",scans.BMI_ge25),
                       ("metabolic_steatosis_275",scans.metabolic_steatosis)]:
        s=scans[mask]; ids=set(s.scan_token); subset={k:d[d.scan_token.isin(ids)] for k,d in sel.items()}
        wide=subset["symmetric_90"]
        subrows.append({"subgroup":name,"linked_scans":len(s),"linked_patients":s.research_token.nunique(),
          "same_day":len(subset['same_day']),"prior":len(subset['prior_90']),"wide":len(wide),"later":int(wide.lag.gt(0).sum()),
          "reuse":int(wide.observation_token.duplicated(False).sum())})
    pd.DataFrame(subrows).to_csv(TABLES/"06_subgroup_support.csv",index=False)
    ranges=[]
    for label,d in [("all_scans",scans_all),("linked_scans",scans),("first_scan",scans_all.sort_values('scan_timestamp').drop_duplicates('research_token')),("AST_events",events)]:
        for var in [v for v in ("age_at_scan","bmi","lsm","cap","AST") if v in d]:
            val=d[var].dropna()
            ranges.append({"population":label,"variable":var,"n":len(val),"missing":len(d)-len(val),
                           **{k:val.quantile(q) for k,q in [("min",0),("p01",.01),("p25",.25),("median",.5),("p75",.75),("p99",.99),("max",1)]}})
    pd.DataFrame(ranges).to_csv(TABLES/"07_component_ranges.csv",index=False)
    first_all=scans_all.sort_values('scan_timestamp').drop_duplicates('research_token')
    anchor_ids=set(a.research_token)
    baserows=[]
    for name,d in [("All patients",first_all),("Laboratory-linked",first_all[first_all.linked]),
                   ("Not linked",first_all[~first_all.linked]),("Paired-comparison patients",first_all[first_all.research_token.isin(anchor_ids)])]:
        row={"group":name,"patients":len(d),"female_n":int(d.sex.isin(['F','female','女']).sum()),
             "BMI_ge25_n":int(d.BMI_ge25.sum()),"metabolic_steatosis_n":int(d.metabolic_steatosis.sum()),
             "LSM_ge9_5_n":int(d.lsm.ge(9.5).sum()),
             "gastro_referral_n":int(d.department.fillna('').str.contains('消化|Gastro',case=False,regex=True).sum()),
             "infectious_referral_n":int(d.department.fillna('').str.contains('感染',regex=True).sum()),
             **{k:int(d[k].sum()) for k in ('HBsAg_tested','HBsAg_positive','anti_HCV_tested','anti_HCV_positive','diabetes_any','MASLD_compatible','viral_markers_negative')},
             **{k:int(d[k].sum()) for k in ('diagnosis_record_available','fatty_history','diabetes_history','viral_history','alcohol_history','autoimmune_history','drug_history')}}
        for col in ('age_at_scan','bmi','lsm','cap'):
            row[col]=f"{d[col].median():.1f} ({d[col].quantile(.25):.1f}-{d[col].quantile(.75):.1f})"
        baserows.append(row)
    pd.DataFrame(baserows).to_csv(TABLES/"08_baseline_first_patient.csv",index=False)
    first_complete=scans.sort_values('scan_timestamp').drop_duplicates('research_token')
    first_complete=first_complete[first_complete.scan_date.le(LAB_END-pd.Timedelta(days=90)) & first_complete.scan_date.ge(pd.Timestamp('2023-03-01')+pd.Timedelta(days=90))]
    fc=rules(pairs(first_complete,events))
    pd.DataFrame(support(fc,'first_scan_calendar_eligible',len(first_complete))).to_csv(TABLES/"09_first_scan_sensitivity.csv",index=False)
    collection=events[events.collection_timestamp.notna() & events.collection_timestamp.le(events.available_timestamp)].copy()
    cs=rules(pairs(scans,collection,date_col='collection_date'))
    pd.DataFrame(support(cs,'collection_time_subset',n_link)).to_csv(TABLES/"10_collection_time_sensitivity.csv",index=False)
    med=events.groupby(['research_token','available_date'],as_index=False).agg(AST=('AST','median'),available_timestamp=('available_timestamp','min'),observation_token=('observation_token','first'))
    medsel=rules(pairs(scans,med))
    pd.DataFrame(comparison_summary(anchor_comparisons(medsel),'daily_median')).to_csv(TABLES/"11_daily_median_sensitivity.csv",index=False)
    same=sel['same_day'].copy(); same['stable25']=(category(fast(same.lsm,same.cap,same.AST*.75))==same.category)&(category(fast(same.lsm,same.cap,same.AST*1.25))==same.category)
    stable=[]
    for cat,g in same.groupby('category'):
        lo,hi=cluster_interval(g,'stable25')
        stable.append({'category':cat,'n':len(g),'patients':g.research_token.nunique(),'stable':int(g.stable25.sum()),'proportion':g.stable25.mean(),'ci_low':lo,'ci_high':hi})
    pd.DataFrame(stable).to_csv(TABLES/"12_same_day_algebraic_sensitivity.csv",index=False)
    base=-1.65+1.07*np.log(scans_all.lsm)+2.66e-8*scans_all.cap**3
    unreachable=base.le(logit(.67))
    th={"unreachable67":int(unreachable.sum()),"denominator":len(scans_all),"proportion":float(unreachable.mean())}
    linkage=json.loads((TABLES/"00_linkage_summary.json").read_text())
    summary={"data":audit,"linkage":linkage,"support":summary,"anchors":rows,"brackets":bracket_summary,
             "incremental_model":model_status,"reachability":th,"baseline":baserows,"subgroups":subrows,
             "first_scan_calendar_eligible_n":len(first_complete),"collection_timestamp_events":len(collection),
             "strict_linkage_scans":len(sl_scans),
             "input_hashes":hashes,"original_files_unchanged":all(sha(p)==h for p,h in hashes.items())}
    assert summary['original_files_unchanged']
    (TABLES/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)),encoding='utf8')
    freeze_manifest={'dataset_id':DATASET,'sensitivity':'S3','contract_sha256':sha(HERE/'ANALYSIS_CONTRACT.md'),
                     'files':{p.name:sha(p) for p in sorted(DATA.glob('*.parquet'))},'source_data':audit}
    manifest=DATA/'MANIFEST.json'
    manifest.write_text(json.dumps(freeze_manifest,ensure_ascii=False,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)),encoding='utf8'); manifest.chmod(0o600)
    register([*sorted(DATA.glob('*.parquet')),manifest])
    print(json.dumps({k:v for k,v in summary.items() if k not in ('baseline','subgroups','input_hashes','support')},ensure_ascii=False,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)))


if __name__=='__main__':
    main()
