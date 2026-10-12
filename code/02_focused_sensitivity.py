"""Focused empirical checks using the immutable rebuilt cohort."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import statsmodels.api as sm

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('analysis',HERE/'01_rebuild_and_analyze.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pd.read_parquet(m.DATA/'scans.parquet')
e=pd.read_parquet(m.DATA/'ast_events.parquet')
a=pd.read_parquet(m.DATA/'same_day_anchor_comparisons.parquet')
sel=m.rules(m.pairs(s,e))
s_all=s.copy()
s=s[s.linked].copy()
rows=[]
first_ids=set(sel['same_day'].sort_values(['scan_timestamp','scan_token']).drop_duplicates('research_token').scan_token)
event_unique=a.sort_values(['abs_lag','abs_hours','scan_token']).drop_duplicates('observation_token')
strict_link=set(s.loc[s.strict_link,'scan_token'])
for name,d in [('all',a),('first_anchor_per_patient',a[a.scan_token.isin(first_ids)]),
               ('unique_remote_AST',event_unique),('AST_both_5_to_100',a[a.AST.between(5,100)&a.reference_AST.between(5,100)]),
               ('AST_both_5_to_200',a[a.AST.between(5,200)&a.reference_AST.between(5,200)]),
               ('strict_linkage',a[a.scan_token.isin(strict_link)]),
               ('HBsAg_positive',a[a.HBsAg_positive]),('HBsAg_positive_ge1',a[a.HBsAg_positive_ge1.fillna(False).astype(bool)]),
               ('viral_markers_negative',a[a.viral_markers_negative]),('MASLD_compatible',a[a.MASLD_compatible]),
               ('diabetes_any',a[a.diabetes_any]),('BMI_ge25',a[a.bmi.ge(25)]),('metabolic_steatosis',a[a.metabolic_steatosis])]:
    rows+=m.comparison_summary(d,name)
pd.DataFrame(rows).to_csv(m.TABLES/'13_anchor_sensitivity.csv',index=False)
strata=[]
for field in ('reference_category','direction'):
    for key,g in a.groupby(field):
        lo,hi=m.cluster_interval(g,'discordant')
        strata.append(dict(stratum=field,level=key,n=len(g),patients=g.research_token.nunique(),
                           changed=int(g.discordant.sum()),proportion=g.discordant.mean(),ci_low=lo,ci_high=hi))
a['lag_band']=pd.cut(a.abs_lag,[0,7,30,60,90],labels=['1-7','8-30','31-60','61-90'])
for key,g in a.groupby('lag_band',observed=True):
    lo,hi=m.cluster_interval(g,'discordant')
    strata.append(dict(stratum='lag_band',level=key,n=len(g),patients=g.research_token.nunique(),
                       changed=int(g.discordant.sum()),proportion=g.discordant.mean(),ci_low=lo,ci_high=hi))
pd.DataFrame(strata).to_csv(m.TABLES/'14_actual_reclassification_strata.csv',index=False)
for (grp,direction),g in a.groupby(['etiology_group','direction']):
    lo,hi=m.cluster_interval(g,'discordant')
    strata.append(dict(stratum='etiology_'+direction,level=grp,n=len(g),patients=g.research_token.nunique(),
                       changed=int(g.discordant.sum()),proportion=g.discordant.mean(),ci_low=lo,ci_high=hi))
for key,g in a.groupby('etiology_group'):
    lo,hi=m.cluster_interval(g,'discordant')
    strata.append(dict(stratum='etiology_group',level=key,n=len(g),patients=g.research_token.nunique(),
                       changed=int(g.discordant.sum()),proportion=g.discordant.mean(),ci_low=lo,ci_high=hi))
pd.DataFrame(strata).to_csv(m.TABLES/'14_actual_reclassification_strata.csv',index=False)
# Whether the same-day result sat above the conventional upper reference limit (40 U/L).
a['reference_AST_band']=np.where(a.reference_AST.gt(40),'same_day_AST_gt40','same_day_AST_le40')
band=[]
for grp,g in a.groupby('reference_AST_band'):band+=m.comparison_summary(g,grp)
pd.DataFrame(band).to_csv(m.TABLES/'19_reference_AST_band.csv',index=False)
etio=[]
for grp,g in a.groupby('etiology_group'):etio+=m.comparison_summary(g,grp)
pd.DataFrame(etio).to_csv(m.TABLES/'18_etiology_direction.csv',index=False)
trans=[]
for direction,g in a.groupby('direction'):
    for rc in m.LABELS:
        for nc in m.LABELS:
            trans.append(dict(direction=direction,same_day_category=rc,remote_category=nc,
                              n=int((g.reference_category.eq(rc)&g.category.eq(nc)).sum())))
pd.DataFrame(trans).to_csv(m.TABLES/'15_directional_transitions.csv',index=False)

B=-1.65+1.07*np.log(s_all.lsm)+2.66e-8*s_all.cap**3
reach=[]
for c in (.35,.67):
    possible=B>logit(c)
    ac=63.3/(B[possible]-logit(c))
    interval=m.cluster_interval(pd.DataFrame({'research_token':s_all.research_token,'unreachable':~possible}),'unreachable')
    reach.append({'boundary':c,'unreachable':int((~possible).sum()),'n':len(s_all),'ci_low':interval[0],'ci_high':interval[1],
                  'required_AST_p25':float(ac.quantile(.25)),'required_AST_median':float(ac.median()),'required_AST_p75':float(ac.quantile(.75)),
                  'required_AST_le40':int(ac.le(40).sum())})
pd.DataFrame(reach).to_csv(m.TABLES/'16_thresholds.csv',index=False)
same=sel['same_day'].copy()
bb=-1.65+1.07*np.log(same.lsm)+2.66e-8*same.cap**3
ds=[]
for c in (.35,.67):
    diff=bb-logit(c)
    ac=np.where(diff>0,63.3/diff,np.nan)
    ds.append(np.abs(np.log(ac/same.AST)))
same['margin']=np.nanmin(np.stack(ds),axis=0)
same['required_relative_change']=1-np.exp(-same.margin)
mr=[]
for c,g in same.groupby('category'):
    lo,hi=m.cluster_interval(g,'margin','median')
    mr.append(dict(category=c,n=len(g),margin_median=g.margin.median(),margin_p25=g.margin.quantile(.25),
                   margin_p75=g.margin.quantile(.75),ci_low=lo,ci_high=hi))
pd.DataFrame(mr).to_csv(m.TABLES/'17_same_day_margin.csv',index=False)

# CK and HBV DNA from the liver LIS export (patients linked to that source only).
lab=pd.read_parquet(m.DATA/'liver_lis_events.parquet')
lab['date']=pd.to_datetime(lab.report_timestamp).dt.normalize()
ck=lab[lab.analyte.eq('CK')&lab.value.notna()][['research_token','date','value']]
def ck_high(tok,dates):
    g=ck[ck.research_token.eq(tok)]
    return any(((g.date-d).dt.days.abs()<=3).any() and g.loc[(g.date-d).dt.days.abs()<=3,'value'].gt(200).any() for d in dates)
a['ref_date']=a.scan_date; a['other_date']=a.available_date
ck_tokens=set(ck.research_token)
a['CK_measured']=[tok in ck_tokens and any(((ck.loc[ck.research_token.eq(tok),'date']-d).dt.days.abs()<=3).any() for d in (r,o)) for tok,r,o in zip(a.research_token,a.ref_date,a.other_date)]
a['CK_high']=[tok in ck_tokens and ck_high(tok,(r,o)) for tok,r,o in zip(a.research_token,a.ref_date,a.other_date)]
dna=lab[lab.analyte.eq('HBV_DNA')][['research_token','date','value','qualifier']]
def dna_status(tok,d):
    g=dna[dna.research_token.eq(tok)&((dna.date-d).dt.days.abs()<=180)]
    if g.empty:return 'untested'
    return 'detectable' if (g.qualifier.eq('exact')&g.value.gt(0)).any() or g.qualifier.eq('above_limit').any() else 'undetectable'
a['HBV_DNA_status']=[dna_status(tok,d) if hb else 'not_HBsAg' for tok,d,hb in zip(a.research_token,a.scan_date,a.HBsAg_positive)]
extra=[]
for name,d in [('CK_not_high',a[~a.CK_high]),('CK_measured_not_high',a[a.CK_measured&~a.CK_high]),
               ('HBsAg_DNA_detectable',a[a.HBV_DNA_status.eq('detectable')]),('HBsAg_DNA_undetectable',a[a.HBV_DNA_status.eq('undetectable')])]:
    extra+=m.comparison_summary(d,name)
pd.DataFrame(extra).to_csv(m.TABLES/'20_ck_hbvdna.csv',index=False)
ck_counts={'comparisons_CK_measured':int(a.CK_measured.sum()),'comparisons_CK_high':int(a.CK_high.sum()),
           'HBsAg_DNA_status':a.HBV_DNA_status.value_counts().to_dict()}
# About 10% of results are signed off the day after collection; drop comparisons at ±1 day.
pd.DataFrame(m.comparison_summary(a[a.lag.abs()>1],'exclude_1day')).to_csv(m.TABLES/'22_exclude_1day.csv',index=False)
out={'excluded_1day_comparisons':int(a.lag.abs().le(1).sum()),'ck_hbvdna':ck_counts,'anchor_scans':int(a.scan_token.nunique()),'anchor_patients':int(a.research_token.nunique()),
     'anchor_comparisons':len(a),'actual_reclassification_strata':strata,'thresholds':reach,
     'AST_over100_events':int(e.AST.gt(100).sum()),'AST_over200_events':int(e.AST.gt(200).sum()),
     'clinical_history_missing_patients':int((~s.sort_values('scan_timestamp').drop_duplicates('research_token').diagnosis_record_available).sum())}
(m.TABLES/'focused_summary.json').write_text(json.dumps(out,indent=2,default=lambda v:float(v)),encoding='utf8')
assert np.all(np.diff(m.fast(10,280,np.linspace(1,1000,1000)))>0)
for c in (.35,.67):
    possible=B>logit(c); ac=63.3/(B[possible]-logit(c))
    assert np.allclose(m.fast(s_all.loc[possible,'lsm'],s_all.loc[possible,'cap'],ac),c,atol=1e-12)
print(json.dumps(out,indent=2,default=lambda v:float(v)))
