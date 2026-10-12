"""Editable, panel-grouped figures for the reconstructed clinical manuscript."""
import importlib.util
import json
import re
import textwrap
from pathlib import Path
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.special import logit
from lxml import etree as ET

H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('analysis',H/'01_rebuild_and_analyze.py')
m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
F=H/'figures'; F.mkdir(exist_ok=True)
S=pd.read_parquet(m.DATA/'scans.parquet'); E=pd.read_parquet(m.DATA/'ast_events.parquet')
A=pd.read_parquet(m.DATA/'same_day_anchor_comparisons.parquet')
SEL=m.rules(m.pairs(S,E))
T=lambda name:pd.read_csv(m.TABLES/name)
SUMMARY=json.loads((m.TABLES/'summary.json').read_text())
FOCUS=json.loads((m.TABLES/'focused_summary.json').read_text())
EXTP=H/'results'/'external_reference'  # public-cohort results shipped with this code
EXT=pd.read_csv(EXTP/'11_supplementary_table_4_robustness.tsv',sep='\t')
EXT_STABLE=pd.read_csv(EXTP/'04_stability.tsv',sep='\t')
EXT_STABLE['category']=EXT_STABLE.category.str.replace('-','_')
BLUE='#277DA8'; ORANGE='#D27A21'; BERRY='#A84162'; GREY='#7E858B'; TEAL='#268C88'
PAL={'rule_out':BLUE,'indeterminate':ORANGE,'rule_in':BERRY}
NAMES={'rule_out':'Rule-out','indeterminate':'Indeterminate','rule_in':'Rule-in'}
GROUP=['prior','wide_only','unavailable']; GN=['Earlier AST','Wide-only','No captured AST']
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.titlesize':13,'axes.labelsize':12,
 'xtick.labelsize':11,'ytick.labelsize':11,'svg.fonttype':'none','pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.5,
 'legend.fontsize':10,'savefig.facecolor':'white','mathtext.fontset':'stix'})
MAN=[]

def plate(name,titles,rows=2,cols=3):
    fig,axs=plt.subplots(rows,cols,figsize=(12,8),layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.13,h_pad=.15,wspace=.10,hspace=.14)
    for i,(ax,title) in enumerate(zip(axs.flat,titles)):
        ax.set_gid(f'{name}_panel_{chr(65+i)}')
        ax.set_title(f'{chr(65+i)}   '+textwrap.fill(title,width=29),loc='left',pad=12,fontweight='bold')
        ax.tick_params(length=3,width=.7)
    return fig,list(axs.flat)

def save(fig,name,cn):
    for fmt in ('svg','pdf','png'):
        fig.savefig(F/f'{name}.{fmt}',dpi=600 if fmt=='png' else 300)
    MAN.append({'name':name,'cn':cn})
    plt.close(fig)

def clean(ax):
    ax.set_axis_off()

def box(ax,x,y,w,h,text,color=GREY,fs=11):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color='#171717')

def forest(ax,df,label,value='proportion',low='ci_low',high='ci_high',percent=True,color=BLUE):
    v=df[value].to_numpy(float); l=df[low].to_numpy(float); h=df[high].to_numpy(float)
    mult=100 if percent else 1
    y=np.arange(len(df)); ax.errorbar(v*mult,y,xerr=np.stack([v-l,h-v])*mult,fmt='o',color=color,capsize=3)
    ax.set_yticks(y,df[label]); ax.invert_yaxis(); ax.set_xlim(0,max(h*mult)*1.20)
    ax.grid(axis='x',alpha=.18); ax.set_xlabel('Participants / comparisons (%)' if percent else 'Estimate')

def category_stack(ax,df,label,absolute=False):
    bottom=np.zeros(len(df)); y=np.arange(len(df))
    totals=df[m.LABELS].sum(axis=1)
    for cat in m.LABELS:
        v=df[cat].to_numpy() if absolute else 100*df[cat].to_numpy()/totals
        ax.barh(y,v,left=bottom,color=PAL[cat],label=NAMES[cat],height=.65,edgecolor='white',lw=.5)
        bottom+=v
    ax.set_yticks(y,df[label]);ax.invert_yaxis();ax.set_xlabel('FAST results' if absolute else 'FAST composition (%)')
    if not absolute:ax.set_xlim(0,100)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=3,frameon=False,fontsize=9)

def empirical_delta(ax):
    for d,col in [('earlier',BLUE),('later',ORANGE)]:
        g=A[A.direction.eq(d)]
        ax.scatter(g.lag,g.delta_FAST,s=17,color=col,alpha=.65,label=f'{d.capitalize()} (n={len(g)})')
    ax.axhline(0,color=GREY,lw=1);ax.axvline(0,color=GREY,lw=.8,ls=':')
    ax.set(xlabel='AST result date minus scan date (days)',ylabel='Nonconcurrent minus same-day FAST',xlim=(-95,95))
    ax.legend(frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,markerscale=1.4)

def transition(ax,direction):
    mat=pd.crosstab(A.loc[A.direction.eq(direction),'reference_category'],A.loc[A.direction.eq(direction),'category']).reindex(index=m.LABELS,columns=m.LABELS,fill_value=0)
    ax.imshow(mat,cmap='Blues',vmin=0)
    for y in range(3):
        for x in range(3):ax.text(x,y,int(mat.iloc[y,x]),ha='center',va='center',color='white' if mat.iloc[y,x]>mat.to_numpy().max()/2 else 'black',fontsize=14)
    ax.set_xticks(range(3),['RO','IND','RI']);ax.set_yticks(range(3),['RO','IND','RI'])
    ax.set_xlabel(f'{direction.capitalize()} AST category');ax.set_ylabel('Same-day category')

def threshold_map(ax,c=.67):
    l=np.geomspace(2,40,180);cap=np.linspace(150,380,180); C,L=np.meshgrid(cap,l)
    b=-1.65+1.07*np.log(L)+2.66e-8*C**3
    v=np.where(b>logit(c),63.3/(b-logit(c)),np.nan)
    pm=ax.pcolormesh(C,L,np.ma.masked_invalid(v),cmap='viridis',norm=LogNorm(20,500),rasterized=True,shading='auto')
    ax.contour(C,L,b,[logit(c)],colors=BERRY,linewidths=1.5)
    cs=ax.contour(C,L,v,[40,80,160],colors='white',linewidths=.6);ax.clabel(cs,fmt='%d',fontsize=8)
    ax.set(xlabel='CAP (dB/m)',ylabel='LSM (kPa)',yscale='log',ylim=(2,40))
    ax.set_yticks([2,5,10,20,40],['2','5','10','20','40']);ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.text(153,2.5,'Unreachable',fontsize=10,color=BERRY)
    cb=plt.colorbar(pm,ax=ax,pad=.02,fraction=.05);cb.set_label('Required AST (U/L)',fontsize=10);cb.set_ticks([20,50,100,200,500],labels=['20','50','100','200','500']);cb.ax.minorticks_off();cb.ax.tick_params(labelsize=9)

# Figure 1: laboratory linkage and the mechanics of AST selection inside the linked population.
LINK=SUMMARY['linkage']; DATA_=SUMMARY['data']
fig,ax=plate('Supplementary_Figure_7',['Study population','Linkage validation','FAST availability by window',
                        'Direction of selected AST','Reuse across scans','FAST category composition'])
clean(ax[0]);ax[0].set_xlim(0,1);ax[0].set_ylim(0,1)
t=T('01_timing_support.csv');lk=t[t.dataset.eq('linked')].set_index('rule')
anc=T('03_observed_anchor_reclassification.csv').query("dataset=='linked'")
flow=[(f"{DATA_['eligible_scans']:,} eligible scans\n{DATA_['eligible_record_numbers']:,} record numbers",GREY),
      (f"{DATA_['linked_patients']:,} laboratory-linked patients\n{DATA_['linked_record_numbers']:,} record numbers; {DATA_['linked_scans']:,} scans",BLUE),
      (f"Same-day AST: {int(lk.loc['same_day','scans']):,} scans\n{int(lk.loc['same_day','patients']):,} patients",ORANGE),
      (f"Paired comparisons: {int(anc.comparisons.sum())}\n{FOCUS['anchor_patients']} patients",BERRY)]
for i,(txt,col) in enumerate(flow):
    y=.80-i*.25;box(ax[0],.04,y-.02,.92,.19,txt,col)
    if i:ax[0].annotate('',(.5,y+.18),(.5,y+.23),arrowprops={'arrowstyle':'->','color':GREY})
q=T('00_linkage_quality.csv').query('department_match==True').sort_values(['source','tolerance_days'])
for j,(src,col,lab) in enumerate([('warehouse_LIS',GREY,'Repository LIS'),('liver_LIS_20261003',BLUE,'Second LIS extract')]):
    g=q[q.source.eq(src)];xs=np.arange(len(g))+(j-.5)*.38
    ax[1].bar(xs,g.linked_patients,width=.36,color=col,label=lab)
    for x,r in zip(xs,g.itertuples()):ax[1].text(x,r.linked_patients+60,f"{100*r.reference_agreement:.1f}\n{100*r.estimated_false_link_fraction:.1f}",ha='center',fontsize=8)
ax[1].set_xticks(range(3),['±0 days','±1 day','±3 days']);ax[1].set(ylabel='Linked record numbers',ylim=(0,4600),xlabel='Encounter tolerance (primary: ±1 day)')
ax[1].legend(frameon=False,fontsize=9,loc='upper left')

g=t[(t.dataset=='linked')&t.rule.str.startswith('symmetric')]; w=g.rule.str.split('_').str[-1].astype(int)
ax[2].plot(w,100*g.scans/g.denominator_scans,'o',ls='-',color=BLUE)
ax[2].set(xlabel='Symmetric half-window (days)',ylabel='Linked scans with FAST (%)',ylim=(0,100))
d=t[(t.dataset=='linked')&t.rule.str.startswith('symmetric')].copy();d['window']=d.rule.str.split('_').str[-1].astype(int)
for i,g in enumerate([SEL['symmetric_7'],SEL['symmetric_30'],SEL['symmetric_90']]):
    counts=[g.lag.lt(0).sum(),g.lag.eq(0).sum(),g.lag.gt(0).sum()];left=0
    for v,c,lab in zip(counts,[BLUE,GREY,ORANGE],['Earlier','Same day','Later']):
        ax[3].barh(i,100*v/len(g),left=left,color=c,label=lab if i==0 else None);left+=100*v/len(g)
ax[3].set_yticks(range(3),['±7 days','±30 days','±90 days']);ax[3].set(xlabel='Selected AST direction (%)',xlim=(0,100));ax[3].legend(frameon=False,ncol=3,fontsize=9,loc='lower center',bbox_to_anchor=(.5,-.35))
ax[4].plot(d.window,d.reused_n/d.scans*100,'s-',color=BERRY)
ax[4].set(xlabel='Symmetric half-window (days)',ylabel='Scans sharing an AST result (%)',ylim=(0,15))
g=t[(t.dataset=='linked')&t.rule.isin(['prior_90','same_day','symmetric_90'])].copy();g['label']=g.rule.map({'prior_90':'Earlier only','same_day':'Same day','symmetric_90':'±90 days'})
category_stack(ax[5],g,'label')
save(fig,'Supplementary_Figure_7','超声导出的病历号不进入LIS身份体系。仓库LIS与2026-10-03肝病LIS导出分别按出生日期、性别及扫描前后1天同科室检验做唯一连接；已连接者以身份证号确定性挂接新导出。两来源对参照连接一致率>99%，估计误连<1%。')

# Figure 2: who the linked and paired patients are, and whether findings hold by etiology.
ETN={'HBsAg_positive':'HBsAg positive','anti_HCV_positive':'Anti-HCV positive','MASLD_compatible':'MASLD-compatible','other_or_undetermined':'Other or undetermined'}
fig,ax=plate('Supplementary_Figure_8',['Laboratory etiology groups','Liver stiffness by group','CAP by group',
                        'Reclassification by group','Median AST ratio by group','By same-day AST level'])
first=S[S.linked].sort_values('scan_timestamp').drop_duplicates('research_token')
order=['MASLD_compatible','HBsAg_positive','anti_HCV_positive','other_or_undetermined']
cnt=first.etiology_group.value_counts().reindex(order).fillna(0)
cols=[TEAL,ORANGE,BERRY,GREY]
ax[0].barh(['MASLD-compatible','HBsAg positive','Anti-HCV positive','Other/undetermined'],cnt,color=cols);ax[0].invert_yaxis();ax[0].set(xlabel='Linked patients (first scan)',xlim=(0,cnt.max()*1.65))
for i,v in enumerate(cnt):ax[0].text(v+15,i,f"{int(v):,} ({100*v/len(first):.1f}%)",va='center',fontsize=10)
mk=SUMMARY['data']['linked_first_scan_markers']
ax[0].text(.98,.36,f"Tested: HBsAg {mk['HBsAg_tested']:,}\nanti-HCV {mk['anti_HCV_tested']:,}",transform=ax[0].transAxes,ha='right',va='center',fontsize=9,color='#555555')
for idx,col in [(1,'lsm'),(2,'cap')]:
    vs=[first.loc[first.etiology_group.eq(g),col].to_numpy() for g in order]
    v=ax[idx].violinplot(vs,showmedians=True,showextrema=False)
    for b,c in zip(v['bodies'],cols):b.set_facecolor(c);b.set_alpha(.6)
    ax[idx].set_xticks(range(1,5),['MASLD','HBsAg+','HCV+','Other'],fontsize=10)
    ax[idx].set_ylabel('LSM (kPa)' if col=='lsm' else 'CAP (dB/m)')
    if col=='lsm':ax[idx].set_ylim(0,30)
ed=T('18_etiology_direction.csv');ed=ed[ed.dataset.ne('anti_HCV_positive')].copy()
SH={'MASLD_compatible':'MASLD','HBsAg_positive':'HBsAg+','other_or_undetermined':'Other'}
ed['label']=ed.apply(lambda r:f"{SH[r.dataset]}, {r.direction} ({r.discordant}/{r.comparisons})",axis=1)
ed['o']=ed.dataset.map({k:i for i,k in enumerate(order)})*2+ed.direction.eq('later');ed=ed.sort_values('o')
forest(ax[3],ed,'label');ax[3].set_xlabel('Reclassified comparisons (%)');ax[3].set_xlim(0,60);ax[3].tick_params(axis='y',labelsize=10)
forest(ax[4],ed.assign(label=ed.label.str.replace(r' \(.*\)','',regex=True)),'label',value='median_AST_ratio',low='median_AST_ratio_low',high='median_AST_ratio_high',percent=False,color=TEAL)
ax[4].axvline(1,color=GREY,ls=':');ax[4].set_xlim(.5,1.5);ax[4].set_xlabel('Median nonconcurrent / same-day AST');ax[4].tick_params(axis='y',labelsize=10)
bd=T('19_reference_AST_band.csv');bd['label']=bd.apply(lambda r:f"{'>40' if 'gt40' in r.dataset else '≤40'}, {r.direction} ({r.discordant}/{r.comparisons})",axis=1)
forest(ax[5],bd,'label');ax[5].set_xlabel('Reclassified comparisons (%)\nby same-day AST (U/L)');ax[5].set_xlim(0,60)
save(fig,'Supplementary_Figure_8','HBsAg/抗HCV定义病毒性病因；符合MASLD定义需CAP≥275、至少一项心代谢指标且病毒标志物不阳性（饮酒资料缺失，MetALD无法排除）。各病因组均为较早偏高、较晚偏低；符合MASLD组改判比例不低于总体。')

# Main figures: 2 x 2 panels, 8-9 pt type at 7.2 in width, readable in grayscale
# (dark/light tones, marker shapes and hatching carry the meaning, not hue alone).
DARK='#1F4E79'; LIGHT='#E3A33B'
MAINRC={'font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,'lines.linewidth':1.0,'axes.linewidth':.6}
def plate_main(name,titles):
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.4),layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.06,h_pad=.08,wspace=.08,hspace=.10)
    for i,(ax,title) in enumerate(zip(axs.flat,titles)):
        ax.set_gid(f'{name}_panel_{chr(65+i)}')
        ax.set_title(f'{chr(65+i)}   {title}',loc='left',pad=6,fontweight='bold')
        ax.tick_params(length=2.5,width=.6)
    return fig,list(axs.flat)

def transition_bw(ax,direction):
    mat=pd.crosstab(A.loc[A.direction.eq(direction),'reference_category'],A.loc[A.direction.eq(direction),'category']).reindex(index=m.LABELS,columns=m.LABELS,fill_value=0)
    ax.imshow(mat,cmap='Greys',vmin=0,vmax=mat.to_numpy().max()*1.15)
    for y in range(3):
        for x in range(3):
            v=int(mat.iloc[y,x]);ax.text(x,y,v,ha='center',va='center',fontsize=9,fontweight='bold' if x==y else 'normal',color='white' if v>mat.to_numpy().max()*.55 else 'black')
    ax.set_xticks(range(3),['RO','IND','RI']);ax.set_yticks(range(3),['RO','IND','RI'])
    ax.set_xlabel(f'{direction.capitalize()} AST category');ax.set_ylabel('Same-day category')
    for sp in ax.spines.values():sp.set_visible(False)

with matplotlib.rc_context(MAINRC):
    # Figure 1: reclassification when same-day AST is replaced by earlier or later AST.
    fig,ax=plate_main('Figure_1',['Observed FAST change','Direction of reclassification','Earlier vs same-day AST','Later vs same-day AST'])
    for d,col,mk,fc in [('earlier',DARK,'o',DARK),('later',LIGHT,'^','white')]:
        g=A[A.direction.eq(d)]
        ax[0].scatter(g.lag,g.delta_FAST,s=9,marker=mk,facecolors=fc,edgecolors=col,linewidths=.6,alpha=.8,label=f'{d.capitalize()} AST (n={len(g)})')
    ax[0].axhline(0,color=GREY,lw=.8);ax[0].axvline(0,color=GREY,lw=.6,ls=':')
    ax[0].set(xlabel='AST date minus scan date (days)',ylabel='Nonconcurrent minus same-day FAST',xlim=(-95,95),ylim=(-.65,.8))
    ax[0].legend(frameon=False,loc='upper center',ncol=2,markerscale=1.3,handletextpad=.3,columnspacing=1.0)
    z=T('03_observed_anchor_reclassification.csv').query("dataset=='linked'").set_index('direction').loc[['earlier','later']]
    for i,(d,r) in enumerate(z.iterrows()):
        ax[1].barh(i,-r.downward,height=.5,color=DARK,edgecolor=DARK,label='Lower category' if i==0 else None)
        ax[1].barh(i,r.upward,height=.5,color='white',edgecolor=DARK,hatch='////',lw=.6,label='Higher category' if i==0 else None)
        ax[1].text(-r.downward-4,i,str(int(r.downward)),ha='right',va='center',fontsize=8)
        ax[1].text(r.upward+4,i,str(int(r.upward)),ha='left',va='center',fontsize=8)
        ax[1].text(0,i-.36,f"{int(r.discordant)}/{int(r.comparisons)} reclassified, {100*r.proportion:.1f}% ({100*r.ci_low:.1f}–{100*r.ci_high:.1f})",ha='center',va='center',fontsize=7.5)
    ax[1].set_yticks(range(2),['Earlier AST','Later AST']);ax[1].invert_yaxis();ax[1].axvline(0,color='black',lw=.6)
    ax[1].set(xlabel='Comparisons',xlim=(-175,175),ylim=(1.6,-.6))
    ax[1].set_xticks([-150,-100,-50,0,50,100,150],['150','100','50','0','50','100','150'])
    ax[1].legend(frameon=False,loc='lower right')
    transition_bw(ax[2],'earlier');transition_bw(ax[3],'later')
    save(fig,'Figure_1','固定同一次扫描的 LSM/CAP，用真实较早或较晚 AST 替代同日 AST：A 为 FAST 变化与间隔，B 为下调/上调例数及改判比例，C–D 为分类矩阵。')

    # Figure 2: scan-specific thresholds judged against the RCV of AST.
    RCV=json.loads((m.TABLES/'rcv_summary.json').read_text());RUP,RDN=RCV['rcv_up'],RCV['rcv_down']
    RF=T('23_rcv_threshold_flag.csv').query("analysis=='primary'")
    fig,ax=plate_main('Figure_2',['AST required for rule-in','AST change vs RCV','Reclassification by RCV flag','Reporting check'])
    l=np.geomspace(2,40,180);cap=np.linspace(150,380,180);C,L=np.meshgrid(cap,l)
    b=-1.65+1.07*np.log(L)+2.66e-8*C**3;v=np.where(b>logit(.67),63.3/(b-logit(.67)),np.nan)
    pm=ax[0].pcolormesh(C,L,np.ma.masked_invalid(v),cmap='viridis',norm=LogNorm(20,500),rasterized=True,shading='auto')
    ax[0].contour(C,L,b,[logit(.67)],colors='black',linewidths=1.0,linestyles='--')
    cs=ax[0].contour(C,L,v,[40,80,160],colors='white',linewidths=.5);ax[0].clabel(cs,fmt='%d',fontsize=7)
    ax[0].set(xlabel='CAP (dB/m)',ylabel='LSM (kPa)',yscale='log',ylim=(2,40))
    ax[0].set_yticks([2,5,10,20,40],['2','5','10','20','40']);ax[0].yaxis.set_minor_formatter(plt.NullFormatter())
    ax[0].text(155,2.4,'Rule-in\nunreachable',fontsize=7.5,va='bottom')
    cb=plt.colorbar(pm,ax=ax[0],pad=.02,fraction=.06);cb.set_label('Required AST (U/L)',fontsize=8);cb.set_ticks([20,50,100,200,500],labels=['20','50','100','200','500']);cb.ax.minorticks_off();cb.ax.tick_params(labelsize=7.5)
    lr=np.log(A.AST/A.reference_AST)
    hc=ax[1].hist([lr[A.direction.eq(d)] for d in ['earlier','later']],bins=np.linspace(-2,2,41),stacked=True,color=[DARK,'#F2D49B'],edgecolor=['white','#8A6A2A'],linewidth=.3,label=['Earlier AST','Later AST'])[0]
    ax[1].set_ylim(0,hc[-1].max()*1.5)
    ax[1].axvspan(np.log(1-RDN),np.log(1+RUP),facecolor='#EFEFEF',lw=0,zorder=0)
    ax[1].axvline(np.log(1-RDN),color='black',lw=.6,ls='--');ax[1].axvline(np.log(1+RUP),color='black',lw=.6,ls='--')
    out=RF.query("label=='AST_change_beyond_RCV'").iloc[0]
    ax[1].text(.03,.97,f"Shaded: RCV\n(+{100*RUP:.1f}%/−{100*RDN:.1f}%)\nBeyond RCV: {100*out.proportion:.1f}%",transform=ax[1].transAxes,va='top',ha='left',fontsize=7.5)
    ax[1].set(xlabel='Log nonconcurrent / same-day AST',ylabel='Comparisons',xlim=(-2,2));ax[1].legend(frameon=False,loc='upper right')
    x=np.arange(2)
    for k,(lab,name,fc,hatch) in enumerate([('reclassified_if_flagged','Flagged (within RCV)',DARK,None),('reclassified_if_not_flagged','Not flagged',  'white','////')]):
        g=RF[RF.label.eq(lab)].set_index('direction').loc[['earlier','later']];xs=x+(k-.5)*.36
        ax[2].bar(xs,100*g.proportion,width=.34,color=fc,edgecolor=DARK,hatch=hatch,lw=.6,label=name)
        ax[2].errorbar(xs,100*g.proportion,yerr=100*np.stack([g.proportion-g.ci_low,g.ci_high-g.proportion]),fmt='none',ecolor='black',capsize=2.5,lw=.7)
        for xx,r in zip(xs,g.itertuples()):ax[2].text(xx,100*r.ci_high+1.5,f"{r.events}/{r.n}",ha='center',fontsize=7)
    ax[2].set_xticks(x,['Earlier AST','Later AST']);ax[2].set(ylabel='Reclassified by same-day AST (%)',ylim=(0,62));ax[2].legend(frameon=False,loc='upper left')
    clean(ax[3]);ax[3].set_xlim(0,1);ax[3].set_ylim(0,1)
    steps=[(.89,'FAST calculated from LSM, CAP\nand the AST in hand','-'),(.645,'Report AST date and value\nand interval from the scan','-'),
           (.40,'Required AST within RCV?\nNo: category robust to\nbiological variation','--'),(.155,'Yes, AST nonconcurrent: recommend\nsame-day AST. Yes, same-day:\nreport as near a threshold',':')]
    for y,txt,ls in steps:
        ax[3].add_patch(Rectangle((.03,y-.10),.94,.20,facecolor='white',edgecolor='black',lw=.8,ls=ls))
        ax[3].text(.5,y,txt,ha='center',va='center',fontsize=7.5)
    for y0,y1 in [(.745,.79),(.50,.545),(.255,.30)]:ax[3].annotate('',(.5,y0),(.5,y1),arrowprops={'arrowstyle':'->','color':'black','lw':.7})
    save(fig,'Figure_2','A 为达到 FAST 0.67 所需的 AST；B 为非同日/同日 AST 比值与 RCV 区间；C 为被标记与未标记结果的改判率；D 为建议的报告流程。')

# Supplement 1: laboratory sources, AST measurement and timing.
fig,ax=plate('Supplementary_Figure_1',['AST results by source','AST distribution by source','Collection to report',
                                     'Original and expanded support','Component ranges','AST by chemistry line'])
SRCN={'linked':'Repository','expanded':'Repository','liver_export_20261003':'Second extract'}
E['src']=E.source_kind.map(SRCN);sc=E.src.value_counts().reindex(['Repository','Second extract'])
ax[0].bar(sc.index,sc.values,color=[GREY,BLUE]);ax[0].set_ylabel('Retained AST results')
for x,v in enumerate(sc.values):ax[0].text(x,v*1.01,f'{v:,}',ha='center',va='bottom',fontsize=10)
ax[0].set_ylim(0,sc.max()*1.15)
bins=np.geomspace(5,1200,40)
for lab,col in [('Repository',GREY),('Second extract',BLUE)]:
    ax[1].hist(E.loc[E.src.eq(lab),'AST'],bins=bins,density=True,histtype='step',lw=2,color=col,label=f"{lab} (median {E.loc[E.src.eq(lab),'AST'].median():.0f})")
ax[1].set(xscale='log',xlabel='AST (U/L)',ylabel='Density');ax[1].legend(frameon=False,fontsize=9)
ch=T('21_collection_to_report_hours.csv');ct=SUMMARY['data']['collection_to_report']
ax[2].bar(ch.hours_bin_start+.5,ch.n,width=1,color=ORANGE);ax[2].axvline(24,color=GREY,ls=':')
ax[2].set(xlabel='Report time minus collection time (hours)',ylabel='AST results',xlim=(0,48))
ax[2].text(.97,.95,f"n={ct['AST_results_with_both_times']:,}\nmedian {ct['median_hours']:.1f} h\n≤24 h: {100*ct['within_24h']:.1f}%\nsame date: {100*ct['same_calendar_day']:.1f}%",transform=ax[2].transAxes,ha='right',va='top',fontsize=10)
for idx,label in enumerate(['submitted_baseline','linked']):
    g=t[(t.dataset==label)&t.rule.isin(['prior_90','same_day','symmetric_90'])].set_index('rule').loc[['prior_90','same_day','symmetric_90']]
    ax[3].bar(np.arange(3)+(idx-.5)*.36,g.scans,width=.34,color=[GREY,BLUE][idx],label={'submitted_baseline':'Submitted (v9)','linked':'Current linked'}[label])
ax[3].set_xticks(range(3),['Earlier','Same-day','±90']);ax[3].set_ylabel('Scans with FAST');ax[3].legend(frameon=False,fontsize=9)
rg=T('07_component_ranges.csv').query("population=='all_scans'")
clean(ax[4]);ax[4].text(0,.93,'Observed values (1st–99th percentile)',fontsize=12)
range_labels={'age_at_scan':'Age (years)','bmi':'BMI (kg/m²)','lsm':'LSM (kPa)','cap':'CAP (dB/m)'}
for i,r in enumerate(rg.itertuples()):ax[4].text(0,.70-.18*i,f'{range_labels[r.variable]}: {r.p01:.1f}–{r.p99:.1f}',fontsize=12)
li=pd.Series(SUMMARY['data']['liver_export_AST_instrument']).sort_values(ascending=False).head(2)
ax[5].bar(['Chemistry\nline 1','Chemistry\nline 2'],li.values,color=[BLUE,TEAL]);ax[5].set_ylim(0,li.max()*1.2);ax[5].set_ylabel('AST results, second extract')
ax[5].text(.5,.95,'Both lines report AST in U/L',transform=ax[5].transAxes,ha='center',va='top',fontsize=10)
save(fig,'Supplementary_Figure_1','两个检验来源的 AST 数量与分布；采血至报告时间间隔（中位 3 小时，99.6% 在 24 小时内）；两条生化流水线均以 U/L 报告。')

# Supplement 2: sensitivity forest.
fig,ax=plate('Supplementary_Figure_2',['Earlier AST sensitivity','Later AST sensitivity','Daily-median reference',
                                     'Earlier AST change','Later AST change','Between-day bracketing'])
ss=T('13_anchor_sensitivity.csv'); extra=T('20_ck_hbvdna.csv').query("dataset=='CK_not_high'")
ss=pd.concat([ss,extra],ignore_index=True)
short={'all':'All','first_anchor_per_patient':'First anchor','unique_remote_AST':'Each AST used once','AST_both_5_to_100':'AST 5–100','AST_both_5_to_200':'AST 5–200','strict_linkage':'Same-day linkage','CK_not_high':'CK ≤200 U/L'}
ss=ss[ss.dataset.isin(short)]
ss['label']=ss.dataset.map(short)
for i,direction in enumerate(['earlier','later']):
    g=ss[ss.direction.eq(direction)];forest(ax[i],g,'label');ax[i].set_xlabel('Observed reclassification (%)');ax[i].set_xlim(0,70)
daily=T('11_daily_median_sensitivity.csv');daily['label']=daily.direction.str.capitalize();forest(ax[2],daily,'label');ax[2].set_xlabel('Observed reclassification (%)')
for i,direction in [(3,'earlier'),(4,'later')]:
    g=ss[ss.direction.eq(direction)];forest(ax[i],g,'label',value='median_delta_FAST',low='median_delta_FAST_low',high='median_delta_FAST_high',percent=False)
    ax[i].set_xlim(-.17,.17);ax[i].axvline(0,color=GREY,ls=':');ax[i].set_xlabel('Median nonconcurrent minus same-day FAST')
br=pd.read_parquet(m.DATA/'bracket_comparisons.parquet');ax[5].scatter(br.FAST_earlier,br.FAST_later,s=15,color=BLUE,alpha=.6);ax[5].plot([0,1],[0,1],color=GREY,ls=':');ax[5].set(xlabel='Earlier FAST',ylabel='Later FAST',xlim=(0,1),ylim=(0,1));bs=SUMMARY['brackets'];ax[5].text(.04,.91,f"{bs['discordant']}/{bs['scans']} scans reclassified",transform=ax[5].transAxes,fontsize=11)
save(fig,'Supplementary_Figure_2','真实 AST 配对的稳健性检查：每人首个锚点、去掉重复 AST、去极端值、BMI 与代谢代理亚组，以及排除 CK 升高。')

# Supplement 3: actual trajectories and model discrimination.
fig,ax=plate('Supplementary_Figure_3',['Lag-band sample size','Reclassification by lag band','Observed AST ratios',
                                     'Observed FAST changes','Grouped cross-fitted loss','Grouped cross-fitted discrimination'])
lg=T('14_actual_reclassification_strata.csv').query("stratum=='lag_band'");ax[0].bar(lg.level,lg.n,color=BLUE);ax[0].set(xlabel='Absolute interval (days)',ylabel='Comparisons')
forest(ax[1],lg,'level');ax[1].set_xlabel('Observed reclassification (%)')
for i,var in [(2,'delta_log_AST'),(3,'delta_FAST')]:
    ax[i].hist([A.loc[A.direction.eq(d),var] for d in ['earlier','later']],bins=18,label=['Earlier','Later'],color=[BLUE,ORANGE],alpha=.65)
    ax[i].axvline(0,color=GREY);ax[i].set(xlabel='Log nonconcurrent / same-day AST' if i==2 else 'Nonconcurrent minus same-day FAST',ylabel='Comparisons');ax[i].legend(frameon=False)
mod=T('04_cross_fitted_model_comparison.csv')
for i,var in [(4,'Brier'),(5,'AUC')]:
    ax[i].bar(['Boundary\n+ AST','+ Timing\n+ BMI'],mod[var],color=[BLUE,ORANGE]);ax[i].set_ylabel(var);ax[i].set_ylim(0,.30 if var=='Brier' else 1)
save(fig,'Supplementary_Figure_3','不强行拟合统一的时间偏差曲线。按实际样本量呈现各时间段；患者分组交叉验证表明，增加时间方向、间隔和 BMI 未改善 Brier 分数。')

# Supplement 4: etiology, metabolic and anthropometric subgroups.
fig,ax=plate('Supplementary_Figure_4',['Linked scans by subgroup','Same-day AST share','Later AST, ±90 d',
                                     'Shared AST, ±90 d','Earlier AST changed','Later AST changed'])
SUBN={'all':'All linked','HBsAg_positive':'HBsAg positive','viral_markers_negative':'Viral markers negative','MASLD_compatible':'MASLD-compatible','diabetes_any':'Diabetes','BMI_ge25':'BMI ≥25','metabolic_steatosis_275':'CAP ≥275 + BMI ≥23','metabolic_steatosis':'CAP ≥275 + BMI ≥23'}
sub=T('06_subgroup_support.csv');sub['label']=sub.subgroup.map(SUBN)
for i,var,denom in [(0,'linked_scans',None),(1,'same_day','linked_scans'),(2,'later','wide'),(3,'reuse','wide')]:
    vals=sub[var] if denom is None else 100*sub[var]/sub[denom]
    ax[i].barh(sub.label,vals,color=BLUE);ax[i].invert_yaxis();ax[i].set_xlabel('Scans' if denom is None else '% of relevant scans')
ss=T('13_anchor_sensitivity.csv');ss=ss[ss.dataset.isin(SUBN)].copy();ss['label']=ss.apply(lambda r:f"{SUBN[r.dataset]} ({r.discordant}/{r.comparisons})",axis=1)
for i,direction in [(4,'earlier'),(5,'later')]:
    g=ss[ss.direction.eq(direction)];forest(ax[i],g,'label');ax[i].set_xlabel('Reclassified comparisons (%)');ax[i].set_xlim(0,65);ax[i].tick_params(axis='y',labelsize=9)
save(fig,'Supplementary_Figure_4','按实验室病因（HBsAg、病毒标志物阴性、符合MASLD定义）、糖尿病与BMI分层；各亚组较早/较晚AST的改判方向一致。')

# Supplement 5: external mathematical robustness retained from validated sources.
fig,ax=plate('Supplementary_Figure_5',['AST +25% reclassification','Rule-in unreachable','Indeterminate stability',
                                     'Rule-out stability','Rule-in stability','Acquisition settings'])
ex=EXT.query("category=='indeterminate'").copy();ex['label']=ex.cohort
forest(ax[0],ex,'label',value='estimate_plus25',low='ci_low_plus25',high='ci_high_plus25');ax[0].set_xlabel('Reclassified (%)')
forest(ax[1],ex,'label',value='estimate_unreachable',low='ci_low_unreachable',high='ci_high_unreachable');ax[1].set_xlabel('Rule-in unreachable (%)')
for i,c in [(2,'indeterminate'),(3,'rule_out'),(4,'rule_in')]:
    g=EXT_STABLE[EXT_STABLE.category.eq(c)].copy();g['label']=g.cohort;forest(ax[i],g,'label',value='estimate',low='ci_low',high='ci_high');ax[i].set_xlabel('Retained under ±25% AST (%)');ax[i].set_xlim(0,110)
clean(ax[5]);ax[5].text(0,.92,'Six separate clinical populations',fontweight='bold')
for i,txt in enumerate(['NHANES: same research visit','Japan: contemporaneous components','CORONASH: same calendar day','Singapore T2DM: single study visit','Germany CAD: short preset window']):ax[5].text(0,.73-i*.15,txt,fontsize=11)
save(fig,'Supplementary_Figure_5','六队列仅验证公式敏感性与边界结构。NHANES 保留复杂抽样估计；其余为队列独立区间；无跨队列合并比例或时间窗因果比较。')

# Supplement 6: algebra, full component range, margin distributions.
fig,ax=plate('Supplementary_Figure_6',['FAST equation','Threshold geometry','Exact AST-only change',
                                     'FAST dependence on AST','LSM–CAP scan density','Local boundary margins'])
for i in range(3):clean(ax[i])
ax[0].text(.01,.69,r'$\eta=-1.65+1.07\log L$',fontsize=20);ax[0].text(.01,.43,r'$\quad+2.66\times10^{-8}C^3-63.3/A$',fontsize=19);ax[0].text(.01,.15,r'$\mathrm{FAST}=\mathrm{expit}(\eta)$',fontsize=19)
ax[1].text(.01,.73,r'$\mathrm{FAST}_{\max}=\mathrm{expit}(B)$',fontsize=19);ax[1].text(.01,.44,r'$A_c=\frac{63.3}{B-\mathrm{logit}(c)}$',fontsize=22);ax[1].text(.01,.15,r'$B>\mathrm{logit}(c)$',fontsize=19)
ax[2].text(.01,.70,r'$\Delta\eta=63.3(1/A_0-1/A_\tau)$',fontsize=18);ax[2].text(.01,.39,r'$\frac{\partial\eta}{\partial A}=\frac{63.3}{A^2}$',fontsize=24);ax[2].text(.01,.10,r'$\frac{\partial\eta}{\partial\log A}=\frac{63.3}{A}$',fontsize=24)
aa=np.geomspace(5,1000,200)
for l,c,col in [(5,230,BLUE),(8,280,ORANGE),(12,310,BERRY)]:ax[3].plot(aa,m.fast(l,c,aa),color=col,label=f'LSM {l}, CAP {c}')
for c in [.35,.67]:ax[3].axhline(c,color=GREY,ls=':')
ax[3].set(xscale='log',xlabel='AST (U/L)',ylabel='FAST',ylim=(0,1));ax[3].legend(frameon=False,fontsize=10)
ax[4].hexbin(S.cap,S.lsm,gridsize=35,bins='log',cmap='Blues',mincnt=1);ax[4].set(xlabel='CAP (dB/m)',ylabel='LSM (kPa)',ylim=(2,30))
same=SEL['same_day'];b=-1.65+1.07*np.log(same.lsm)+2.66e-8*same.cap**3;ds=[]
for c in [.35,.67]:
    dif=b-logit(c);ds.append(np.abs(np.log(np.where(dif>0,63.3/dif,np.nan)/same.AST)))
margin=np.nanmin(ds,axis=0)
for c in m.LABELS:ax[5].hist(margin[same.category.eq(c)],bins=np.linspace(0,3,20),histtype='step',lw=1.7,color=PAL[c],label=NAMES[c])
ax[5].set(xlabel='Nearest log-AST margin',ylabel='Same-day scans');ax[5].legend(frameon=False)
save(fig,'Supplementary_Figure_6','公式给出精确边界和局部导数；图示曲线为预设示例。数学不变性与真实跨日变化分别呈现，不把公式特征包装成诊断验证。')

# Independent SVG groups, namespaced IDs, Chinese explanations below each plate.
NS='http://www.w3.org/2000/svg'; INK='http://www.inkscape.org/namespaces/inkscape'
def element(tag,**kw):return ET.Element('{'+NS+'}'+tag,**{k:str(v) for k,v in kw.items()})
width=1900;tilew=900;tileh=760;totalh=330+5*tileh+950
root=ET.Element('{'+NS+'}svg',nsmap={None:NS,'inkscape':'http://www.inkscape.org/namespaces/inkscape'},width=str(width),height=str(totalh),viewBox=f'0 0 {width} {totalh}')
root.append(element('rect',x=0,y=0,width=width,height=totalh,fill='white'))
def text(parent,x,y,content,size=25):
    t=element('text',x=x,y=y,fill='#171717',**{'font-family':'Arial, PingFang SC, sans-serif','font-size':size});t.text=content;parent.append(t)
text(root,35,55,'VCTE–FAST v16 (CCA)：2 张主图 + 8 张补充图 + 数学公式',34)
text(root,35,103,'蓝色：较早 AST / rule-out；橙色：较晚 AST / indeterminate；莓红：rule-in；灰色：未捕获 / 参照',23)
text(root,35,150,'数值来自 2026-10-03 v13 双来源实验室连接冻结集。每张组图及内部面板为独立 SVG 组；原投稿图未覆盖。',22)
for i,item in enumerate(MAN):
    x=35+(i%2)*940;y=215+(i//2)*tileh
    group=element('g',id=f'board_{item["name"]}');root.append(group)
    text(group,x,y,item['name'].replace('_',' '),26)
    svg=ET.parse(str(F/f'{item["name"]}.svg')).getroot();vb=list(map(float,svg.get('viewBox').split()))
    ids={el.get('id'):f'{item["name"]}_{el.get("id")}' for el in svg.iter() if el.get('id')}
    for el in svg.iter():
        for key,value in list(el.attrib.items()):
            if key=='id':el.set(key,ids[value])
            else:
                value=re.sub(r'url\(#([^)]*)\)',lambda mt:f'url(#{ids.get(mt.group(1),mt.group(1))})',value)
                if value.startswith('#') and value[1:] in ids:value='#'+ids[value[1:]]
                el.set(key,value)
    svg.set('x',str(x));svg.set('y',str(y+20));svg.set('width',str(tilew));svg.set('height','600')
    group.append(svg)
    # Keep English terms intact and leave clearance below the SVG viewport.
    chunks=[];line=''
    for token in re.findall(r'[A-Za-z0-9]+|.',item['cn']):
        if len(line)+len(token)>37:chunks.append(line);line=''
        line+=token
    if line:chunks.append(line)
    for j,line in enumerate(chunks):text(group,x,y+660+j*25,line,20)

formula_fig,formula_ax=plt.subplots(figsize=(24,8));formula_ax.set_axis_off()
formula_list=[r'$\eta=-1.65+1.07\log L+2.66\times10^{-8}C^3-63.3/A$',r'$\mathrm{FAST}=\mathrm{expit}(\eta),\quad B=-1.65+1.07\log L+2.66\times10^{-8}C^3$',
 r'$\sup_{A>0}\mathrm{FAST}=\mathrm{expit}(B),\quad A_c=63.3/[B-\mathrm{logit}(c)]$',r'$D_c=\log(A_c/A),\quad R=\min_{c\in\mathcal{B}}|D_c|$',
 r'$\Delta\eta_{AST}=63.3(1/A_0-1/A_\tau),\quad\Delta\mathrm{FAST}=\mathrm{FAST}_\tau-\mathrm{FAST}_0$',
 r'$\partial\eta/\partial A=63.3/A^2,\quad\partial\eta/\partial\log A=63.3/A$',
 r'$\Delta\eta=1.07\log(L_2/L_1)+2.66\times10^{-8}(C_2^3-C_1^3)+63.3(1/A_1-1/A_2)$',
 r'$L_c(C)=\exp\{[\mathrm{logit}(c)+1.65-2.66\times10^{-8}C^3]/1.07\}$']
for i,t in enumerate(formula_list):formula_ax.text(.01,.97-i*.12,t,fontsize=24,va='top')
with matplotlib.rc_context({'svg.fonttype':'path'}):formula_fig.savefig(F/'Formula_Shelf.svg')
plt.close(formula_fig)
fx=ET.parse(str(F/'Formula_Shelf.svg')).getroot();y=215+5*tileh
text(root,35,y,'可拆取公式区：公式为矢量数学字形；图表标签保留文本',28)
fx.set('x','35');fx.set('y',str(y+30));fx.set('width','1830');fx.set('height','610')
# Namespace the formula glyph IDs as well.
ids={el.get('id'):'formula_'+el.get('id') for el in fx.iter() if el.get('id')}
for el in fx.iter():
    for key,val in list(el.attrib.items()):
        if key=='id':el.set(key,ids[val])
        else:
            val=re.sub(r'url\(#([^)]*)\)',lambda mt:f'url(#{ids.get(mt.group(1),mt.group(1))})',val)
            if val.startswith('#') and val[1:] in ids:val='#'+ids[val[1:]]
            el.set(key,val)
root.append(fx)
ET.ElementTree(root).write(str(F/'VCTE_FAST_v16_All_Figures_CN.svg'),encoding='utf-8',xml_declaration=True)
(F/'FIGURE_MANIFEST.json').write_text(json.dumps(MAN,ensure_ascii=False,indent=2),encoding='utf8')
allids=[el.get('id') for el in root.iter() if el.get('id')];assert len(allids)==len(set(allids))
print(f'Generated {len(MAN)} figures and the editable Chinese board')
