"""Regenerate Figs. 2-3 of the BIBM 2026 paper from data/ and results/."""
import sys, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from decimal import Decimal, ROUND_HALF_UP
def fmt(v,n=3): return str(Decimal(str(v)).quantize(Decimal("1."+"0"*n),rounding=ROUND_HALF_UP))
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Liberation Serif"],"font.size":8,
 "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#555","axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7})
INK="#1a1a1a"; BLUE="#2a78d6"; ORANGE="#eb6834"; GRAY="#8c8c8c"
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J=lambda *p: os.path.join(ROOT,*p)
DATA=J("data","combined_with_label.csv"); OUT=J("figures","fig2_main.png"); BASE=J("results","baseline_results.csv"); LLM=J("results","llm_results.csv")
llm=pd.read_csv(LLM); base=pd.read_csv(BASE); d=pd.read_csv(DATA)
rows=[(r.model,"llm",r.precision,r.recall,r.f1,0.0) for r in llm.itertuples()]
rows+=[(r.model,"prior" if r.model=="Kinase prior" else "sup",r.precision,r.recall,r.f1,r.f1_sd) for r in base.itertuples()]
T=pd.DataFrame(rows,columns=["model","cat","p","r","f1","sd"]).sort_values("f1")
fig=plt.figure(figsize=(3.45,5.3),dpi=600)
gs=fig.add_gridspec(1,1,left=0.30,right=0.975,bottom=0.535,top=0.885)
ax=fig.add_subplot(gs[0]); y=np.arange(len(T))
col={"llm":BLUE,"sup":ORANGE,"prior":GRAY}
ax.barh(y,T.f1,height=0.66,color=[col[c] for c in T.cat],edgecolor="white",lw=0.5,zorder=3)
m=T.sd.values>0
ax.errorbar(T.f1.values[m],y[m],xerr=T.sd.values[m],fmt="none",ecolor="#444",elinewidth=0.7,capsize=1.8,capthick=0.7,zorder=5)
prior_f=float(T[T.cat=="prior"].f1.iloc[0])
for i,(f,s) in enumerate(zip(T.f1,T.sd)): ax.text(max(f+(s if s>0 else 0),prior_f)+0.010,i,fmt(f),va="center",fontsize=6.6,color=INK)
ax.set_yticks(y); ax.set_yticklabels(T.model,fontsize=7.3); ax.tick_params(axis="y",length=0)
prior=float(T[T.cat=="prior"].f1.iloc[0])
ax.axvline(prior,color=GRAY,ls=(0,(3,2)),lw=0.8,zorder=4)
ax.set_xlim(0,0.80); ax.set_xticks([0,0.2,0.4,0.6,0.8]); ax.set_xlabel("F1-score")
ax.grid(True,axis="x",color="#ececec",lw=0.5,zorder=0); ax.spines["left"].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label="RAG-LLM (zero-shot)"),Patch(color=ORANGE,label="Supervised (5-fold CV)"),Patch(color=GRAY,label="Kinase prior")],
          ncol=2,frameon=False,loc="lower left",bbox_to_anchor=(-0.42,1.0),fontsize=6.6,handlelength=1.0,columnspacing=0.8)
# (b) per-kinase positive rate
ax=fig.add_axes([0.155,0.075,0.82,0.335])
k=d.groupby("kinase").label.agg(["count","mean"]); k=k[k["count"]>=5]
bins=np.linspace(0,1,11); h,_=np.histogram(k["mean"],bins=bins)
cols=[BLUE if b<0.2-1e-9 else (ORANGE if b>=0.8-1e-9 else "#b9b9b9") for b in bins[:-1]]
ax.bar(bins[:-1]+0.05,h,width=0.092,color=cols,edgecolor="white",lw=0.5,zorder=3)
for x_,v in zip(bins[:-1]+0.05,h):
    if v: ax.text(x_,v+0.8,str(v),ha="center",fontsize=6.5,color=INK)
lo=int((k["mean"]<0.2).sum()); hi=int((k["mean"]>=0.8).sum())
ax.text(0.31,max(h)*0.90,f"{lo} kinases with < 20% positives",fontsize=6.8,color=BLUE,ha="left")
ax.text(0.98,max(h)*0.45,f"{hi} kinases with\n\u2265 80% positives",fontsize=6.8,color=ORANGE,ha="right")
ax.set_xlim(0,1); ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0]); ax.set_xlabel("Fraction of positive pairs per kinase")
ax.set_ylabel(f"Kinases (n = {len(k)})"); ax.set_ylim(0,max(h)*1.12)
ax.grid(True,axis="y",color="#ececec",lw=0.5,zorder=0)
fig.text(0.01,0.992,"(a)",fontsize=9.5,weight="bold",va="top"); fig.text(0.085,0.992,"Training-free RAG-LLMs vs. supervised models",fontsize=8,va="top",style="italic")
fig.text(0.01,0.462,"(b)",fontsize=9.5,weight="bold",va="top"); fig.text(0.085,0.462,"Kinase-level label polarization",fontsize=8,va="top",style="italic")
fig.savefig(OUT,dpi=600)

# ---------------- Fig. 3: context source (training vs validation) ----------------
plt.close("all")
cs=pd.read_csv(J("results","context_source_results.csv"))
D={}
for r in cs.itertuples(): D.setdefault(r.context,{})[r.subset]=[r.precision,r.recall,r.f1]
fig,axs=plt.subplots(1,2,figsize=(3.45,2.35),dpi=600,sharey=True)
fig.subplots_adjust(left=0.13,right=0.99,bottom=0.13,top=0.80,wspace=0.08)
TR="#a9c8ee"; VA="#2a78d6"; x=np.arange(3); w=0.36
for ax,(key,ttl) in zip(axs,[("protein_uniprot","Protein (UniProt)"),("gene_pubmed","Gene (PubMed)")]):
    t=np.array(D[key]["development"]); v=np.array(D[key]["held_out"])
    ax.bar(x-w/2,t,w*0.92,color=TR,edgecolor="white",lw=0.5,zorder=3,label="Development")
    ax.bar(x+w/2,v,w*0.92,color=VA,edgecolor="white",lw=0.5,zorder=3,label="Held-out")
    for i in range(3):
        ax.text(x[i]-w/2,t[i]+0.008,fmt(t[i],2),ha="center",fontsize=5.8,color="#444")
        ax.text(x[i]+w/2,v[i]+0.008,fmt(v[i],2),ha="center",fontsize=5.8,color=INK)
        ax.text(x[i],0.86,"\u0394 \u2212"+fmt(t[i]-v[i]),ha="center",fontsize=6.2,color="#333")
    ax.set_xticks(x); ax.set_xticklabels(["Precision","Recall","F1"],fontsize=7)
    ax.set_title(ttl,fontsize=8.5,weight="bold",pad=12)
    ax.set_ylim(0,0.93); ax.set_yticks([0,0.2,0.4,0.6,0.8]); ax.grid(True,axis="y",color="#ececec",lw=0.5,zorder=0); ax.tick_params(axis="x",length=0)
axs[0].set_ylabel("Score"); axs[1].spines["left"].set_visible(False); axs[1].tick_params(axis="y",length=0)
axs[0].legend(ncol=2,frameon=False,loc="lower left",bbox_to_anchor=(0.35,1.13),fontsize=7,handlelength=1.1)
fig.savefig(J("figures","fig3_context_source.png"),dpi=600)
print("figures written to", J("figures"))
