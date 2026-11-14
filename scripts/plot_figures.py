"""Reproduce manuscript Figures 2–5, preserving metric definitions and error bars."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from _paths import FIGURES, result_file
OUT=FIGURES; OUT.mkdir(parents=True,exist_ok=True)
mg=pd.read_csv(result_file('tables/main_original_metrics_ci.csv'))
sg=pd.read_csv(result_file('tables/sensitivity_original_metrics_ci.csv'))
SC=['low_load','medium_load','high_load','burst_load','extended_workload']; SL=['Low','Medium','High','Burst','Extended']
POL=['edge_only','cloud_only','hybrid_cloud_edge','deadline_aware_hybrid','utilization_based','network_aware','queue_length_based','deadline_guarded']
PL=['Edge-only','Cloud-only','Queue hybrid','Deadline (original)','Utilization','Network-aware','Queue length','Deadline (guarded)']
M=['avg_response_time_ms','avg_waiting_time_ms','throughput_tasks_per_s','blocking_probability','cloud_offloading_ratio','sla_violation_rate','avg_cost_proxy','avg_energy_proxy','on_time_completion_rate']
MNAME=['Response time','Waiting time','Throughput','Blocking probability','Cloud offloading ratio','SLA violation rate','Cost proxy','Energy proxy','On time / offered']
F=[1,1,1,100,100,100,1,1,100];D=[2,2,2,2,2,2,5,4,2]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.2,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,'axes.labelsize':9.3,'axes.titlesize':10,'xtick.labelsize':8.4,'ytick.labelsize':8.1})
def dump(fig,n):
    for ext in ['svg','pdf','png']:fig.savefig(OUT/f'figure{n}.{ext}',dpi=330,facecolor='white')
    plt.close(fig)
colors=['#333333','#BD5710','#007681','#835598']; markers=['o','s','^','D']
def drawmain(n,ms,titles):
    fig,axs=plt.subplots(1,3,figsize=(7.4,2.65));fig.subplots_adjust(left=.10,right=.985,top=.72,bottom=.24,wspace=.39)
    h=[]
    for ax,m,title,letter in zip(axs,ms,titles,'abc'):
        for j,policy in enumerate(POL[:4]):
            p=mg[(mg.metric==m)&(mg['mode']==policy)&(mg.network_model=='independent')].set_index('scenario').loc[SC]
            label=['Edge-only','Cloud-only','Queue hybrid','Deadline (original)'][j]
            ln=ax.errorbar(np.arange(5)+(j-1.5)*.15,p['mean'],yerr=p.halfwidth,fmt=markers[j],color=colors[j],ms=4.2,mew=.8,elinewidth=.95,capsize=1.8,label=label)
            if ax==axs[0]:h.append(ln)
        ax.set_title('('+letter+') '+title,loc='left',pad=7);ax.set_xticks(np.arange(5),['Low','Med.','High','Burst','Ext.'],rotation=30,ha='right');ax.set_xlim(-.48,4.51);ax.grid(axis='y',alpha=.22,linewidth=.65)
        if m in M[:2]:
            assert p.lower.min()>0;ax.set_yscale('log');ax.set_ylabel('ms (log scale)');ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:g}'))
        elif m==M[2]:ax.set_ylabel('tasks/s');ax.set_ylim(bottom=0)
        else:ax.set_ylabel('%');ax.set_ylim(-4,104);ax.set_yticks([0,25,50,75,100])
    fig.legend(h,[a.get_label() for a in h],loc='upper center',bbox_to_anchor=(.5,.99),ncol=4,columnspacing=1.1,handletextpad=.28,fontsize=9.3)
    dump(fig,n)
drawmain(2,[M[0],M[1],M[2]],['Response time','Compute waiting','Throughput'])
drawmain(3,[M[3],M[5],M[4]],['Blocking','Late completions','Cloud share'])
def drawsweep(n,param,xlabel):
    fig,axs=plt.subplots(1,2,figsize=(7.4,2.7));fig.subplots_adjust(left=.09,right=.985,top=.73,bottom=.24,wspace=.31)
    h=[]
    for ax,m,title in zip(axs,[M[0],M[5]],['(a) Response time','(b) Late completions']):
        for j,sc in enumerate(SC[2:4]):
            p=sg[(sg.scenario==sc)&(sg.metric==m)&(sg.parameter==param)&(sg.network_model=='independent')].sort_values('level')
            ln=ax.errorbar(p.level,p['mean'],yerr=p.halfwidth,fmt='o-' if j==0 else 's--',capsize=2,linewidth=1.4,ms=4,color=['#007681','#BD5710'][j],label=['High load','Burst load'][j])
            if ax==axs[0]:h.append(ln)
        ax.set_xticks(p.level);ax.set_xlabel(xlabel);ax.set_ylabel('ms' if m==M[0] else '% of completions');ax.set_title(title,loc='left',pad=6);ax.grid(axis='y',alpha=.22,linewidth=.65);ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.legend(h,[a.get_label() for a in h],loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,fontsize=9.6)
    dump(fig,n)
drawsweep(4,'threshold_ms','Edge waiting threshold (ms)')
drawsweep(5,'propagation_multiplier','Propagation multiplier')
print('Regenerated Figures 2–5 from run-level result summaries.')
