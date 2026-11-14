"""Reproduce the routing and finite-queue framework in Figure 1."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
from _paths import FIGURES
O=FIGURES; O.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(9.6,7.2));fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,100),ylim=(0,75),aspect='auto');ax.axis('off')
line='#263242'; blue='#eaf2fb'; green='#eaf5ed'; cream='#fff7e8'
def box(x,y,w,h,fill,head,sub='',fs=11.4,sfs=10.1):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.18,rounding_size=0.8',fc=fill,ec=line,lw=1.05);ax.add_patch(p)
 if sub:
  ax.text(x+w/2,y+h*.69,head,ha='center',va='center',weight='bold',size=fs,color='#18232e')
  ax.text(x+w/2,y+h*.305,sub,ha='center',va='center',size=sfs,color='#18232e',linespacing=1.06)
 else: ax.text(x+w/2,y+h/2,head,ha='center',va='center',size=fs,weight='bold',color='#18232e')
def path(pts,end=True,c=line):
 for t,(a,b) in enumerate(zip(pts,pts[1:])):
  if t==len(pts)-2 and end: ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12.7,linewidth=1.15,color=c,shrinkA=0,shrinkB=0))
  else:ax.add_line(Line2D([a[0],b[0]],[a[1],b[1]],color=c,lw=1.15))
ax.text(50,72.4,'Proposed Cloud–Edge Workload Handling Framework',ha='center',va='center',weight='bold',size=17,color='#122335')
for x,w,h,s in [(2,20,'Workload sources','IoT, sensors, users,\nreal-time applications'),(26,20,'Task features','Arrival, demand, payload,\ndeadline, propagation'),(50,20,'Queue state','Current tier occupancy\nand dispatched uploads'),(74,24,'Processing mode','Four focal routing policies')]:box(x,61.3,w,8.2,blue if x!=74 else cream,h,s,11.8,9.85)
for a,b in [(22,26),(46,50),(70,74)]:path([(a+.2,65.5),(b-.2,65.5)])
ax.add_patch(FancyBboxPatch((2,43.0),96,14.9,boxstyle='round,pad=.2,rounding_size=.8',fc='#fcf8f0',ec=line,lw=1.05))
ax.text(50,55.6,'Decision layer',size=13.0,weight='bold',ha='center',va='center',color=line)
path([(86,61.1),(86,58.2)])
for x,head,txt in [(3.9,'Edge-only','Request edge'),(27.5,'Baseline hybrid','Wait threshold and\nresponse comparison'),(51.1,'Deadline-aware hybrid','Feasibility, slack\nand predicted lateness'),(74.7,'Cloud-only','Request cloud; admit\nafter transmission')]:box(x,44.0,21.4,8.9,'#ffffff',head,txt,10.65,10.0)
edge=19.6;cloud=80.4
path([(14.6,43.8),(14.6,35.3)]);path([(85.4,43.8),(85.4,35.3)])
path([(38.2,43.8),(38.2,40.5),(49.95,40.5)],False);path([(61.8,43.8),(61.8,40.5),(49.95,40.5)],False)
path([(50,40.5),(50,38.8),(30,38.8),(30,35.3)]);path([(50,38.8),(70,38.8),(70,35.3)])
ax.text(31.6,39.75,'edge',size=9.6,ha='center',va='center',color=line);ax.text(68.4,39.75,'cloud',size=9.6,ha='center',va='center',color=line)
box(5,25.3,39,9.6,blue,'Edge layer','Finite compute queue and parallel servers;\nlocal admission at source arrival',12.3,10.7)
box(56,25.3,39,9.6,green,'Cloud layer','Finite compute queue and parallel servers;\nadmission after upload and propagation',12.3,10.7)
for x,fc,lab in [(4,blue,'Completed\nedge tasks'),(28,'#fbe9e6','Blocked\nedge tasks'),(52,green,'Completed\ncloud tasks'),(76,'#fbe9e6','Blocked\ncloud tasks')]:
 box(x,15.2,20,6.3,fc,lab,fs=11.0);path([(x+10,25.1),(x+10,21.8)]);path([(x+10,15),(x+10,11.5)])
box(4,3,92,8.1,'#f2edf9','Performance evaluation','Response and waiting time, throughput, utilization, queue length, blocking, offloading,\ndeadline outcomes, cost proxy and energy proxy',12.2,10.3)
ax.text(50,1.05,'Policies receive the same exogenous workload and system configuration within each comparison.',ha='center',va='center',size=9.7,style='italic',color=line)
for ext in ('pdf','svg','png'):fig.savefig(O/f'figure1.{ext}',dpi=320,facecolor='white')
