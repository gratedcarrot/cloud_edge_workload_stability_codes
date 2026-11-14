// C++17 equivalent of the frozen event model. A prefix state belongs ONLY to
// already-dispatched uploads. Inserting an upload before an existing arrival
// invalidates and recomputes that suffix, preserving genuine arrival order.
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <limits>
#include <utility>
#include <cstddef>
using std::vector;
constexpr int NC=25;
enum Out {DEST,OK,START,FINISH,SERVICE,WAIT,RESP,NWAIT,UPLOAD,ARRIVEC,RET,COST,ENERGY,REASON,CHANGED,EPRED,CPRED,CPWAIT,EQ,CQ,EB,CB,EW,ACTQ,INFLIGHT};
struct Layer{
 vector<double> ends, starts;
 Layer(int n=1):ends(n,0){}
 void purge(double t){starts.erase(starts.begin(),std::upper_bound(starts.begin(),starts.end(),t));}
 int q(double t)const{return starts.end()-std::upper_bound(starts.begin(),starts.end(),t);}
 int busy(double t)const{return std::count_if(ends.begin(),ends.end(),[=](double s){return s>t;});}
 double wait(double t)const{return std::max(0.,*std::min_element(ends.begin(),ends.end())-t);}
 bool avail(double t,int cap)const{return *std::min_element(ends.begin(),ends.end())<=t || q(t)<cap;}
 double sched(double t,double service,int cap){
   purge(t); auto it=std::min_element(ends.begin(),ends.end()); double start=std::max(t,*it);
   if(start>t && (int)starts.size()>=cap)return -1.;
   if(start>t)starts.push_back(start); // FCFS implies monotone service-start times.
   *it=start+service;return start;
 }
};
struct Pending{int id;double service;Layer after;};
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
extern "C" EXPORT int sim_v2(int n,const double *x, const double *cfg,int pol,int shared,double*out){
try {
 const int en=cfg[0], cn=cfg[1], ec=cfg[2],cc=cfg[3];
 const double er=cfg[4],cr=cfg[5],bw=cfg[6],threshold=cfg[7],div=cfg[8],ecost=cfg[9],ccost=cfg[10],ncost=cfg[11],een=cfg[12],cen=cfg[13],nen=cfg[14],wc=cfg[15],we=cfg[16],gap=cfg[17],slack=cfg[18],ut=cfg[19],nt=cfg[20],probe=cfg[21],bt=cfg[23];
 const bool published=cfg[22]>0.5, proposed=(pol==2||pol==3||pol==7);
 double next_free=0.,nan=std::numeric_limits<double>::quiet_NaN();
 Layer edge(en),cloud(cn);using Key=std::pair<double,int>;std::map<Key,Pending> pending;
 std::fill(out,out+n*NC,nan);
 auto execute=[&](int i,int dst,Layer &L,double t,double sv,int cap){
   const double *a=x+5*i; double *r=out+NC*i;
   r[ACTQ]=L.q(t);double start=L.sched(t,sv,cap);r[OK]=(start>=0);
   double size=(dst==1?a[1]/div:0);
   double s=(start>=0?sv:0);
   r[COST]=(dst==0?ecost:ccost)*s+ncost*size;r[ENERGY]=(dst==0?een:cen)*s+nen*size;
   if(start>=0){r[START]=start;r[FINISH]=start+sv+r[RET];r[WAIT]=start-t;r[RESP]=r[FINISH]-a[0]*1000.;r[SERVICE]=sv;}
 };
 auto drain=[&](double t){
   while(!pending.empty()&&pending.begin()->first.first<=t){
     auto it=pending.begin();execute(it->second.id,1,cloud,it->first.first,it->second.service,cc);
     pending.erase(it);
   }
 };
 for(int i=0;i<n;i++){
   const double*a=x+5*i;double*r=out+NC*i;double now=a[0]*1000.,kb=a[1],mi=a[2],d=a[3],prop=a[4];drain(now);
   edge.purge(now);cloud.purge(now);r[EQ]=edge.q(now);r[CQ]=cloud.q(now);r[EB]=edge.busy(now);r[CB]=cloud.busy(now);r[INFLIGHT]=pending.size();
   double es=std::max(5.,1000./er*std::max(1.,mi)/100.),cs=std::max(5.,1000./cr*std::max(1.,mi)/100.);
   double nw=(shared?std::max(0.,next_free-now):0.),up=kb*8./bw,arrival=now+nw+up+prop,ret=std::max(5.,prop*0.65);
   double ew=edge.wait(now),ep=ew+es,cp=nan,cwait=nan; bool ea=edge.avail(now,ec),ca=true;
   if(proposed){
     auto it=pending.upper_bound({arrival,std::numeric_limits<int>::max()});
     const Layer& before=(it==pending.begin()?cloud:std::prev(it)->second.after);
     cwait=before.wait(arrival); ca=before.avail(arrival,cc);
     cp=arrival-now+cwait+cs+ret;r[EPRED]=ep;r[CPRED]=cp;r[CPWAIT]=cwait;r[EW]=ew;
   }
   int sel=0, reason=0; r[CHANGED]=0;r[NWAIT]=0;r[UPLOAD]=0;r[RET]=0;
   if(pol==0){sel=0;reason=0;}
   else if(pol==1){sel=1;reason=1;}
   else if(pol==4){sel=(r[EB]/en>ut);reason=2;}
   else if(pol==5){
    double probe_kb=published?probe:128.;
    double probe_service=probe_kb*8./bw;
    double free_bw=probe_kb*8./(nw+probe_service);
    double util=std::max(0.,std::min(1.,1.-free_bw/bw));
    sel=published?(util<nt):(1000./(nw+probe_service)>bt);reason=3;
   }else if(pol==6){sel=((r[EB]+r[EQ])/en>(r[CB]+r[CQ])/cn);reason=4;}
   else{
     sel=(ea&&ew<=threshold&&ep<=cp)?0:1;reason=5;
     if(pol==3||pol==7){
       if(sel==1&&!ca){sel=ea?0:2;reason=6;}
       int old=sel;bool ef=ea&&ep<=d+1e-9,cf=ca&&cp<=d+1e-9;
       if(sel==0&&!ef&&cf){sel=1;reason=7;}
       else if(sel==1&&!cf&&ef){sel=0;reason=8;}
       else if(ef&&cf){
         if(std::abs(ep-cp)<=gap&&d-std::min(ep,cp)>=slack){
           double x0=ecost*es,y0=een*es,x1=ccost*cs+ncost*(kb/div),y1=cen*cs+nen*(kb/div);
           sel=(wc*x0+we*y0<=wc*x1+we*y1)?0:1;reason=(sel==0?9:10);r[CHANGED]=(old!=sel);
         }
       }else if(pol==3&&!ef&&!cf&&sel!=2){
         double le=ea?ep-d:INFINITY,lc=ca?cp-d:INFINITY;
         if(le<lc){sel=0;reason=11;}else if(lc<le){sel=1;reason=11;}
       }
     }
   }
   r[DEST]=sel;r[REASON]=reason;
   if(sel==0)execute(i,0,edge,now,es,ec);
   else if(sel==1){
     if(shared)next_free=now+nw+up;r[NWAIT]=nw;r[UPLOAD]=up;r[RET]=ret;r[ARRIVEC]=arrival;
     auto it=pending.emplace(Key{arrival,i},Pending{i,cs,Layer(1)}).first;
     if(proposed){
       // Correctly update even if newly inserted upload overtakes older ones.
       Layer prior=(it==pending.begin()?cloud:std::prev(it)->second.after);
       for(;it!=pending.end();++it){prior.sched(it->first.first,it->second.service,cc);it->second.after=prior;}
     }
   }else{r[OK]=0;r[COST]=r[ENERGY]=0;r[ACTQ]=r[EQ];}
 }
 drain(INFINITY);return 0;
}catch(...){return -1;}
}
