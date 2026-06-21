import argparse,json,os,glob
from typing import Dict,List,Optional,Tuple
import numpy as np
from offline_safe.dataset.replay_buffer_v2 import MAPDNTransitionDataset
from offline_safe.eval.evaluate_policy_v2 import evaluate_mabcq_lag_v2

def discover_epochs(ckpt_dir:str,min_epoch:int,max_epoch:int,stride:int)->List[str]:
    out=[]
    for p in glob.glob(os.path.join(ckpt_dir,"model_epoch_*.pt")):
        try: ep=int(os.path.basename(p).replace("model_epoch_","").replace(".pt",""))
        except: continue
        if min_epoch<=ep<=max_epoch and (ep-min_epoch)%stride==0: out.append(f"{ep:04d}")
    return sorted(set(out))

def load_dataset_and_action_space(data_roots:List[str],gamma:float,budget_quantile:float)->Tuple[MAPDNTransitionDataset,float,float]:
    dataset=MAPDNTransitionDataset(data_roots,normalize_obs=True,normalize_state=True,cache_in_memory=True,gamma=gamma,budget_quantile=budget_quantile)
    low,high=-1.0,1.0
    for root in data_roots:
        meta=os.path.join(root,"meta.json")
        if os.path.exists(meta):
            with open(meta,"r",encoding="utf-8") as f: m=json.load(f)
            try: low=float(np.min(m["action_low"]));high=float(np.max(m["action_high"]));break
            except: pass
    return dataset,low,high

def estimate_cost_limit(dataset,gamma=0.99,ratio=0.8,quantile=0.5)->float:
    vals=[]
    for epi in dataset.data:
        costs=epi["costs"].reshape(-1);vals.append(sum((gamma**t)*float(c) for t,c in enumerate(costs)))
    return ratio*float(np.quantile(vals,quantile))

def _convert_agg(agg):
    m={"mean_avg_return":"avg_return","mean_avg_cost":"avg_cost","mean_avg_discounted_cost":"avg_discounted_cost","mean_avg_v_out":"avg_v_out",
       "mean_avg_destroy":"avg_destroy","mean_avg_q_loss":"avg_q_loss","mean_avg_length":"avg_length","mean_cr":"cr","mean_pl":"pl","mean_v_dev":"v_dev",
       "mean_max_v_drop_dev":"max_v_drop_dev","mean_max_v_rise_dev":"max_v_rise_dev","mean_no_destroy_rate":"no_destroy_rate","mean_worst_episode_cost":"worst_episode_cost",
       "mean_worst_episode_v_out":"worst_episode_v_out","mean_cvar95_cost":"cvar95_cost","mean_constraint_satisfaction_rate":"constraint_satisfaction_rate"}
    return {dst:agg[src] for src,dst in m.items() if src in agg}

def load_baseline_summary(path)->Optional[Dict[str,float]]:
    if not path: return None
    with open(path,"r",encoding="utf-8") as f: data=json.load(f)
    if "final_test_result" in data and data["final_test_result"] is not None: return data["final_test_result"].get("summary")
    if "selected_best_on_validation" in data and data["selected_best_on_validation"] is not None:
        s=data["selected_best_on_validation"].get("summary")
        if s is not None: return s
    if "summary" in data: return data["summary"]
    if "aggregate_summary" in data: return _convert_agg(data["aggregate_summary"])
    return data

sf=lambda x,d=0.0: float(d if x is None else x)
rel_gap_higher=lambda base,val:max(0.0,(base-val)/max(abs(base),1e-6))
rel_gap_lower=lambda base,val:max(0.0,(val-base)/max(abs(base),1e-6))
rel_improve_lower=lambda base,val:(base-val)/max(abs(base),1e-6)

def mean_or_none(vals):
    vals=[v for v in vals if v is not None];return None if not vals else float(sum(vals)/len(vals))

def aggregate_window_summaries(window_results:List[Dict])->Optional[Dict]:
    if not window_results: return None
    keys=["avg_return","avg_cost","avg_discounted_cost","avg_v_out","avg_destroy","avg_q_loss","avg_length","cr","pl","v_dev","max_v_drop_dev","max_v_rise_dev","no_destroy_rate","worst_episode_cost","worst_episode_v_out","cvar95_cost","mean_cost_violation_ratio"]
    s={"eval_windows":len(window_results)}
    for k in keys:
        vals=[w["summary"].get(k,0.0) for w in window_results if w.get("summary") is not None];s[k]=float(sum(vals)/len(vals)) if vals else 0.0
    s["constraint_satisfaction_rate"]=mean_or_none([w["summary"].get("constraint_satisfaction_rate") for w in window_results if w.get("summary") is not None]);return s

def make_eval_score(summary,baseline_summary=None,soft_return_drop_ratio=0.10,max_return_drop_ratio=0.15)->float:
    avg_return,cr,viol,pl=sf(summary.get("avg_return"),-1e9),sf(summary.get("cr")),sf(summary.get("mean_cost_violation_ratio")),sf(summary.get("pl"),1e9)
    avg_q,avg_cost,avg_vout,cvar,worst,no_destroy=sf(summary.get("avg_q_loss"),1e9),sf(summary.get("avg_cost"),1e9),sf(summary.get("avg_v_out"),1e9),sf(summary.get("cvar95_cost"),1e9),sf(summary.get("worst_episode_cost"),1e9),sf(summary.get("no_destroy_rate"),0.0)
    if baseline_summary is None:
        return float(2.0*(-avg_return)+3.0*(1.0-cr)+5.0*viol+2.5*pl+1.5*avg_q+2.0*avg_cost+4.0*avg_vout+1.0*cvar+0.25*worst+10.0*(1.0-no_destroy))
    b_ret,b_cr,b_viol,b_pl,b_q,b_cost,b_vout,b_cvar,b_worst=[sf(baseline_summary.get(k),v) for k,v in
        [("avg_return",avg_return),("cr",cr),("mean_cost_violation_ratio",viol),("pl",pl),("avg_q_loss",avg_q),("avg_cost",avg_cost),("avg_v_out",avg_vout),("cvar95_cost",cvar),("worst_episode_cost",worst)]]
    ret_drop=rel_gap_higher(b_ret,avg_return);soft=max(0.0,ret_drop-soft_return_drop_ratio);hard=max(0.0,ret_drop-max_return_drop_ratio)
    safety_gain=2.0*max(0.0,rel_improve_lower(b_cost,avg_cost))+2.0*max(0.0,rel_improve_lower(b_vout,avg_vout))+1.5*max(0.0,rel_improve_lower(b_cvar,cvar))+1.0*max(0.0,rel_improve_lower(b_worst,worst))+1.0*max(0.0,(cr-b_cr)/max(abs(b_cr),1e-6))+1.5*max(0.0,rel_improve_lower(b_viol,viol))
    regress=2.0*rel_gap_higher(b_cr,cr)+3.0*rel_gap_lower(b_viol,viol)+1.0*rel_gap_lower(b_pl,pl)+0.5*rel_gap_lower(b_q,avg_q)+0.5*(1.0-no_destroy)
    return float(8.0*soft+20.0*hard+regress-6.0*safety_gain)

def build_gate_info(summary,baseline_summary=None,max_return_drop_ratio=0.15)->Dict[str,float]:
    avg_cost=sf(summary.get("avg_cost"),1e9);cr=sf(summary.get("cr"));pl=sf(summary.get("pl"),1e9)
    vout=sf(summary.get("avg_v_out"),1e9);no_destroy=sf(summary.get("no_destroy_rate"))
    viol=sf(summary.get("mean_cost_violation_ratio"))  # continuous violation ratio, 0 = no violation
    fail_destroy=int(no_destroy<1.0)
    if baseline_summary is None:
        fail_cost_safety=int(viol>0.50)
        fail_cr,fail_pl,fail_vout,fail_return=int(cr<0.85),int(pl>0.12),int(vout>0.02),0
    else:
        b_cost=sf(baseline_summary.get("avg_cost"),avg_cost);b_cr=sf(baseline_summary.get("cr"),cr)
        b_pl=sf(baseline_summary.get("pl"),pl);b_vout=sf(baseline_summary.get("avg_v_out"),vout)
        b_ret=sf(baseline_summary.get("avg_return"),sf(summary.get("avg_return")))
        ret_drop=rel_gap_higher(b_ret,sf(summary.get("avg_return"),b_ret))
        cost_regression=avg_cost>b_cost*1.02
        fail_cost_safety=int(cost_regression and viol>0.30)
        fail_cr=int(cr<max(0.86,b_cr-0.01));fail_pl=int(pl>max(0.12,b_pl*1.15))
        fail_vout=int(vout>min(0.02,b_vout*1.10+1e-8));fail_return=int(ret_drop>max_return_drop_ratio)
    n_fail=fail_cost_safety+fail_cr+fail_pl+fail_vout+fail_destroy+fail_return
    return {"pass":int(n_fail==0),"n_fail":int(n_fail),"fail_cost_safety":fail_cost_safety,"fail_cr":fail_cr,"fail_pl":fail_pl,"fail_vout":fail_vout,"fail_destroy":fail_destroy,"fail_return":fail_return}

def candidate_sort_key(summary,eval_score,baseline_summary=None,max_return_drop_ratio=0.15)->Tuple:
    g=build_gate_info(summary,baseline_summary,max_return_drop_ratio)
    return (g["n_fail"],float(eval_score),sf(summary.get("avg_cost"),1e9),sf(summary.get("avg_v_out"),1e9),sf(summary.get("cvar95_cost"),1e9),
            sf(summary.get("mean_cost_violation_ratio")),-sf(summary.get("cr")),-sf(summary.get("avg_return"),-1e9))

def eval_one_model_multi_windows(model_path,dataset,action_low,action_high,stats_path,scenario,mode,voltage_barrier_type,episode_limit,seed,eval_episodes,
                                 manual_reset,start_days,day_step,hour,quarter,device,actor_model,critic_model,actor_hidden_dims,critic_hidden_dims,critic_attend_heads,cost_limit):
    window_results=[];merged=[]
    for sd in start_days:
        summary,records=evaluate_mabcq_lag_v2(model_path,dataset.obs_dim,dataset.state_dim,dataset.act_dim,dataset.n_agents,action_low,action_high,
                                             scenario,mode,voltage_barrier_type,episode_limit,seed,eval_episodes,manual_reset,sd,day_step,hour,quarter,cost_limit,device,stats_path,
                                             actor_model,critic_model,tuple(actor_hidden_dims),tuple(critic_hidden_dims),critic_attend_heads,True,20)
        window_results.append({"start_day":sd,"summary":summary,"records":records});merged.extend(records)
    return aggregate_window_summaries(window_results),merged,window_results

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-roots",nargs="+",required=True);p.add_argument("--ckpt-dir",type=str,required=True);p.add_argument("--out-json",type=str,required=True)
    p.add_argument("--min-epoch",type=int,default=1);p.add_argument("--max-epoch",type=int,default=80);p.add_argument("--stride",type=int,default=1);p.add_argument("--topk",type=int,default=5)
    p.add_argument("--scenario",type=str,default="case33_3min_final");p.add_argument("--mode",type=str,default="distributed");p.add_argument("--voltage-barrier-type",type=str,default="l1")
    p.add_argument("--episode-limit",type=int,default=480);p.add_argument("--seed",type=int,default=1);p.add_argument("--manual-reset",action="store_true");p.add_argument("--day-step",type=int,default=1)
    p.add_argument("--hour",type=int,default=23);p.add_argument("--quarter",type=int,default=2);p.add_argument("--device",type=str,default="cpu")
    p.add_argument("--val-start-days",type=int,nargs="+",default=[730]);p.add_argument("--val-eval-episodes",type=int,default=20)
    p.add_argument("--test-start-days",type=int,nargs="+",default=[760]);p.add_argument("--test-eval-episodes",type=int,default=20)
    p.add_argument("--actor-model",type=str,default="mlp");p.add_argument("--critic-model",type=str,default="maac");p.add_argument("--actor-hidden-dims",type=int,nargs="+",default=[256,256])
    p.add_argument("--critic-hidden-dims",type=int,nargs="+",default=[512,512]);p.add_argument("--critic-attend-heads",type=int,default=4)
    p.add_argument("--cost-limit-ratio",type=float,default=0.8);p.add_argument("--cost-limit-quantile",type=float,default=0.5);p.add_argument("--state-budget-quantile",type=float,default=0.5)
    p.add_argument("--baseline-json",type=str,default="");p.add_argument("--soft-return-drop-ratio",type=float,default=0.10);p.add_argument("--max-return-drop-ratio",type=float,default=0.15);args=p.parse_args()

    dataset,action_low,action_high=load_dataset_and_action_space(args.data_roots,0.99,args.state_budget_quantile)
    stats_path=os.path.join(args.ckpt_dir,"dataset_stats.json");cost_limit=estimate_cost_limit(dataset,0.99,args.cost_limit_ratio,args.cost_limit_quantile);baseline_summary=load_baseline_summary(args.baseline_json)
    epochs=discover_epochs(args.ckpt_dir,args.min_epoch,args.max_epoch,args.stride);val_results=[]
    for ep in epochs:
        model_path=os.path.join(args.ckpt_dir,f"model_epoch_{ep}.pt")
        if not os.path.exists(model_path): continue
        s,records,windows=eval_one_model_multi_windows(model_path,dataset,action_low,action_high,stats_path,args.scenario,args.mode,args.voltage_barrier_type,args.episode_limit,args.seed,args.val_eval_episodes,args.manual_reset,args.val_start_days,args.day_step,args.hour,args.quarter,args.device,args.actor_model,args.critic_model,args.actor_hidden_dims,args.critic_hidden_dims,args.critic_attend_heads,cost_limit)
        score=make_eval_score(s,baseline_summary,args.soft_return_drop_ratio,args.max_return_drop_ratio);val_results.append({"epoch":ep,"model_path":model_path,"summary":s,"records":records,"window_results":windows,"eval_score":score,"gate":build_gate_info(s,baseline_summary,args.max_return_drop_ratio)})
    val_results=sorted(val_results,key=lambda x:candidate_sort_key(x["summary"],x["eval_score"],baseline_summary,args.max_return_drop_ratio));topk=val_results[:args.topk];tested=[]
    for item in topk:
        ts,tr,tw=eval_one_model_multi_windows(item["model_path"],dataset,action_low,action_high,stats_path,args.scenario,args.mode,args.voltage_barrier_type,args.episode_limit,args.seed,args.test_eval_episodes,args.manual_reset,args.test_start_days,args.day_step,args.hour,args.quarter,args.device,args.actor_model,args.critic_model,args.actor_hidden_dims,args.critic_hidden_dims,args.critic_attend_heads,cost_limit)
        tscore=make_eval_score(ts,baseline_summary,args.soft_return_drop_ratio,args.max_return_drop_ratio)
        tested.append({"epoch":item["epoch"],"model_path":item["model_path"],"validation_summary":item["summary"],"validation_eval_score":item["eval_score"],"validation_gate":item["gate"],"test_summary":ts,"test_records":tr,"test_window_results":tw,"test_eval_score":tscore,"test_gate":build_gate_info(ts,baseline_summary,args.max_return_drop_ratio)})
    tested=sorted(tested,key=lambda x:candidate_sort_key(x["test_summary"],x["test_eval_score"],baseline_summary,args.max_return_drop_ratio));best=tested[0] if tested else None
    payload={"validation_eval_episodes_per_window":args.val_eval_episodes,
             "test_eval_episodes_per_window":args.test_eval_episodes,
             "val_start_days":args.val_start_days,
             "test_start_days":args.test_start_days,
             "cost_limit":cost_limit,
             "baseline_summary_used_for_selection":baseline_summary,
             "validation_results":val_results,
             "topk_validation_candidates":topk,
             "tested_topk_candidates":tested,
             "selected_best_on_validation":topk[0] if topk else None,
             "final_test_result":None if best is None else {"epoch":best["epoch"],"model_path":best["model_path"],"summary":best["test_summary"],"records":best["test_records"],"window_results":best["test_window_results"],"eval_score":best["test_eval_score"],"gate":best["test_gate"]}}
    os.makedirs(os.path.dirname(args.out_json),exist_ok=True) if os.path.dirname(args.out_json) else None
    with open(args.out_json,"w",encoding="utf-8") as f: json.dump(payload,f,indent=2,ensure_ascii=False)
    print(f"Saved split review to: {args.out_json}")

if __name__=="__main__": main()
