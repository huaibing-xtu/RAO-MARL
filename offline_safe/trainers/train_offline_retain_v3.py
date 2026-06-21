import argparse,json,os,random,sys,time
from datetime import datetime
from typing import Dict,List,Optional,Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from offline_safe.dataset.replay_buffer_v3 import MAPDNTransitionDataset
from offline_safe.eval.evaluate_policy_v2 import evaluate_mabcq_lag_v2
from offline_safe.eval.judgement_relaxed import compare_with_relaxed_judgement
from offline_safe.algos.ma_bcq_retain_lag_v3 import MABCQRetainLagV3

def aggregate_logs(logs:List[Dict[str,float]])->Dict[str,float]:
    if not logs: return {}
    return {k:float(np.mean([x[k] for x in logs if k in x])) for k in logs[0].keys()}

def estimate_cost_limit(dataset,gamma=0.99,ratio=0.8,quantile=0.5)->float:
    vals=[]
    for epi in range(len(dataset.data)):
        costs=dataset.data[epi]["costs"].reshape(-1);disc=sum((gamma**t)*float(c) for t,c in enumerate(costs));vals.append(disc)
    return ratio*float(np.quantile(vals,quantile))

def _convert_agg(agg):
    m={"mean_avg_return":"avg_return","mean_avg_cost":"avg_cost","mean_avg_discounted_cost":"avg_discounted_cost","mean_avg_v_out":"avg_v_out",
       "mean_avg_destroy":"avg_destroy","mean_avg_q_loss":"avg_q_loss","mean_avg_length":"avg_length","mean_cr":"cr","mean_pl":"pl",
       "mean_v_dev":"v_dev","mean_max_v_drop_dev":"max_v_drop_dev","mean_max_v_rise_dev":"max_v_rise_dev","mean_no_destroy_rate":"no_destroy_rate",
       "mean_worst_episode_cost":"worst_episode_cost","mean_worst_episode_v_out":"worst_episode_v_out","mean_cvar95_cost":"cvar95_cost",
       "mean_constraint_satisfaction_rate":"constraint_satisfaction_rate"}
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

def sf(x,d=0.0):
    try: return float(d if x is None else x)
    except: return float(d)
def rel_gap_higher(base,val): return max(0.0,(base-val)/max(abs(base),1e-6))
def rel_gap_lower(base,val): return max(0.0,(val-base)/max(abs(base),1e-6))
def rel_improve_lower(base,val): return (base-val)/max(abs(base),1e-6)

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
        fail_cost_safety=int(viol>0.50)  # >50% over cost_limit on average → fail
        fail_cr,fail_pl,fail_vout,fail_return=int(cr<0.85),int(pl>0.12),int(vout>0.02),0
    else:
        b_cost=sf(baseline_summary.get("avg_cost"),avg_cost);b_cr=sf(baseline_summary.get("cr"),cr)
        b_pl=sf(baseline_summary.get("pl"),pl);b_vout=sf(baseline_summary.get("avg_v_out"),vout)
        b_ret=sf(baseline_summary.get("avg_return"),sf(summary.get("avg_return")))
        ret_drop=rel_gap_higher(b_ret,sf(summary.get("avg_return"),b_ret))
        # fail only if cost regresses (worse than baseline) AND violation is substantial
        cost_regression=avg_cost>b_cost*1.02  # cost increased >2% vs baseline
        fail_cost_safety=int(cost_regression and viol>0.30)  # both cost regression + high violation
        fail_cr=int(cr<max(0.86,b_cr-0.01));fail_pl=int(pl>max(0.12,b_pl*1.15))
        fail_vout=int(vout>min(0.02,b_vout*1.10+1e-8));fail_return=int(ret_drop>max_return_drop_ratio)
    n_fail=fail_cost_safety+fail_cr+fail_pl+fail_vout+fail_destroy+fail_return
    return {"pass":int(n_fail==0),"n_fail":int(n_fail),"fail_cost_safety":fail_cost_safety,"fail_cr":fail_cr,"fail_pl":fail_pl,"fail_vout":fail_vout,"fail_destroy":fail_destroy,"fail_return":fail_return}

def candidate_sort_key(item:Dict)->Tuple:
    s=item["summary"];g=item["gate"]
    return (g["n_fail"],float(item["eval_score"]),sf(s.get("avg_cost"),1e9),sf(s.get("avg_v_out"),1e9),sf(s.get("cvar95_cost"),1e9),sf(s.get("mean_cost_violation_ratio")),-sf(s.get("cr")),-sf(s.get("avg_return"),-1e9))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-roots",nargs="+",required=True)
    p.add_argument("--save-dir",type=str,required=True)
    p.add_argument("--baseline-json",type=str,default="")
    p.add_argument("--cost-limit",type=float,default=None)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--epochs",type=int,default=120)
    p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--device",type=str,default="cpu")
    p.add_argument("--gamma",type=float,default=0.99)
    p.add_argument("--tau",type=float,default=0.005)
    p.add_argument("--phi",type=float,default=0.05)
    p.add_argument("--latent-dim",type=int,default=16)
    p.add_argument("--actor-lr",type=float,default=3e-4)
    p.add_argument("--critic-lr",type=float,default=3e-4)
    p.add_argument("--vae-lr",type=float,default=3e-4)
    p.add_argument("--lag-lr",type=float,default=1e-4)
    p.add_argument("--cost-limit-ratio",type=float,default=0.8)
    p.add_argument("--cost-limit-quantile",type=float,default=0.5)
    p.add_argument("--state-budget-quantile",type=float,default=0.5)
    p.add_argument("--unsafe-weight-coef",type=float,default=1.5)
    p.add_argument("--action-low",type=float,default=None)
    p.add_argument("--action-high",type=float,default=None)
    p.add_argument("--bc-coef",type=float,default=0.25)
    p.add_argument("--bc-coef-end",type=float,default=0.05)
    p.add_argument("--action-l2-coef",type=float,default=1e-4)
    p.add_argument("--cql-alpha-reward",type=float,default=0.05)
    p.add_argument("--cql-alpha-cost",type=float,default=0.05)
    p.add_argument("--warmup-actor-steps",type=int,default=1000)
    p.add_argument("--lambda-warmup-steps",type=int,default=None)
    p.add_argument("--actor-update-interval",type=int,default=2)
    p.add_argument("--safety-margin-coef",type=float,default=0.25)
    p.add_argument("--cost-weight-coef",type=float,default=1.5)
    p.add_argument("--budget-mix-ratio",type=float,default=0.7)
    p.add_argument("--actor-model",type=str,default="mlp")
    p.add_argument("--critic-model",type=str,default="maac")
    p.add_argument("--actor-hidden-dims",type=int,nargs="+",default=[256,256])
    p.add_argument("--critic-hidden-dims",type=int,nargs="+",default=[512,512])
    p.add_argument("--critic-attend-heads",type=int,default=4)
    p.add_argument("--do-eval",action="store_true")
    p.add_argument("--eval-every",type=int,default=1)
    p.add_argument("--eval-episodes",type=int,default=5)
    p.add_argument("--topk-checkpoints",type=int,default=5)
    p.add_argument("--scenario",type=str,default="case33_3min_final")
    p.add_argument("--mode",type=str,default="distributed")
    p.add_argument("--voltage-barrier-type",type=str,default="l1")
    p.add_argument("--episode-limit",type=int,default=480)
    p.add_argument("--seed",type=int,default=1);p.add_argument("--manual-reset",action="store_true")
    p.add_argument("--start-day",type=int,default=730)
    p.add_argument("--day-step",type=int,default=1)
    p.add_argument("--hour",type=int,default=23)
    p.add_argument("--quarter",type=int,default=2)
    p.add_argument("--early-stop-patience",type=int,default=4)
    p.add_argument("--early-stop-min-epochs",type=int,default=5)
    p.add_argument("--stop-on-no-improve",action="store_true")
    p.add_argument("--freeze-actor-after-patience",action="store_true")
    p.add_argument("--actor-lr-patience",type=int,default=1)
    p.add_argument("--actor-lr-decay",type=float,default=0.5)
    p.add_argument("--actor-lr-min",type=float,default=1e-5)
    p.add_argument("--use-actor-lr-plateau",action="store_true")
    p.add_argument("--actor-cost-quantile", type=float, default=0.60,
                   help="Quantile for actor episode cost filtering")
    p.add_argument("--actor-disc-cost-quantile", type=float, default=0.60,
                   help="Quantile for actor discounted cost filtering")
    p.add_argument("--actor-return-quantile", type=float, default=0.30,
                   help="Quantile for actor return filtering")
    p.add_argument("--actor-min-keep-ratio", type=float, default=0.35,
                   help="Minimum ratio of episodes to keep for actor")
    p.add_argument("--actor-good-weight", type=float, default=1.0,
                   help="Weight for good episodes")
    p.add_argument("--actor-medium-weight", type=float, default=0.35,
                   help="Weight for medium episodes")
    p.add_argument("--actor-bad-weight", type=float, default=0.0,
                   help="Weight for bad episodes")
    p.add_argument("--use-actor-filter-for-vae", action="store_true",
                   help="Apply actor filtering to VAE training")
    p.add_argument("--use-actor-filter-for-actor", action="store_true",
                   help="Apply actor filtering to actor training")
    p.add_argument("--soft-return-drop-ratio", type=float, default=0.10,
                   help="Soft threshold for return drop (default: 0.10)")
    p.add_argument("--max-return-drop-ratio", type=float, default=0.15,
                   help="Hard threshold for return drop (default: 0.15)")
    args=p.parse_args();os.makedirs(args.save_dir,exist_ok=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)

    dataset=MAPDNTransitionDataset(args.data_roots,normalize_obs=True,normalize_state=True,cache_in_memory=True,gamma=args.gamma,budget_quantile=args.state_budget_quantile,unsafe_weight_coef=args.unsafe_weight_coef,actor_cost_quantile=args.actor_cost_quantile,actor_disc_cost_quantile=args.actor_disc_cost_quantile,actor_return_quantile=args.actor_return_quantile,actor_min_keep_ratio=args.actor_min_keep_ratio,actor_good_weight=args.actor_good_weight,actor_medium_weight=args.actor_medium_weight,actor_bad_weight=args.actor_bad_weight)
    stats_path=os.path.join(args.save_dir,"dataset_stats.json");dataset.save_stats(stats_path)
    print(f"[dataset] episodes={len(dataset.file_list)} obs_dim={dataset.obs_dim} state_dim={dataset.state_dim} act_dim={dataset.act_dim} n_agents={dataset.n_agents}", flush=True)
    print(f"[dataset] batches_per_epoch={len(dataset)//args.batch_size} (batch_size={args.batch_size})", flush=True)
    t_setup_start = time.time()
    g=torch.Generator();g.manual_seed(args.seed)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,drop_last=True,generator=g)
    action_low=-1.0 if args.action_low is None else float(args.action_low);action_high=1.0 if args.action_high is None else float(args.action_high)
    for root in args.data_roots:
        meta_path=os.path.join(root,"meta.json")
        if os.path.exists(meta_path):
            with open(meta_path,"r",encoding="utf-8") as f: meta=json.load(f)
            try: action_low=float(np.min(meta["action_low"]));action_high=float(np.max(meta["action_high"]));break
            except: pass
    cost_limit=estimate_cost_limit(dataset,args.gamma,args.cost_limit_ratio,args.cost_limit_quantile) if args.cost_limit is None else float(args.cost_limit)
    print(f"[cost_limit] estimated={cost_limit:.2f} ratio={args.cost_limit_ratio} quantile={args.cost_limit_quantile}", flush=True)
    algo=MABCQRetainLagV3(obs_dim=dataset.obs_dim,state_dim=dataset.state_dim,act_dim=dataset.act_dim,n_agents=dataset.n_agents,action_low=action_low,action_high=action_high,
                        device=args.device,gamma=args.gamma,tau=args.tau,phi=args.phi,latent_dim=args.latent_dim,actor_lr=args.actor_lr,critic_lr=args.critic_lr,vae_lr=args.vae_lr,
                        lag_lr=args.lag_lr,cost_limit=cost_limit,bc_coef=args.bc_coef,bc_coef_end=args.bc_coef_end,action_l2_coef=args.action_l2_coef,
                        cql_alpha_reward=args.cql_alpha_reward,cql_alpha_cost=args.cql_alpha_cost,warmup_actor_steps=args.warmup_actor_steps,lambda_warmup_steps=args.lambda_warmup_steps,
                        actor_update_interval=args.actor_update_interval,actor_model_name=args.actor_model,critic_model_name=args.critic_model,actor_hidden_dims=tuple(args.actor_hidden_dims),
                        critic_hidden_dims=tuple(args.critic_hidden_dims),critic_attend_heads=args.critic_attend_heads,safety_margin_coef=args.safety_margin_coef,
                        cost_weight_coef=args.cost_weight_coef,budget_mix_ratio=args.budget_mix_ratio,use_actor_filter_for_vae=args.use_actor_filter_for_vae,use_actor_filter_for_actor=args.use_actor_filter_for_actor)
    print(f"[model] bc_coef={args.bc_coef}→{args.bc_coef_end} cql_r={args.cql_alpha_reward} cql_c={args.cql_alpha_cost} safety_margin={args.safety_margin_coef}", flush=True)
    print(f"[model] actor_filter: cost_q={args.actor_cost_quantile} disc_cost_q={args.actor_disc_cost_quantile} ret_q={args.actor_return_quantile} min_keep={args.actor_min_keep_ratio}", flush=True)
    print(f"[model] init done in {time.time()-t_setup_start:.1f}s, starting training...", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Epoch':>6s} {'VAE':>8s} {'QR':>8s} {'QC':>8s} {'Actor':>8s} {'Lambda':>8s} {'BC':>7s} {'EvalCost':>9s} {'EvalRet':>8s} {'CR':>6s} {'Vout':>7s} {'Best':>5s} {'Time':>7s}", flush=True)
    print(f"{'-'*90}", flush=True)
    baseline_summary=load_baseline_summary(args.baseline_json);train_history=[];best_item=None;topk=[];no_improve=0;lr_no_improve=0
    t_train_start = time.time()
    for epoch in range(1,args.epochs+1):
        t_epoch_start = time.time()
        logs=[];[logs.append(algo.update(batch)) for batch in loader];epoch_log=aggregate_logs(logs);epoch_log["epoch"]=epoch;train_history.append(epoch_log);algo.save(os.path.join(args.save_dir,"model_latest.pt"))
        eval_str = ""
        if args.do_eval and (epoch%args.eval_every==0 or epoch==args.epochs):
            model_path=os.path.join(args.save_dir,f"model_epoch_{epoch:04d}.pt");algo.save(model_path)
            summary,records=evaluate_mabcq_lag_v2(model_path,dataset.obs_dim,dataset.state_dim,dataset.act_dim,dataset.n_agents,action_low,action_high,args.scenario,args.mode,args.voltage_barrier_type,
                                                 args.episode_limit,args.seed,args.eval_episodes,args.manual_reset,args.start_day,args.day_step,args.hour,args.quarter,cost_limit,args.device,
                                                 stats_path,args.actor_model,args.critic_model,tuple(args.actor_hidden_dims),tuple(args.critic_hidden_dims),args.critic_attend_heads,True,20)
            score=make_eval_score(summary,baseline_summary,args.soft_return_drop_ratio,args.max_return_drop_ratio);gate=build_gate_info(summary,baseline_summary,args.max_return_drop_ratio)
            item={"epoch":epoch,"model_path":model_path,"eval_score":float(score),"summary":summary,"gate":gate};topk=sorted(topk+[item],key=candidate_sort_key)[:args.topk_checkpoints]
            with open(os.path.join(args.save_dir,"topk_candidates.json"),"w",encoding="utf-8") as f: json.dump(topk,f,indent=2,ensure_ascii=False)
            improved=best_item is None or candidate_sort_key(item)<candidate_sort_key(best_item)
            if improved:
                best_item=item;no_improve=0;lr_no_improve=0;algo.save(os.path.join(args.save_dir,"model_best.pt"))
                payload={"epoch":epoch,"eval_score":float(score),"summary":summary,"records":records,"gate":gate}
                if baseline_summary is not None: payload["comparison"]=compare_with_relaxed_judgement(baseline_summary,summary)
                with open(os.path.join(args.save_dir,"best_eval.json"),"w",encoding="utf-8") as f: json.dump(payload,f,indent=2,ensure_ascii=False)
            else:
                no_improve+=1;lr_no_improve+=1
                if args.use_actor_lr_plateau and lr_no_improve>=args.actor_lr_patience:
                    cur=algo.get_actor_lr();new=max(cur*args.actor_lr_decay,args.actor_lr_min)
                    if new<cur-1e-12: algo.set_actor_lr(new)
                    lr_no_improve=0
                if args.freeze_actor_after_patience and no_improve>=max(1,args.early_stop_patience//2) and not algo.actor_frozen: algo.freeze_actor()
                if args.stop_on_no_improve and epoch>=args.early_stop_min_epochs and no_improve>=args.early_stop_patience: break
            ec = summary.get("avg_cost", 0); er = summary.get("avg_return", 0); cr = summary.get("cr", 0); vo = summary.get("avg_v_out", 0)
            eval_str = f" {ec:9.2f} {er:8.1f} {cr:6.3f} {vo:7.4f} {'*' if improved else '':5s}"
        dt = time.time() - t_epoch_start
        vae = epoch_log.get("vae_loss", 0); qr = epoch_log.get("qr_loss", 0); qc = epoch_log.get("qc_loss", 0)
        act = epoch_log.get("actor_loss", 0); lam = epoch_log.get("lambda", 0); bc = epoch_log.get("bc_reg", 0)
        print(f"{epoch:6d} {vae:8.4f} {qr:8.4f} {qc:8.4f} {act:8.4f} {lam:8.4f} {bc:7.4f}{eval_str} {dt:7.1f}s", flush=True)
        with open(os.path.join(args.save_dir,"train_history.json"),"w",encoding="utf-8") as f: json.dump(train_history,f,indent=2,ensure_ascii=False)
    total_time = time.time() - t_train_start
    print(f"{'='*90}", flush=True)
    print(f"Training finished. Total time: {total_time/60:.1f}min ({total_time:.0f}s). Save dir: {args.save_dir}", flush=True)

if __name__=="__main__": main()
