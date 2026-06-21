import argparse,json,os
from typing import Dict,List,Tuple
import numpy as np
from offline_safe.dataset.common import build_env
from offline_safe.algos.ma_bcq_retain_lag_v2 import MABCQRetainLag
from offline_safe.eval.safety_metrics import EpisodeMetricTracker,summarize_records

def _normalize_obs(obs,np_mean,np_std): return (obs-np_mean.squeeze(0))/(np_std.squeeze(0)+1e-8)
def _normalize_state(state,np_mean,np_std): return (state-np_mean.squeeze(0))/(np_std.squeeze(0)+1e-8)
def _read_ckpt_meta(model_path:str,device:str="cpu")->Dict:
    import torch;ckpt=torch.load(model_path,map_location=device);return ckpt.get("meta",{})

def evaluate_mabcq_lag_v2(model_path:str,obs_dim:int,state_dim:int,act_dim:int,n_agents:int,action_low:float,action_high:float,
                          scenario:str,mode:str,voltage_barrier_type:str,episode_limit:int,seed:int,eval_episodes:int,
                          manual_reset:bool,start_day:int,day_step:int,hour:int,quarter:int,cost_limit:float=5.0,device:str="cpu",
                          dataset_stats_path:str="",actor_model_name:str="mlp",critic_model_name:str="mlp",actor_hidden_dims=(256,256),
                          critic_hidden_dims=(512,512),critic_attend_heads:int=4,deterministic:bool=True,num_candidates:int=20)->Tuple[Dict,List[Dict]]:
    env,_=build_env(env_name="var_voltage_control",scenario=scenario,mode=mode,voltage_barrier_type=voltage_barrier_type,episode_limit=episode_limit,seed=seed)
    obs_mean=obs_std=state_mean=state_std=None
    if dataset_stats_path and os.path.exists(dataset_stats_path):
        with open(dataset_stats_path,"r",encoding="utf-8") as f: stats=json.load(f)
        obs_mean=np.asarray(stats["obs_mean"],dtype=np.float32);obs_std=np.asarray(stats["obs_std"],dtype=np.float32)
        state_mean=np.asarray(stats["state_mean"],dtype=np.float32);state_std=np.asarray(stats["state_std"],dtype=np.float32)
    meta=_read_ckpt_meta(model_path,device=device)
    algo=MABCQRetainLag(obs_dim=obs_dim,state_dim=state_dim,act_dim=act_dim,n_agents=n_agents,action_low=action_low,action_high=action_high,
                        cost_limit=float(meta.get("cost_limit",cost_limit)),device=device,actor_model_name=meta.get("actor_model_name",actor_model_name),
                        critic_model_name=meta.get("critic_model_name",critic_model_name),actor_hidden_dims=tuple(meta.get("actor_hidden_dims",list(actor_hidden_dims))),
                        critic_hidden_dims=tuple(meta.get("critic_hidden_dims",list(critic_hidden_dims))),critic_attend_heads=int(meta.get("critic_attend_heads",critic_attend_heads)),
                        warmup_actor_steps=int(meta.get("warmup_actor_steps",1000)),lambda_warmup_steps=int(meta.get("lambda_warmup_steps",1000)),
                        safety_margin_coef=float(meta.get("safety_margin_coef",0.25)),cost_weight_coef=float(meta.get("cost_weight_coef",1.5)),
                        budget_mix_ratio=float(meta.get("budget_mix_ratio",0.7)))
    algo.load(model_path);records=[]
    for ep in range(eval_episodes):
        if manual_reset: obs,state=env.manual_reset(start_day+ep*day_step,hour,quarter)
        else: obs,state=env.reset()
        raw_obs=np.asarray(obs,dtype=np.float32);raw_state=np.asarray(state,dtype=np.float32)
        obs=_normalize_obs(raw_obs,obs_mean,obs_std) if obs_mean is not None else raw_obs
        state=_normalize_state(raw_state,state_mean,state_std) if state_mean is not None else raw_state
        done=False;tracker=EpisodeMetricTracker(gamma=0.99,cost_limit=cost_limit)
        while not done:
            action=algo.select_action(obs,state=state,deterministic=deterministic,num_candidates=num_candidates).reshape(-1)
            reward,done,info=env.step(action,add_noise=False)
            raw_next_obs=np.asarray(env.get_obs(),dtype=np.float32);raw_next_state=np.asarray(env.get_state(),dtype=np.float32)
            next_obs=_normalize_obs(raw_next_obs,obs_mean,obs_std) if obs_mean is not None else raw_next_obs
            next_state=_normalize_state(raw_next_state,state_mean,state_std) if state_mean is not None else raw_next_state
            tracker.update(reward=reward,info=info,env=env);obs,state=next_obs,next_state
        records.append(tracker.to_record())
    return summarize_records(records),records

def main():
    p=argparse.ArgumentParser();p.add_argument("--model-path",type=str,required=True);
    p.add_argument("--obs-dim",type=int,required=True)
    p.add_argument("--state-dim",type=int,required=True);
    p.add_argument("--act-dim",type=int,required=True);
    p.add_argument("--n-agents",type=int,required=True)
    p.add_argument("--action-low",type=float,default=-1.0);
    p.add_argument("--action-high",type=float,default=1.0)
    p.add_argument("--scenario",type=str,default="case33_3min_final");
    p.add_argument("--mode",type=str,default="distributed")
    p.add_argument("--voltage-barrier-type",type=str,default="l1");
    p.add_argument("--episode-limit",type=int,default=480);
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--eval-episodes",type=int,default=5);
    p.add_argument("--manual-reset",action="store_true");
    p.add_argument("--start-day",type=int,default=730)
    p.add_argument("--day-step",type=int,default=1);
    p.add_argument("--hour",type=int,default=23);
    p.add_argument("--quarter",type=int,default=2)
    p.add_argument("--cost-limit",type=float,default=5.0);
    p.add_argument("--device",type=str,default="cpu");
    p.add_argument("--save-json",type=str,default="")
    p.add_argument("--dataset-stats-path",type=str,default="");
    p.add_argument("--actor-model",type=str,default="mlp",choices=["mlp","residual","gated_residual"])
    p.add_argument("--critic-model",type=str,default="mlp",choices=["mlp","central","maac"]);
    p.add_argument("--actor-hidden-dims",type=int,nargs="+",default=[256,256])
    p.add_argument("--critic-hidden-dims",type=int,nargs="+",default=[512,512]);
    p.add_argument("--critic-attend-heads",type=int,default=4)
    p.add_argument("--deterministic",action="store_true");
    p.add_argument("--num-candidates",type=int,default=20);

    args=p.parse_args()
    summary,records=evaluate_mabcq_lag_v2(args.model_path,args.obs_dim,args.state_dim,args.act_dim,args.n_agents,args.action_low,args.action_high,
                                          args.scenario,args.mode,args.voltage_barrier_type,args.episode_limit,args.seed,args.eval_episodes,args.manual_reset,
                                          args.start_day,args.day_step,args.hour,args.quarter,args.cost_limit,args.device,args.dataset_stats_path,args.actor_model,
                                          args.critic_model,tuple(args.actor_hidden_dims),tuple(args.critic_hidden_dims),args.critic_attend_heads,args.deterministic,args.num_candidates)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json),exist_ok=True) if os.path.dirname(args.save_json) else None
        with open(args.save_json,"w",encoding="utf-8") as f: json.dump({"summary":summary,"records":records},f,indent=2,ensure_ascii=False)

if __name__=="__main__": main()
