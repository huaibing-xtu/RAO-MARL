import os,glob,json
from typing import Dict,List,Sequence
import numpy as np, torch
from torch.utils.data import Dataset

class MAPDNTransitionDataset(Dataset):
    REQUIRED_KEYS=("obs","state","actions","rewards","costs","next_obs","next_state","dones")
    def __init__(self,root_dirs:Sequence[str],normalize_obs:bool=True,normalize_state:bool=True,cache_in_memory:bool=True,
                 gamma:float=0.99,budget_quantile:float=0.50,unsafe_weight_coef:float=1.50):
        if isinstance(root_dirs,str): root_dirs=[root_dirs]
        self.root_dirs=list(root_dirs);self.normalize_obs=normalize_obs;self.normalize_state=normalize_state
        self.cache_in_memory=cache_in_memory;self.gamma=float(gamma);self.budget_quantile=float(budget_quantile)
        self.unsafe_weight_coef=float(unsafe_weight_coef)
        self.file_list=self._collect_files(self.root_dirs)
        if not self.file_list: raise ValueError(f"No ep_*.npz files found in: {self.root_dirs}")
        self.data=self._load_all_files(self.file_list) if cache_in_memory else None
        self._build_index();self._compute_stats();self._compute_budget_stats()

    @staticmethod
    def _collect_files(root_dirs:Sequence[str])->List[str]:
        out=[];[out.extend(sorted(glob.glob(os.path.join(root,"ep_*.npz")))) for root in root_dirs];return out

    def _augment_episode_fields(self,item:Dict[str,np.ndarray])->Dict[str,np.ndarray]:
        rewards=np.asarray(item["rewards"],dtype=np.float32).reshape(-1);costs=np.asarray(item["costs"],dtype=np.float32).reshape(-1);T=len(rewards)
        reward_to_go=np.zeros((T,1),dtype=np.float32);cost_to_go=np.zeros((T,1),dtype=np.float32);rr=0.0;rc=0.0
        for t in reversed(range(T)):
            rr=float(rewards[t])+self.gamma*rr;rc=float(costs[t])+self.gamma*rc
            reward_to_go[t,0]=rr;cost_to_go[t,0]=rc
        item["reward_to_go"]=reward_to_go;item["cost_to_go"]=cost_to_go
        item["timestep"]=np.arange(T,dtype=np.float32).reshape(-1,1);item["remaining_steps"]=(T-1-np.arange(T,dtype=np.float32)).reshape(-1,1)
        return item

    def _load_all_files(self,file_list)->List[Dict[str,np.ndarray]]:
        data=[]
        for fp in file_list:
            with np.load(fp) as d: item={k:d[k].astype(np.float32) for k in d.files}
            self._validate_episode(item,fp);data.append(self._augment_episode_fields(item))
        return data

    def _load_single_file(self,fp:str)->Dict[str,np.ndarray]:
        with np.load(fp) as d: item={k:d[k].astype(np.float32) for k in d.files}
        self._validate_episode(item,fp);return self._augment_episode_fields(item)

    def _validate_episode(self,item,fp)->None:
        for k in self.REQUIRED_KEYS:
            if k not in item: raise KeyError(f"{fp} missing required key: {k}")
        T=item["obs"].shape[0]
        for k in self.REQUIRED_KEYS:
            if item[k].shape[0]!=T: raise ValueError(f"{fp} inconsistent length: {k}={item[k].shape[0]} vs obs={T}")

    def _build_index(self)->None:
        self.index=[];self.episode_lengths=[]
        for epi,fp in enumerate(self.file_list):
            item=self.data[epi] if self.cache_in_memory else self._load_single_file(fp);T=item["obs"].shape[0];self.episode_lengths.append(T)
            for t in range(T): self.index.append((epi,t))

    def _compute_stats(self)->None:
        obs_sum=obs_sq_sum=state_sum=state_sq_sum=None;obs_count=state_count=0
        for epi,fp in enumerate(self.file_list):
            item=self.data[epi] if self.cache_in_memory else self._load_single_file(fp)
            obs,state=item["obs"],item["state"]
            if obs_sum is None:
                obs_sum=obs.sum(axis=(0,1),keepdims=True);obs_sq_sum=(obs**2).sum(axis=(0,1),keepdims=True)
                state_sum=state.sum(axis=0,keepdims=True);state_sq_sum=(state**2).sum(axis=0,keepdims=True)
            else:
                obs_sum+=obs.sum(axis=(0,1),keepdims=True);obs_sq_sum+=(obs**2).sum(axis=(0,1),keepdims=True)
                state_sum+=state.sum(axis=0,keepdims=True);state_sq_sum+=(state**2).sum(axis=0,keepdims=True)
            obs_count+=obs.shape[0]*obs.shape[1];state_count+=state.shape[0]
        self.obs_mean=(obs_sum/max(obs_count,1)).astype(np.float32);self.obs_var=(obs_sq_sum/max(obs_count,1)-self.obs_mean**2).astype(np.float32)
        self.obs_std=np.sqrt(np.maximum(self.obs_var,1e-8)).astype(np.float32)
        self.state_mean=(state_sum/max(state_count,1)).astype(np.float32);self.state_var=(state_sq_sum/max(state_count,1)-self.state_mean**2).astype(np.float32)
        self.state_std=np.sqrt(np.maximum(self.state_var,1e-8)).astype(np.float32)
        first=self.data[0] if self.cache_in_memory else self._load_single_file(self.file_list[0])
        self.n_agents=int(first["obs"].shape[1]);self.obs_dim=int(first["obs"].shape[2]);self.state_dim=int(first["state"].shape[1]);self.act_dim=int(first["actions"].shape[2])

    def _compute_budget_stats(self)->None:
        max_len=max(self.episode_lengths) if self.episode_lengths else 0;budgets=[]
        for t in range(max_len):
            vals=[]
            for epi,fp in enumerate(self.file_list):
                item=self.data[epi] if self.cache_in_memory else self._load_single_file(fp)
                if t<item["cost_to_go"].shape[0]: vals.append(float(item["cost_to_go"][t,0]))
            budgets.append(float(np.quantile(vals,self.budget_quantile)) if vals else (budgets[-1] if budgets else 0.0))
        self.timestep_cost_budgets=np.asarray(budgets,dtype=np.float32)

    def save_stats(self,path:str)->None:
        obj={"num_episodes":len(self.file_list),"num_transitions":len(self.index),"n_agents":self.n_agents,"obs_dim":self.obs_dim,
             "state_dim":self.state_dim,"act_dim":self.act_dim,"obs_mean":self.obs_mean.tolist(),"obs_std":self.obs_std.tolist(),
             "state_mean":self.state_mean.tolist(),"state_std":self.state_std.tolist(),"gamma":self.gamma,
             "budget_quantile":self.budget_quantile,"timestep_cost_budgets":self.timestep_cost_budgets.tolist()}
        with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2,ensure_ascii=False)

    def __len__(self)->int: return len(self.index)

    def __getitem__(self,idx:int)->Dict[str,torch.Tensor]:
        epi,t=self.index[idx];item=self.data[epi] if self.cache_in_memory else self._load_single_file(self.file_list[epi])
        obs,state,actions=item["obs"][t],item["state"][t],item["actions"][t];rewards,costs=item["rewards"][t],item["costs"][t]
        next_obs,next_state,dones=item["next_obs"][t],item["next_state"][t],item["dones"][t]
        reward_to_go,cost_to_go=item["reward_to_go"][t],item["cost_to_go"][t];timestep=item["timestep"][t];remaining=item["remaining_steps"][t]
        if self.normalize_obs:
            obs=(obs-self.obs_mean.squeeze(0))/(self.obs_std.squeeze(0)+1e-8);next_obs=(next_obs-self.obs_mean.squeeze(0))/(self.obs_std.squeeze(0)+1e-8)
        if self.normalize_state:
            state=(state-self.state_mean.squeeze(0))/(self.state_std.squeeze(0)+1e-8);next_state=(next_state-self.state_mean.squeeze(0))/(self.state_std.squeeze(0)+1e-8)
        i=min(int(t),max(0,len(self.timestep_cost_budgets)-1));j=min(int(t+1),max(0,len(self.timestep_cost_budgets)-1))
        state_budget=np.asarray([self.timestep_cost_budgets[i]],dtype=np.float32);next_state_budget=np.asarray([self.timestep_cost_budgets[j]],dtype=np.float32)
        ratio=float(cost_to_go.reshape(-1)[0])/max(float(state_budget[0]),1e-6);unsafe_weight=np.asarray([1.0+self.unsafe_weight_coef*max(0.0,ratio-1.0)],dtype=np.float32)
        cast=lambda x: torch.from_numpy(np.asarray(x,dtype=np.float32))
        return {"obs":cast(obs),"state":cast(state),"actions":cast(actions),"rewards":cast(rewards),"costs":cast(costs),
                "next_obs":cast(next_obs),"next_state":cast(next_state),"dones":cast(dones),"reward_to_go":cast(reward_to_go),
                "cost_to_go":cast(cost_to_go),"timestep":cast(timestep),"remaining_steps":cast(remaining),
                "state_budget":cast(state_budget),"next_state_budget":cast(next_state_budget),"unsafe_weight":cast(unsafe_weight)}
