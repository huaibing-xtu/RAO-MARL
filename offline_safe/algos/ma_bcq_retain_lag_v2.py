import copy
from typing import Dict,Optional
import torch, torch.nn as nn, torch.nn.functional as F
from offline_safe.models.registry import build_behavior_model,build_actor_model,build_critic_model

class MABCQRetainLag:
    def __init__(self,obs_dim:int,state_dim:int,act_dim:int,n_agents:int,action_low:float,action_high:float,
                 device:str="cpu",gamma:float=0.99,tau:float=0.005,phi:float=0.05,latent_dim:int=16,
                 actor_lr:float=3e-4,critic_lr:float=3e-4,vae_lr:float=3e-4,lag_lr:float=1e-4,cost_limit:float=5.0,
                 bc_coef:float=0.2,bc_coef_end:float=0.05,action_l2_coef:float=1e-4,cql_alpha_reward:float=0.05,
                 cql_alpha_cost:float=0.05,warmup_actor_steps:int=1000,lambda_warmup_steps:Optional[int]=None,
                 actor_model_name:str="mlp",critic_model_name:str="mlp",actor_hidden_dims=(256,256),critic_hidden_dims=(512,512),
                 critic_attend_heads:int=4,actor_update_interval:int=2,critic_grad_clip:float=10.0,actor_grad_clip:float=10.0,
                 safety_margin_coef:float=0.25,cost_weight_coef:float=1.50,budget_mix_ratio:float=0.70):
        self.device=torch.device(device);self.gamma=float(gamma);self.tau=float(tau);self.n_agents=int(n_agents);self.act_dim=int(act_dim)
        self.cost_limit=float(cost_limit);self.bc_coef=float(bc_coef);self.bc_coef_end=float(bc_coef_end);self.action_l2_coef=float(action_l2_coef)
        self.cql_alpha_reward=float(cql_alpha_reward);self.cql_alpha_cost=float(cql_alpha_cost);self.warmup_actor_steps=int(warmup_actor_steps)
        self.lambda_warmup_steps=int(lambda_warmup_steps) if lambda_warmup_steps is not None else int(warmup_actor_steps)
        self.total_updates=0;self.actor_update_interval=int(actor_update_interval);self.critic_grad_clip=float(critic_grad_clip);self.actor_grad_clip=float(actor_grad_clip)
        self.safety_margin_coef=float(safety_margin_coef);self.cost_weight_coef=float(cost_weight_coef);self.budget_mix_ratio=float(budget_mix_ratio)
        self.action_low=float(action_low);self.action_high=float(action_high);self.actor_model_name=actor_model_name;self.critic_model_name=critic_model_name
        self.actor_hidden_dims=tuple(actor_hidden_dims);self.critic_hidden_dims=tuple(critic_hidden_dims);self.critic_attend_heads=int(critic_attend_heads)
        self.behavior=build_behavior_model(obs_dim,act_dim,latent_dim=latent_dim,hidden_dims=self.actor_hidden_dims).to(self.device)
        self.actor=build_actor_model(actor_model_name,obs_dim,act_dim,action_low,action_high,phi=phi,hidden_dims=self.actor_hidden_dims).to(self.device)
        self.qr1=build_critic_model(critic_model_name,state_dim,n_agents,act_dim,hidden_dims=self.critic_hidden_dims,attend_heads=critic_attend_heads).to(self.device)
        self.qr2=build_critic_model(critic_model_name,state_dim,n_agents,act_dim,hidden_dims=self.critic_hidden_dims,attend_heads=critic_attend_heads).to(self.device)
        self.qc1=build_critic_model(critic_model_name,state_dim,n_agents,act_dim,hidden_dims=self.critic_hidden_dims,attend_heads=critic_attend_heads).to(self.device)
        self.qc2=build_critic_model(critic_model_name,state_dim,n_agents,act_dim,hidden_dims=self.critic_hidden_dims,attend_heads=critic_attend_heads).to(self.device)
        self.behavior_targ=copy.deepcopy(self.behavior).eval();self.actor_targ=copy.deepcopy(self.actor).eval()
        self.qr1_targ=copy.deepcopy(self.qr1).eval();self.qr2_targ=copy.deepcopy(self.qr2).eval();self.qc1_targ=copy.deepcopy(self.qc1).eval();self.qc2_targ=copy.deepcopy(self.qc2).eval()
        self.behavior_opt=torch.optim.Adam(self.behavior.parameters(),lr=vae_lr);self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.qr_opt=torch.optim.Adam(list(self.qr1.parameters())+list(self.qr2.parameters()),lr=critic_lr)
        self.qc_opt=torch.optim.Adam(list(self.qc1.parameters())+list(self.qc2.parameters()),lr=critic_lr)
        self.log_lam=nn.Parameter(torch.tensor(-2.0,device=self.device));self.lam_opt=torch.optim.Adam([self.log_lam],lr=lag_lr)
        self.actor_frozen=False

    @property
    def lam(self)->torch.Tensor: return F.softplus(self.log_lam)
    def set_actor_lr(self,lr:float)->None:
        for g in self.actor_opt.param_groups: g["lr"]=float(lr)
    def get_actor_lr(self)->float: return float(self.actor_opt.param_groups[0]["lr"])
    def freeze_actor(self)->None: self.actor_frozen=True;self.set_actor_lr(0.0)
    def _reshape_agent_batch(self,x:torch.Tensor)->torch.Tensor: b,n,d=x.shape;return x.reshape(b*n,d)
    def _current_bc_coef(self)->float:
        if self.total_updates<=self.warmup_actor_steps: return self.bc_coef
        span=max(1,self.warmup_actor_steps);prog=min(1.0,(self.total_updates-self.warmup_actor_steps)/float(10*span))
        return float(self.bc_coef_end+0.5*(self.bc_coef-self.bc_coef_end)*(1.0+torch.cos(torch.tensor(prog*3.1415926535)).item()))
    def _sample_random_joint_actions(self,batch_size:int)->torch.Tensor:
        u=torch.rand(batch_size,self.n_agents,self.act_dim,device=self.device);return self.action_low+(self.action_high-self.action_low)*u
    def _resolve_budget(self,state_budget:Optional[torch.Tensor],ref:torch.Tensor)->torch.Tensor:
        if state_budget is None: return torch.full_like(ref,self.cost_limit)
        if state_budget.dim()==1: state_budget=state_budget.unsqueeze(-1)
        if state_budget.shape!=ref.shape: state_budget=state_budget.expand_as(ref)
        return self.budget_mix_ratio*state_budget+(1.0-self.budget_mix_ratio)*self.cost_limit
    def _twin_reward_lb(self,state,actions,use_target=False):
        qr1=(self.qr1_targ if use_target else self.qr1)(state,actions);qr2=(self.qr2_targ if use_target else self.qr2)(state,actions);return torch.min(qr1,qr2)
    def _twin_cost_ub(self,state,actions,use_target=False):
        qc1=(self.qc1_targ if use_target else self.qc1)(state,actions);qc2=(self.qc2_targ if use_target else self.qc2)(state,actions);return torch.max(qc1,qc2)
    def _score_actions(self,state,actions,state_budget:Optional[torch.Tensor]=None,use_target:bool=False):
        qr_lb=self._twin_reward_lb(state,actions,use_target=use_target);qc_ub=self._twin_cost_ub(state,actions,use_target=use_target)
        budget=self._resolve_budget(state_budget,qc_ub);violation=F.relu(qc_ub-budget)
        return qr_lb-self.lam.detach()*qc_ub-self.safety_margin_coef*violation
    def _joint_action_from_local(self,obs:torch.Tensor,state:Optional[torch.Tensor]=None,use_target:bool=False,deterministic:bool=False,
                                 num_candidates:int=1,state_budget:Optional[torch.Tensor]=None)->torch.Tensor:
        b,n,_=obs.shape;flat_obs=self._reshape_agent_batch(obs);behavior_net=self.behavior_targ if use_target else self.behavior;actor_net=self.actor_targ if use_target else self.actor
        if num_candidates<=1 or state is None:
            if use_target:
                with torch.no_grad(): bc=behavior_net.decode(flat_obs,deterministic=deterministic);act=actor_net(flat_obs,bc)
            else:
                bc=behavior_net.decode(flat_obs,deterministic=deterministic);act=actor_net(flat_obs,bc)
            return act.reshape(b,n,self.act_dim)
        cand_actions,cand_scores=[],[]
        for _ in range(num_candidates):
            with torch.no_grad():
                bc=behavior_net.decode(flat_obs,deterministic=False);act=actor_net(flat_obs,bc).reshape(b,n,self.act_dim)
                score=self._score_actions(state,act,state_budget=state_budget,use_target=use_target)
            cand_actions.append(act);cand_scores.append(score)
        stack=torch.stack(cand_scores,dim=0).squeeze(-1);best_idx=torch.argmax(stack,dim=0)
        return torch.stack([cand_actions[int(best_idx[i].item())][i] for i in range(b)],dim=0)
    def _cql_penalty(self,critic:nn.Module,state:torch.Tensor,obs:torch.Tensor,data_actions:torch.Tensor,num_samples:int=5)->torch.Tensor:
        b=obs.shape[0];random_qs=[];policy_qs=[]
        for _ in range(num_samples):
            rand_a=self._sample_random_joint_actions(b);random_qs.append(critic(state,rand_a))
            with torch.no_grad(): pi_a=self._joint_action_from_local(obs,state=state,use_target=False,deterministic=False,num_candidates=1)
            policy_qs.append(critic(state,pi_a))
        random_q=torch.cat(random_qs,dim=1);policy_q=torch.cat(policy_qs,dim=1);data_q=critic(state,data_actions);all_q=torch.cat([random_q,policy_q],dim=1)
        return torch.logsumexp(all_q,dim=1,keepdim=True).mean()-data_q.mean()
    @torch.no_grad()
    def select_action(self,obs,state=None,deterministic:bool=True,num_candidates:int=20):
        obs_t=torch.as_tensor(obs,dtype=torch.float32,device=self.device).unsqueeze(0)
        state_t=None if state is None else torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
        act=self._joint_action_from_local(obs_t,state_t,use_target=False,deterministic=deterministic,num_candidates=num_candidates,state_budget=None)[0]
        return act.cpu().numpy()
    def update(self,batch:Dict[str,torch.Tensor])->Dict[str,float]:
        self.total_updates+=1
        obs,state,actions=batch["obs"].to(self.device),batch["state"].to(self.device),batch["actions"].to(self.device)
        rewards,costs=batch["rewards"].to(self.device),batch["costs"].to(self.device)
        next_obs,next_state,dones=batch["next_obs"].to(self.device),batch["next_state"].to(self.device),batch["dones"].to(self.device)
        state_budget=batch.get("state_budget");next_state_budget=batch.get("next_state_budget");unsafe_weight=batch.get("unsafe_weight")
        state_budget=None if state_budget is None else state_budget.to(self.device);next_state_budget=None if next_state_budget is None else next_state_budget.to(self.device)
        unsafe_weight=torch.ones_like(costs,device=self.device) if unsafe_weight is None else unsafe_weight.to(self.device)
        b,n,_=obs.shape;flat_obs=self._reshape_agent_batch(obs);flat_act=self._reshape_agent_batch(actions)

        recon,mu,std=self.behavior(flat_obs,flat_act);recon_loss=F.mse_loss(recon,flat_act);kl_loss=-0.5*(1+torch.log(std.pow(2)+1e-8)-mu.pow(2)-std.pow(2)).mean();vae_loss=recon_loss+0.5*kl_loss
        self.behavior_opt.zero_grad();vae_loss.backward();torch.nn.utils.clip_grad_norm_(self.behavior.parameters(),max_norm=self.critic_grad_clip);self.behavior_opt.step()

        with torch.no_grad():
            next_joint_action=self._joint_action_from_local(next_obs,state=next_state,use_target=True,deterministic=False,num_candidates=10,state_budget=next_state_budget)
            qr_targ=self._twin_reward_lb(next_state,next_joint_action,use_target=True);qc_targ=self._twin_cost_ub(next_state,next_joint_action,use_target=True)
            target_reward_q=rewards+self.gamma*(1.0-dones)*qr_targ;target_cost_q=costs+self.gamma*(1.0-dones)*qc_targ

        qr1_pred,qr2_pred=self.qr1(state,actions),self.qr2(state,actions);qr_td_loss=F.mse_loss(qr1_pred,target_reward_q)+F.mse_loss(qr2_pred,target_reward_q)
        qr_cql_loss=self._cql_penalty(self.qr1,state,obs,actions)+self._cql_penalty(self.qr2,state,obs,actions);qr_loss=qr_td_loss+self.cql_alpha_reward*qr_cql_loss
        self.qr_opt.zero_grad();qr_loss.backward();torch.nn.utils.clip_grad_norm_(list(self.qr1.parameters())+list(self.qr2.parameters()),max_norm=self.critic_grad_clip);self.qr_opt.step()

        qc1_pred,qc2_pred=self.qc1(state,actions),self.qc2(state,actions);td1=(qc1_pred-target_cost_q).pow(2);td2=(qc2_pred-target_cost_q).pow(2)
        cost_weight=torch.clamp(unsafe_weight,min=1.0,max=1.0+3.0*self.cost_weight_coef)
        qc_td_loss=(cost_weight*td1).mean()+(cost_weight*td2).mean()
        qc_cql_loss=self._cql_penalty(self.qc1,state,obs,actions)+self._cql_penalty(self.qc2,state,obs,actions);qc_loss=qc_td_loss+self.cql_alpha_cost*qc_cql_loss
        self.qc_opt.zero_grad();qc_loss.backward();torch.nn.utils.clip_grad_norm_(list(self.qc1.parameters())+list(self.qc2.parameters()),max_norm=self.critic_grad_clip);self.qc_opt.step()

        joint_pi=self._joint_action_from_local(obs,state=state,use_target=False,deterministic=False,num_candidates=1,state_budget=state_budget)
        qr_pi=torch.min(self.qr1(state,joint_pi),self.qr2(state,joint_pi));qc_pi=torch.max(self.qc1(state,joint_pi),self.qc2(state,joint_pi))
        resolved_budget=self._resolve_budget(state_budget,qc_pi);safety_violation=F.relu(qc_pi-resolved_budget)
        with torch.no_grad(): flat_bc=self.behavior.decode(flat_obs,deterministic=True)
        flat_pi=joint_pi.reshape(b*n,-1);bc_reg=F.mse_loss(flat_pi,flat_bc);act_l2=(flat_pi**2).mean();bc_coef_now=self._current_bc_coef()
        if self.total_updates<=self.warmup_actor_steps: actor_loss=bc_coef_now*bc_reg+self.action_l2_coef*act_l2
        else:
            actor_obj=qr_pi-self.lam.detach()*qc_pi-self.safety_margin_coef*safety_violation
            actor_loss=-actor_obj.mean()+bc_coef_now*bc_reg+self.action_l2_coef*act_l2
        did_actor_update=False
        if (self.total_updates%self.actor_update_interval)==0 and (not self.actor_frozen) and self.get_actor_lr()>0.0:
            self.actor_opt.zero_grad();actor_loss.backward();torch.nn.utils.clip_grad_norm_(self.actor.parameters(),max_norm=self.actor_grad_clip);self.actor_opt.step();did_actor_update=True
        if self.total_updates>self.lambda_warmup_steps:
            lam_loss=-(self.lam*(qc_pi.detach()-resolved_budget.detach())).mean();self.lam_opt.zero_grad();lam_loss.backward();self.lam_opt.step();lam_loss_value=float(lam_loss.item())
        else: lam_loss_value=0.0
        self._soft_update(self.behavior,self.behavior_targ,self.tau);self._soft_update(self.actor,self.actor_targ,self.tau)
        self._soft_update(self.qr1,self.qr1_targ,self.tau);self._soft_update(self.qr2,self.qr2_targ,self.tau);self._soft_update(self.qc1,self.qc1_targ,self.tau);self._soft_update(self.qc2,self.qc2_targ,self.tau)
        return {"vae_loss":float(vae_loss.item()),"qr_loss":float(qr_loss.item()),"qc_loss":float(qc_loss.item()),"actor_loss":float(actor_loss.item()),
                "lambda":float(self.lam.item()),"lam_loss":lam_loss_value,"qr_pi":float(qr_pi.mean().item()),"qc_pi":float(qc_pi.mean().item()),
                "budget_mean":float(resolved_budget.mean().item()),"safety_violation":float(safety_violation.mean().item()),"bc_reg":float(bc_reg.item()),
                "act_l2":float(act_l2.item()),"bc_coef_now":float(bc_coef_now),"qr_cql":float(qr_cql_loss.item()),"qc_cql":float(qc_cql_loss.item()),
                "did_actor_update":float(did_actor_update),"actor_lr":float(self.get_actor_lr())}
    @staticmethod
    def _soft_update(src,tgt,tau:float=0.005):
        for p,tp in zip(src.parameters(),tgt.parameters()): tp.data.copy_(tau*p.data+(1.0-tau)*tp.data)
    def save(self,path:str)->None:
        ckpt={"behavior":self.behavior.state_dict(),"actor":self.actor.state_dict(),"qr1":self.qr1.state_dict(),"qr2":self.qr2.state_dict(),"qc1":self.qc1.state_dict(),"qc2":self.qc2.state_dict(),
              "behavior_targ":self.behavior_targ.state_dict(),"actor_targ":self.actor_targ.state_dict(),"qr1_targ":self.qr1_targ.state_dict(),"qr2_targ":self.qr2_targ.state_dict(),
              "qc1_targ":self.qc1_targ.state_dict(),"qc2_targ":self.qc2_targ.state_dict(),"behavior_opt":self.behavior_opt.state_dict(),"actor_opt":self.actor_opt.state_dict(),
              "qr_opt":self.qr_opt.state_dict(),"qc_opt":self.qc_opt.state_dict(),"lam_opt":self.lam_opt.state_dict(),"log_lam":self.log_lam.data,"total_updates":self.total_updates,
              "actor_frozen":self.actor_frozen,"meta":{"n_agents":self.n_agents,"act_dim":self.act_dim,"cost_limit":self.cost_limit,"actor_model_name":self.actor_model_name,
              "critic_model_name":self.critic_model_name,"actor_hidden_dims":list(self.actor_hidden_dims),"critic_hidden_dims":list(self.critic_hidden_dims),
              "critic_attend_heads":self.critic_attend_heads,"warmup_actor_steps":self.warmup_actor_steps,"lambda_warmup_steps":self.lambda_warmup_steps,
              "safety_margin_coef":self.safety_margin_coef,"cost_weight_coef":self.cost_weight_coef,"budget_mix_ratio":self.budget_mix_ratio}};torch.save(ckpt,path)
    def load(self,path:str)->None:
        ckpt=torch.load(path,map_location=self.device)
        for k,m in [("behavior",self.behavior),("actor",self.actor),("qr1",self.qr1),("qr2",self.qr2),("qc1",self.qc1),("qc2",self.qc2)]: m.load_state_dict(ckpt[k])
        if "behavior_targ" in ckpt:
            self.behavior_targ.load_state_dict(ckpt["behavior_targ"]);self.actor_targ.load_state_dict(ckpt["actor_targ"])
            self.qr1_targ.load_state_dict(ckpt["qr1_targ"]);self.qr2_targ.load_state_dict(ckpt["qr2_targ"]);self.qc1_targ.load_state_dict(ckpt["qc1_targ"]);self.qc2_targ.load_state_dict(ckpt["qc2_targ"])
        if "behavior_opt" in ckpt: self.behavior_opt.load_state_dict(ckpt["behavior_opt"])
        if "actor_opt" in ckpt: self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if "qr_opt" in ckpt: self.qr_opt.load_state_dict(ckpt["qr_opt"])
        if "qc_opt" in ckpt: self.qc_opt.load_state_dict(ckpt["qc_opt"])
        if "lam_opt" in ckpt: self.lam_opt.load_state_dict(ckpt["lam_opt"])
        self.log_lam.data.copy_(ckpt["log_lam"]);self.total_updates=int(ckpt.get("total_updates",0));self.actor_frozen=bool(ckpt.get("actor_frozen",False))
