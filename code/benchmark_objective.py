"""Side-effect-free centralized-benchmark objective and gradient (extracted from the timing
driver so the validator can import it WITHOUT running any timing campaign or writing files)."""
import numpy as np
from dsceos_validation import aggregate_output, utilization, objective_terms

def obj_and_grad(pflat, arrays, W, Ablk, target, cfg, shape):
    p=pflat.reshape(shape); N=p.shape[0]
    rest=arrays["rest"]; lw=arrays["loss_weight"]; sc=arrays["service_capacity"]
    delta=p-rest; y=aggregate_output(Ablk,p)
    val=objective_terms(p,arrays,W,Ablk,target,cfg)["objective_value"]
    g=np.zeros_like(p)
    g += cfg.aggregate_tracking_weight*(y-target)[None,:]
    g += cfg.loss_weight_scale*lw*delta
    rho=utilization(p,arrays); deg=np.sum(W,axis=1); Lrho=deg*rho - W@rho
    g[:,0] += cfg.sharing_weight*Lrho/sc
    degree=np.sum(W,axis=1)
    # Released formulation: internal term is (lam/2) delta^T L delta with L = D - W,
    # so the exact gradient is lam * L delta -- identical to the controller's local term.
    Ld=degree[:,None]*delta - W@delta
    g += cfg.internal_weight*Ld
    return val, g.reshape(-1)

def reachable_range(Ablk,lower,upper):
    return aggregate_output(Ablk,lower), aggregate_output(Ablk,upper)
