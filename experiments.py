"""Reproducible CVaR vs expected-energy ablation; do not cherry-pick seeds."""
import json
from pathlib import Path
from statistics import mean,stdev
from energy.quantum import build_problem,run_qaoa

if __name__=='__main__':
    p=build_problem();rows=[];report=[]
    for alpha in [.25,1.]:
        for depth in [1,2]:
            group=[]
            for seed in [7,23,41]:
                r=run_qaoa(p,depth,alpha=alpha,restarts=3,maxiter=100,shots=8192,seed=seed)
                rows.append(r);group.append(r['p_optimal_exact_distribution'])
            item=dict(alpha=alpha,depth=depth,seeds=[7,23,41],ideal_p_optimal_mean=mean(group),
                      ideal_p_optimal_stdev=stdev(group),individual=group)
            report.append(item);print(item)
    Path('results').mkdir(exist_ok=True)
    Path('results/ablation.json').write_text(json.dumps(dict(summary=report,rows=rows),indent=2))
