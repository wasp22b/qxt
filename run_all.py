"""Reproduce scenarios and QAOA experiments. All data is synthetic."""
import argparse
import json
from pathlib import Path
from energy.scenarios import make_scenario
from energy.dispatch import solve
from energy.quantum import benchmark, from_dispatch, build_problem
from energy.integration import evaluate_city_candidates

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--fast',action='store_true'); p.add_argument('--out',default='results')
    a=p.parse_args(); out=Path(a.out); out.mkdir(exist_ok=True)
    for stress in ['normal','peak','cloudy','grid_outage','severe']:
        result=solve(make_scenario(stress=stress))
        (out/f'{stress}.json').write_text(json.dumps(result,indent=2))
        print(stress,result['summary'])
        if stress=='normal': normal=result
    for name,problem in [('eight_qubit',build_problem()),('hour13',from_dispatch(normal))]:
        r=benchmark(problem,restarts=1 if a.fast else 3,maxiter=35 if a.fast else 100,shots=2048 if a.fast else 8192)
        if name=='hour13': r['city_recourse']=evaluate_city_candidates(normal,r,13)
        (out/f'quantum_{name}.json').write_text(json.dumps(r,indent=2))
        for row in r['rows']:
            print(name,'p=',row['depth'],'qubits=',row['n_qubits'],'P(opt)=',round(row['p_optimal'],4),
                  'uniform=',row['p_optimal_uniform'],'feasible=',round(row['p_feasible'],4),'seconds=',round(row['runtime'],3))
    print('Saved audited results to',out.resolve())
