"""Classical recourse for measured one-slot QAOA source proposals.
This is hybrid candidate evaluation, not a quantum solution of the full city MILP.
"""
from copy import deepcopy
from .dispatch import solve


def evaluate_city_candidates(baseline,experiment,hour):
    c=baseline['config']; ids=[s['id'] for s in c['sources']]
    indices=[ids.index(k) for k in ['solar','wind','grid','thermal']]
    candidates=[];seen=set(); best=baseline;chosen='Classical baseline'
    def key(plan):
        # Preserve service, emergency and fairness priorities before comparing cost.
        return [s['objective'] for s in plan['stages'][:5]]+[plan['summary']['cost']+c['carbon_price']*plan['summary']['co2']]
    def no_worse(a,b):
        for x,y in zip(key(a),key(b)):
            if x<y-1e-4: return True
            if x>y+1e-4: return False
        return True
    for row in experiment['rows']:
        if row['depth']==0 or not row['best_sample']: continue
        proposal=row['best_sample']['energy_kwh']
        signature=tuple(round(v,7) for v in proposal)
        if signature in seen: continue
        seen.add(signature)
        cfg=deepcopy(c); totals=[0.]*len(ids)
        for s,value in zip(indices,proposal): totals[s]=value
        cfg.setdefault('fixed_dispatch',{})[str(hour)]=totals
        item=dict(depth=row['depth'],hour=hour,proposed_source_kwh=totals)
        try:
            candidate=solve(cfg)
            item.update(status='Physically feasible after full-horizon recourse',validation=candidate['validation'],
                        summary=candidate['summary'],no_worse_than_baseline=no_worse(candidate,baseline))
            if no_worse(candidate,best): best=candidate;chosen=f"QAOA p={row['depth']} + classical recourse"
        except RuntimeError as e:
            item.update(status='Rejected: cannot extend to full-city constraints',error=str(e))
        candidates.append(item)
    return dict(candidates=candidates,selected_method=chosen,selected_summary=best['summary'],
                baseline_summary=baseline['summary'],
                selected_plan=best,
                note='Quantum proposes one-slot source energy; classical MILP completes zone routes and storage horizon. Selection respects all lexicographic priorities before cost. A certified classical optimum cannot be beaten on the same model beyond solver tolerances.')
