"""Measured gate-based QAOA on a bounded, discrete one-slot dispatch benchmark.
No slack, no warm start. Exact enumeration audits encoding; not a quantum speedup.
"""
import time
import numpy as np
from scipy.optimize import minimize


def bounded_weights(cap):
    """Every integer 0..cap, no out-of-cap states. Some caps have duplicate encodings."""
    weights=[]; left=int(cap); w=1
    while left:
        take=min(w,left); weights.append(take); left-=take; w*=2
    return weights


def build_problem(capacities=(3,3,3,3), costs=(1.8,2.4,7.2,3.2), target=5, step=10., labels=None):
    started=time.perf_counter()
    if any(int(v)!=v for v in capacities): raise ValueError('Capacity units must be integers')
    caps=np.array(capacities,int); costs=np.array(costs,float)
    if len(caps)!=len(costs) or (caps<0).any() or (caps>3).any() or not np.isfinite(costs).all() or (costs<0).any():
        raise ValueError('Use nonnegative costs and integer capacities 0..3')
    if target<0 or int(target)!=target or target>caps.sum() or step<=0:
        raise ValueError('Discrete target cannot be met; change slot/resolution/capacities')
    weights=[]; owner=[]
    for s,cap in enumerate(caps):
        for w in bounded_weights(cap): weights.append(w); owner.append(s)
    n=len(weights)
    if not 1<=n<=10: raise ValueError('Benchmark needs 1..10 qubits')
    a=np.array(weights,float)
    cost=np.array([costs[s]*w*step for s,w in zip(owner,weights)])
    scale=max(1.,float(np.dot(costs,caps)*step))
    # Objective range <= 1. Integer equality residual is >=1 when invalid.
    # A=2 therefore makes every feasible state cheaper than every infeasible state.
    penalty=2.
    linear=cost/scale+penalty*(a*a-2*target*a)
    pairs={(i,j):float(2*penalty*a[i]*a[j]) for i in range(n) for j in range(i+1,n)}
    constant=penalty*target**2
    states=np.arange(2**n)
    bits=((states[:,None]>>np.arange(n))&1).astype(float)
    totals=bits@a; actual=bits@cost
    feasible=np.isclose(totals,target)
    best=float(actual[feasible].min())
    optimal=feasible & np.isclose(actual,best,atol=1e-8,rtol=0)
    energies=constant+bits@linear
    for (i,j),v in pairs.items(): energies+=v*bits[:,i]*bits[:,j]
    ground=np.isclose(energies,energies.min(),atol=1e-8,rtol=0)
    if not np.array_equal(ground,optimal): raise AssertionError('Penalty ground states disagree with exact dispatch optimum')
    # x=(1-Z)/2; diagonal offset does not affect QAOA probabilities.
    h=-linear/2.; J={}; offset=constant+linear.sum()/2.
    for (i,j),v in pairs.items():
        h[i]-=v/4; h[j]-=v/4; J[i,j]=v/4; offset+=v/4
    z=1-2*bits
    check=offset+z@h
    for (i,j),v in J.items(): check+=v*z[:,i]*z[:,j]
    if not np.allclose(check,energies,atol=1e-8,rtol=0): raise AssertionError('Ising/QUBO mismatch')
    return dict(enumeration_seconds=time.perf_counter()-started,n=n,weights=a,owner=owner,costs=costs,step=step,labels=labels or [f'Source {i}' for i in range(len(caps))],
                bits=bits,feasible=feasible,optimal=optimal,energies=energies,actual=actual,best=best,
                h=h,J=J,offset=offset,linear=linear,constant=constant,pairs=pairs,
                target=int(target),capacities=caps,penalty=penalty,scale=scale)


def from_dispatch(dispatch,hour=13):
    """Restricted one-hour benchmark, not a claim of solving the full MILP with 8 qubits.
Thermal cap is the certified day's scheduled discharge, a conservative electrical budget.
Charging is deducted from source availability. Backup is outside this normal-supply benchmark.
"""
    c=dispatch['config']; t=int(hour)
    if not 0<=t<c['hours']: raise ValueError('Hour outside horizon')
    dt=c['dt']
    try: ids=[[s['id'] for s in c['sources']].index(k) for k in ['solar','wind','grid','thermal']]
    except ValueError: raise ValueError('Scenario benchmark requires solar, wind, grid, thermal source IDs')
    demand=sum(c['demand'][t])*dt
    if demand<=0: raise ValueError('No demand in chosen slot')
    step=demand/5.
    available=np.minimum(c['availability'][t],[s['capacity'] for s in c['sources']])*dt-np.array(dispatch['charge'][t])
    available[ids[3]]=sum(dispatch['allocation'][t][ids[3]])
    caps=[min(3,max(0,int(np.floor(available[s]/step+1e-9)))) for s in ids]
    p=build_problem(caps,[c['sources'][s]['cost']*c.get('tariff_multipliers',np.ones((c['hours'],len(c['sources']))))[t][s] for s in ids],5,step,[c['sources'][s]['name'] for s in ids])
    p['context']=dict(hour=t,step_kwh=step,available_kwh=available[ids].tolist(),
                      modeled_capacity_kwh=(np.array(caps)*step).tolist(),
                      scope='Restricted aggregate one-slot benchmark. Does not replace the zone-wise multi-slot MILP. Measured proposals require full classical recourse before city-plan selection.')
    return p


def cvar(energy,prob,alpha):
    if not 0<alpha<=1: raise ValueError('CVaR alpha must be in (0,1]')
    order=np.argsort(energy,kind='stable'); p=prob[order]
    before=np.cumsum(p)-p
    take=np.minimum(p,np.maximum(0,alpha-before))
    return float(take@energy[order]/alpha)


def circuit(p,depth,angles,measure=False):
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(p['n']); qc.h(range(p['n']))
    # Global energy normalization changes angle scale, not the optimization problem.
    norm=max(1.,float(np.max(np.abs(p['h']))),max(map(abs,p['J'].values()),default=0))
    for layer in range(depth):
        gamma,beta=angles[layer],angles[depth+layer]
        for i,h in enumerate(p['h']):
            if abs(h)>1e-12: qc.rz(2*gamma*h/norm,i)
        for (i,j),v in p['J'].items():
            if abs(v)>1e-12: qc.rzz(2*gamma*v/norm,i,j)
        qc.rx(2*beta,range(p['n']))
    if measure: qc.measure_all()
    return qc


def run_qaoa(p,depth=1,alpha=.25,restarts=2,maxiter=70,shots=4096,seed=7):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    if depth not in (0,1,2,3) or not 1<=restarts<=8 or not 1<=maxiter<=500 or not 128<=shots<=65536:
        raise ValueError('QAOA settings out of supported bounds')
    if not 0<alpha<=1: raise ValueError('Invalid CVaR alpha')
    start=time.perf_counter(); rng=np.random.default_rng(seed)
    sim=AerSimulator(method='statevector',max_parallel_threads=1)
    if depth:
        from qiskit.circuit import ParameterVector
        theta=ParameterVector('theta',2*depth)
        template=circuit(p,depth,list(theta)); template.save_statevector()
        compiled=transpile(template,sim,optimization_level=1,seed_transpiler=seed)
        def probs(params):
            bound=compiled.assign_parameters(dict(zip(theta,params)))
            sv=np.asarray(sim.run(bound,seed_simulator=seed).result().get_statevector())
            prob=np.abs(sv)**2
            return prob/prob.sum()
        histories=[]; best_score=np.inf; best_angles=None; optimizers=[]
        for _ in range(restarts):
            history=[]
            def objective(params):
                value=cvar(p['energies'],probs(params),alpha); history.append(value); return value
            initial=np.r_[rng.uniform(0,2*np.pi,depth),rng.uniform(0,np.pi,depth)]
            r=minimize(objective,initial,method='COBYLA',options={'maxiter':maxiter,'rhobeg':.6,'tol':1e-4})
            histories.append(history)
            optimizers.append(dict(success=bool(r.success),message=str(r.message),evaluations=int(r.nfev)))
            if r.fun<best_score: best_score=float(r.fun); best_angles=r.x
        prob=probs(best_angles)
        measured=circuit(p,depth,best_angles,measure=True)
    else:
        best_angles=np.array([]); histories=[]; optimizers=[]
        prob=np.full(2**p['n'],1/2**p['n']); measured=circuit(p,0,[],measure=True)
    compiled_measure=transpile(measured,sim,optimization_level=1,seed_transpiler=seed)
    counts=sim.run(compiled_measure,shots=shots,seed_simulator=seed).result().get_counts()
    measured_prob=np.zeros(2**p['n'])
    for key,v in counts.items(): measured_prob[int(key.replace(' ',''),2)]=v/shots
    sampled=[i for i in range(len(prob)) if measured_prob[i]>0 and p['feasible'][i]]
    best=min(sampled,key=lambda i:p['actual'][i]) if sampled else None
    def decode(i):
        if i is None: return None
        levels=np.zeros(len(p['capacities']))
        for b,(owner,w) in enumerate(zip(p['owner'],p['weights'])): levels[owner]+=p['bits'][i,b]*w
        return dict(bitstring=format(i,f"0{p['n']}b"),levels=levels.astype(int).tolist(),energy_kwh=(levels*p['step']).tolist(),cost=float(p['actual'][i]))
    po=float(measured_prob[p['optimal']].sum()); pf=float(measured_prob[p['feasible']].sum())
    # Wilson interval treats final fixed-circuit measurement shots as binomial trials.
    denom=1+1.96**2/shots
    center=(po+1.96**2/(2*shots))/denom
    half=1.96*np.sqrt(po*(1-po)/shots+1.96**2/(4*shots**2))/denom
    return dict(depth=depth,alpha=alpha,n_qubits=p['n'],shots=shots,seed=seed,
                backend='Qiskit Aer statevector (ideal classical simulation)',runtime=time.perf_counter()-start,
                p_optimal=po,p_optimal_exact_distribution=float(prob[p['optimal']].sum()),
                p_optimal_uniform=float(p['optimal'].mean()),p_feasible=pf,p_feasible_uniform=float(p['feasible'].mean()),
                p_optimal_95ci=[float(max(0,center-half)),float(min(1,center+half))],
                best_sample=decode(best),exact=decode(int(np.flatnonzero(p['optimal'])[0])),
                gap_pct=100*(p['actual'][best]-p['best'])/p['best'] if best is not None and p['best']>0 else (0. if best is not None else None),
                histories=histories,angles=best_angles.tolist(),optimizer_runs=optimizers,
                transpiled_depth=compiled_measure.depth(),gate_counts=dict(compiled_measure.count_ops()),
                labels=p['labels'],context=p.get('context',{}),
                checks=dict(ground_state_matches_exact=True,ising_matches_qubo=True),
                circuit_text=str(measured.draw(output='text',fold=100)))


def benchmark(p=None,depths=(0,1,2),alpha=.25,restarts=2,maxiter=70,shots=4096,seed=7):
    t=time.perf_counter()
    p=build_problem() if p is None else p
    enumeration_seconds=p['enumeration_seconds']
    from scipy.optimize import milp, Bounds, LinearConstraint
    t=time.perf_counter()
    baseline=milp(p['costs']*p['step'],integrality=np.ones(len(p['capacities'])),
                  bounds=Bounds(np.zeros(len(p['capacities'])),p['capacities']),
                  constraints=LinearConstraint(np.ones((1,len(p['capacities']))),p['target'],p['target']))
    baseline_seconds=time.perf_counter()-t
    if not baseline.success or not np.isclose(baseline.fun,p['best']):
        raise AssertionError('Discrete MILP does not match enumeration')
    rows=[run_qaoa(p,d,alpha,restarts,maxiter,shots,seed) for d in depths]
    sampled=[(row['best_sample']['cost'],row['depth']) for row in rows if row['depth']>0 and row['best_sample']]
    matches=[depth for cost,depth in sampled if np.isclose(cost,baseline.fun,atol=1e-7,rtol=0)]
    return dict(rows=rows,enumeration_seconds=enumeration_seconds,
                selection=dict(method='Exact classical MILP (certified optimal); QAOA can tie, not beat this optimum',
                               cost=float(baseline.fun),qaoa_depths_matching_optimum=matches),
                classical_milp=dict(cost=float(baseline.fun),levels=np.rint(baseline.x).astype(int).tolist(),runtime=baseline_seconds),
                warning='No quantum advantage claimed. Simulator runtime and all optimization restarts included. Depth/CVaR gains are empirical, not guaranteed.')
