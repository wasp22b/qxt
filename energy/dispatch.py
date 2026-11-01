"""Multi-period MILP. Lexicographic safety, backup, fairness, then operating cost.
All flows are electrical kWh per slot; SOC is kWh_th. HiGHS via scipy.milp.
"""
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

TOL = 2e-5


def validate_inputs(c):
    T, S, Z = c['hours'], len(c['sources']), len(c['zones'])
    if not 1 <= T <= 48 or not 1 <= S <= 12 or not 1 <= Z <= 20:
        raise ValueError('Use 1..48 slots, 1..12 sources, 1..20 zones')
    if sum(s['kind']=='storage' for s in c['sources']) != 1:
        raise ValueError('Exactly one thermal storage converter is required')
    if len({s['id'] for s in c['sources']}) != S or len({z['id'] for z in c['zones']}) != Z:
        raise ValueError('Source and zone IDs must be unique')
    if any(s['kind'] not in ('renewable','storage','grid','backup') for s in c['sources']):
        raise ValueError('Invalid source kind')
    if any(z['kind'] not in ('critical','household','flexible') for z in c['zones']):
        raise ValueError('Invalid zone kind')
    for k, shape in [('availability', (T,S)), ('demand', (T,Z))]:
        a = np.array(c[k], float)
        if a.shape != shape or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f'{k}: expected finite nonnegative {shape}')
    prices=np.array(c.get('tariff_multipliers',np.ones((T,S))),float)
    if prices.shape!=(T,S) or not np.isfinite(prices).all() or (prices<0).any():
        raise ValueError('tariff_multipliers must be a finite nonnegative hours × sources matrix')
    st = c['storage']
    if not (0 <= st['reserve'] <= st['initial'] <= st['capacity'] and
            0 < st['eta_charge'] <= 1 and 0 < st['eta_discharge'] <= 1 and st['charge_limit'] >= 0):
        raise ValueError('Invalid thermal storage bounds or efficiencies')
    if not np.isfinite(list(st.values())).all() or not np.isfinite(c['dt']) or c['dt'] <= 0:
        raise ValueError('Invalid time interval/storage number')
    if not np.isfinite(c['carbon_price']) or c['carbon_price'] < 0:
        raise ValueError('Carbon price must be finite and nonnegative')
    for s in c['sources']:
        if any(not np.isfinite(s[k]) or s[k] < 0 for k in ['capacity','cost','co2']):
            raise ValueError('Source capacity, cost and emissions must be finite and nonnegative')
    for z in c['zones']:
        if not 0 <= z['minimum'] <= 1 or not np.isfinite(z['feeder']) or z['feeder'] < 0:
            raise ValueError('Invalid zone minimum or feeder limit')
    for key, values in c.get('fixed_dispatch', {}).items():
        if not str(key).isdigit() or not 0<=int(key)<T or len(values)!=S or not np.isfinite(values).all() or min(values)<0:
            raise ValueError('fixed_dispatch maps slot indices to nonnegative source electrical kWh totals')
    for key, v in c.get('line_limits', {}).items():
        try:
            s,z = map(int,key.split(':'))
            assert 0 <= s < S and 0 <= z < Z and np.isfinite(v) and v >= 0
        except Exception:
            raise ValueError('Line limits use sourceIndex:zoneIndex and nonnegative kW')


def solve(c):
    validate_inputs(c)
    start = time.perf_counter()
    T, S, Z, dt = c['hours'], len(c['sources']), len(c['zones']), c['dt']
    thermal=next(i for i,s in enumerate(c['sources']) if s['kind']=='storage')
    backups=[i for i,s in enumerate(c['sources']) if s['kind']=='backup']
    renewables=[i for i,s in enumerate(c['sources']) if s['kind']=='renewable']
    D = np.array(c['demand'],float)
    A = np.minimum(np.array(c['availability'],float), [s['capacity'] for s in c['sources']]) * dt
    # demand input is kW; convert to energy for each interval.
    D *= dt
    st = c['storage']
    prices=np.array(c.get('tariff_multipliers',np.ones((T,S))))*np.array([s['cost'] for s in c['sources']])
    x = np.arange(T*S*Z).reshape(T,S,Z)
    pos = x.size
    charge = np.arange(pos, pos+T*S).reshape(T,S); pos += T*S
    soc = np.arange(pos,pos+T+1); pos += T+1
    unmet = np.arange(pos,pos+T*Z).reshape(T,Z); pos += T*Z
    deficit = np.arange(pos,pos+T*Z).reshape(T,Z); pos += T*Z
    fair = np.arange(pos,pos+T); pos += T
    mode = np.arange(pos,pos+T); pos += T
    n = pos
    lo = np.zeros(n); hi = np.full(n,np.inf); integer = np.zeros(n)
    hi[unmet] = D; hi[deficit] = D
    hi[fair] = 1.; hi[mode] = 1.; integer[mode] = 1
    lo[soc] = st['reserve']; hi[soc] = st['capacity']
    lo[soc[0]] = hi[soc[0]] = st['initial']
    for s in range(S):
        if c['sources'][s]['kind'] != 'renewable':
            hi[charge[:,s]] = 0.
    rows, lbs, ubs = [], [], []
    def add(terms, lb=-np.inf, ub=np.inf):
        rows.append(dict(terms)); lbs.append(lb); ubs.append(ub)
    for t in range(T):
        for z in range(Z):
            add([(int(i),1.) for i in x[t,:,z]]+[(int(unmet[t,z]),1.)], D[t,z], D[t,z])
            add([(int(i),1.) for i in x[t,:,z]], ub=c['zones'][z]['feeder']*dt)
            add([(int(i),1.) for i in x[t,:,z]]+[(int(deficit[t,z]),1.)],
                lb=D[t,z]*c['zones'][z]['minimum'])
            if c['zones'][z]['kind'] != 'critical' and D[t,z] > 0:
                add([(int(i),1.) for i in x[t,:,z]]+[(int(fair[t]),-D[t,z])], lb=0.)
        for s in range(S):
            add([(int(i),1.) for i in x[t,s,:]]+[(int(charge[t,s]),1.)],ub=A[t,s])
            for z in range(Z):
                lim = c.get('line_limits',{}).get(f'{s}:{z}')
                if lim is not None:
                    hi[x[t,s,z]] = lim*dt
        add([(int(i),1.) for i in charge[t]]+[(int(mode[t]),-st['charge_limit']*dt)],ub=0.)
        # mode=1 allows charge only; mode=0 allows discharge only.
        add([(int(i),1.) for i in x[t,thermal]]+[(int(mode[t]),A[t,thermal])],ub=A[t,thermal])
        add([(int(soc[t+1]),1.),(int(soc[t]),-1.)] +
            [(int(i),-st['eta_charge']) for i in charge[t]] +
            [(int(i),1/st['eta_discharge']) for i in x[t,thermal]],0.,0.)
    for key,values in c.get('fixed_dispatch',{}).items():
        t=int(key)
        for s,value in enumerate(values):
            add([(int(i),1.) for i in x[t,s]],float(value),float(value))
    stages=[]
    def optimize(obj, name):
        mat=lil_matrix((len(rows),n))
        for r, terms in enumerate(rows):
            for col,v in terms.items(): mat[r,col]=v
        r=milp(obj,integrality=integer,bounds=Bounds(lo,hi),
               constraints=LinearConstraint(mat.tocsr(),lbs,ubs),
               options={'time_limit':30.,'mip_rel_gap':1e-8})
        if not r.success or r.x is None:
            raise RuntimeError(f'{name}: solver did not certify success: {r.message}')
        stages.append(dict(stage=name,objective=float(obj@r.x),gap=float(r.mip_gap or 0)))
        return r.x
    def freeze(obj,v):
        terms=[(int(i),float(obj[i])) for i in np.flatnonzero(obj)]
        if terms: add(terms,ub=float(obj@v)+1e-6)
    for kind,name in [('critical','1 · Critical minimum shortfall'),('household','2 · Household minimum shortfall')]:
        obj=np.zeros(n)
        for z in range(Z):
            if c['zones'][z]['kind']==kind: obj[deficit[:,z]]=1.
        v=optimize(obj,name); freeze(obj,v)
    obj=np.zeros(n); obj[unmet]=1.
    v=optimize(obj,'3 · Total unmet energy'); freeze(obj,v)
    obj=np.zeros(n); obj[x[:,backups,:]]=1.
    v=optimize(obj,'4 · Emergency generation only when necessary'); freeze(obj,v)
    obj=np.zeros(n); obj[fair]=-1.
    v=optimize(obj,'5 · Max-min noncritical service per interval')
    lo[fair]=np.maximum(0,v[fair]-1e-7)
    obj=np.zeros(n)
    for s,src in enumerate(c['sources']):
        unit=prices[:,s]+c['carbon_price']*src['co2']
        obj[x[:,s,:]]=unit[:,None]; obj[charge[:,s]]=unit
    v=optimize(obj,'6 · Minimum cost + optional carbon price')
    flows=np.maximum(0,v[x]); ch=np.maximum(0,v[charge]); states=v[soc]
    used=flows.sum(axis=2)+ch
    served=flows.sum(axis=1)
    shortage=np.maximum(0,D-served)
    unused=np.maximum(0,A-used)
    cost=float((used*prices).sum())
    co2=float(sum(used[:,s].sum()*src['co2'] for s,src in enumerate(c['sources'])))
    slots=[]
    for t in range(T):
        ratio=np.divide(served[t],D[t],out=np.ones(Z),where=D[t]>0)
        non=ratio[[z['kind']!='critical' for z in c['zones']]]
        critical_ok=all(served[t,z]+TOL>=D[t,z]*zone['minimum'] for z,zone in enumerate(c['zones']) if zone['kind']=='critical')
        actions=[]
        if flows[t,backups].sum()>TOL: actions.append('Emergency backup required to preserve service.')
        if shortage[t].sum()>TOL: actions.append('Unavoidable load shedding; protect minima, then share noncritical shortages.')
        if not critical_ok: actions.append('CRITICAL ALERT: physical capacity cannot protect all critical minima.')
        renewable_unused=float(unused[t,renewables].sum())
        if renewable_unused>TOL: actions.append('Renewable potential remains: consider flexible loads/export; export is not modeled.')
        slots.append(dict(hour=t,demand=float(D[t].sum()),supply=float(A[t].sum()),served=float(served[t].sum()),
                          shortage=float(shortage[t].sum()),backup=float(flows[t,backups].sum()),critical_ok=critical_ok,
                          fair_floor=float(non.min()),jain=float(non.sum()**2/(len(non)*(non@non))) if non@non>1e-12 else 0.,
                          cost=float((used[t]*prices[t]).sum()),
                          renewable_potential_unused=renewable_unused,actions=actions))
    out=dict(config=c,allocation=flows.tolist(),charge=ch.tolist(),soc=states.tolist(),
             unmet=shortage.tolist(),unused_capacity=unused.tolist(),slots=slots,stages=stages,
             summary=dict(demand=float(D.sum()),served=float(served.sum()),shortage=float(shortage.sum()),
                          cost=cost,co2=co2,backup=float(flows[:,backups,:].sum()),
                          critical_ok=all(s['critical_ok'] for s in slots),
                          renewable_potential_unused=float(unused[:,renewables].sum()),
                          runtime=time.perf_counter()-start))
    out['validation']=validate_plan(c,out)
    if not out['validation']['valid']: raise RuntimeError(str(out['validation']))
    return out


def validate_plan(c,r):
    """Independent physical audit; reject malformed/nonfinite plans, not only bad totals."""
    errors=[]
    T,S,Z=c['hours'],len(c['sources']),len(c['zones'])
    try:
        f=np.array(r['allocation'],float); q=np.array(r['charge'],float); soc=np.array(r['soc'],float)
        if f.shape!=(T,S,Z) or q.shape!=(T,S) or soc.shape!=(T+1,):
            return dict(valid=False,errors=['Array shape mismatch'])
        if not all(np.isfinite(a).all() for a in [f,q,soc]):
            return dict(valid=False,errors=['Nonfinite plan'])
        dt=c['dt']; st=c['storage']; D=np.array(c['demand'])*dt
        thermal=next(i for i,s in enumerate(c['sources']) if s['kind']=='storage')
        nonrenewable=[i for i,s in enumerate(c['sources']) if s['kind']!='renewable']
        A=np.minimum(c['availability'],[s['capacity'] for s in c['sources']])*dt
        checks={
            'negative allocation': (f < -TOL).any() or (q < -TOL).any(),
            'source overcapacity': (f.sum(2)+q > A+TOL).any(),
            'zone oversupply': (f.sum(1)>D+TOL).any(),
            'feeder overcapacity': (f.sum(1)>np.array([z['feeder'] for z in c['zones']])*dt+TOL).any(),
            'storage bounds': (soc<st['reserve']-TOL).any() or (soc>st['capacity']+TOL).any(),
            'initial SOC': abs(soc[0]-st['initial'])>TOL,
            'storage conservation': not np.allclose(np.diff(soc),q.sum(1)*st['eta_charge']-f[:,thermal,:].sum(1)/st['eta_discharge'],atol=TOL,rtol=0),
            'charge rate': (q.sum(1)>st['charge_limit']*dt+TOL).any(),
            'nonrenewable charging': (q[:,nonrenewable] > TOL).any(),
            'simultaneous charge/discharge': ((q.sum(1)>TOL)&(f[:,thermal,:].sum(1)>TOL)).any(),
        }
        for key,lim in c.get('line_limits',{}).items():
            s,z=map(int,key.split(':'))
            if (f[:,s,z]>lim*dt+TOL).any(): errors.append('route capacity '+key)
        for key,values in c.get('fixed_dispatch',{}).items():
            if not np.allclose(f[int(key)].sum(1),values,atol=TOL,rtol=0): errors.append('fixed source dispatch violated')
        errors += [key for key,bad in checks.items() if bad]
        critical=[]
        for z,zone in enumerate(c['zones']):
            if zone['kind']=='critical':
                critical.extend(np.maximum(0,D[:,z]*zone['minimum']-f[:,:,z].sum(1)).tolist())
        return dict(valid=not errors,errors=errors,critical_minimum_shortfall=float(sum(critical)),
                    note='Physical feasibility is distinct from service adequacy: impossible minima are flagged, not hidden.')
    except (KeyError,ValueError,IndexError,TypeError) as e:
        return dict(valid=False,errors=[str(e)])
