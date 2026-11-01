import itertools
import numpy as np
import pytest
from energy.quantum import build_problem, bounded_weights, cvar, circuit, run_qaoa, benchmark, from_dispatch
from energy.scenarios import make_scenario
from energy.dispatch import solve

@pytest.mark.parametrize('cap',range(4))
def test_encoding(cap):
    w=bounded_weights(cap)
    values={sum(b*v for b,v in zip(bits,w)) for bits in itertools.product([0,1],repeat=len(w))}
    assert values==set(range(cap+1))

@pytest.mark.parametrize('caps',[(3,3,3,3),(2,1,3,0),(1,1,1,2),(0,0,3,3)])
def test_qubo_ising_ground_states(caps):
    p=build_problem(caps,target=min(5,sum(caps)))
    assert np.array_equal(np.isclose(p['energies'],p['energies'].min()),p['optimal'])
    assert np.all(p['actual'][p['feasible']]>=p['best']-1e-8)

def test_degenerate_uniform_baseline():
    p=build_problem(costs=[1,1,1,1])
    assert p['optimal'].sum()>1
    r=run_qaoa(p,0,shots=1024)
    assert r['p_optimal_uniform']==p['optimal'].sum()/2**p['n']
    assert r['p_optimal_uniform']!=1/256

def test_cvar_fractional_boundary():
    e=np.array([0.,10.,100.]);pr=np.array([.1,.3,.6])
    assert cvar(e,pr,.25)==pytest.approx(6.)
    assert cvar(e,pr,1)==pytest.approx(63.)

def test_circuit_signs_match_ising():
    from qiskit.quantum_info import Statevector
    p=build_problem();gamma=.37
    sv=np.asarray(Statevector.from_instruction(circuit(p,1,[gamma,0.])))
    norm=max(1.,max(abs(p['h'])),max(map(abs,p['J'].values())))
    expected=np.exp(-1j*gamma*(p['energies']-p['offset'])/norm)/np.sqrt(2**p['n'])
    assert np.allclose(sv,expected,atol=1e-10)

def test_aer_measured_results():
    p=build_problem()
    r=run_qaoa(p,1,restarts=1,maxiter=40,shots=2048,seed=7)
    assert r['best_sample'] is not None
    assert sum(r['best_sample']['levels'])==5
    assert 0<=r['p_optimal']<=r['p_feasible']<=1
    assert r['runtime']>0
    assert r['angles']

def test_discrete_classical_baseline():
    b=benchmark(build_problem(),depths=(0,),shots=512)
    assert b['classical_milp']['cost']==pytest.approx(b['rows'][0]['exact']['cost'])

def test_actual_hour_problem():
    r=solve(make_scenario());p=from_dispatch(r,13)
    assert p['n']<=8
    assert p['target']*p['step']==pytest.approx(r['slots'][13]['demand'])
    assert all(a<=b+1e-6 for a,b in zip(p['context']['modeled_capacity_kwh'],p['context']['available_kwh']))

def test_impossible_quantum_discretization():
    with pytest.raises(ValueError): build_problem((0,0,1,1),target=5)

def test_quantum_proposal_full_horizon_recourse():
    from energy.integration import evaluate_city_candidates
    baseline=solve(make_scenario());p=from_dispatch(baseline,13)
    experiment=benchmark(p,depths=(1,),restarts=1,maxiter=20,shots=1024)
    result=evaluate_city_candidates(baseline,experiment,13)
    assert result['candidates']
    assert result['selected_plan']['validation']['valid']
    assert result['selected_summary']['cost']<=baseline['summary']['cost']+1e-4
    assert result['candidates'][0]['status'].startswith('Physically feasible')
