from copy import deepcopy
import numpy as np
import pytest
from energy.scenarios import make_scenario
from energy.dispatch import solve,validate_plan

@pytest.mark.parametrize('stress',['normal','peak','cloudy','grid_outage','severe'])
def test_scenarios(stress):
    c=make_scenario(stress=stress);r=solve(c)
    assert r['validation']['valid']
    assert abs(r['summary']['served']+r['summary']['shortage']-r['summary']['demand'])<1e-4
    assert len(r['stages'])==6
    for stage in r['stages']: assert stage['gap']<1e-6

@pytest.mark.parametrize('mode',['weekday','weekend','holiday','event'])
def test_modes(mode):
    assert solve(make_scenario(mode=mode))['validation']['valid']

def test_no_unnecessary_backup():
    r=solve(make_scenario())
    assert r['summary']['backup']<1e-4
    assert r['summary']['shortage']<1e-4

def test_feeder_bottleneck_is_explicit():
    r=solve(make_scenario(stress='peak'))
    assert not r['summary']['critical_ok']
    assert r['validation']['critical_minimum_shortfall']>0

def test_zero_supply_does_not_fake_protection():
    c=make_scenario(hours=3);c['availability']=(np.zeros((3,5))).tolist()
    c['storage']['initial']=c['storage']['reserve']
    r=solve(c)
    assert not r['summary']['critical_ok']
    assert r['summary']['served']<1e-4
    assert r['summary']['shortage']==pytest.approx(r['summary']['demand'],abs=1e-4)

def test_critical_then_households_then_fairness():
    c=make_scenario(hours=1);c['availability']=[[0,0,400,0,0]]
    c['storage']['initial']=c['storage']['reserve']
    r=solve(c);served=np.array(r['allocation']).sum(1)[0]
    assert r['summary']['critical_ok']
    assert served[2]>=c['demand'][0][2]*.35-1e-4
    ratio=served[2:]/np.array(c['demand'][0][2:])
    assert ratio.max()-ratio.min()<1e-4

def test_reject_invalid_plan():
    c=make_scenario(hours=2);r=solve(c)
    r['allocation'][0][0][0]+=1e4
    assert not validate_plan(c,r)['valid']
    r['allocation'][0][0][0]=float('nan')
    assert not validate_plan(c,r)['valid']

def test_storage_conservation():
    c=make_scenario();r=solve(c);f=np.array(r['allocation']);q=np.array(r['charge'])
    assert np.allclose(np.diff(r['soc']),q.sum(1)*.9-f[:,3,:].sum(1)/.4,atol=1e-4)
    assert not ((q.sum(1)>1e-4)&(f[:,3,:].sum(1)>1e-4)).any()

def test_capacity_change_and_reoptimize():
    c=make_scenario(hours=2);c['sources'][2]['capacity']=20
    r=solve(c)
    assert np.array(r['allocation'])[:,2,:].sum(1).max()<=20+1e-4

def test_extra_source_zone_and_route():
    c=make_scenario(hours=2)
    c['sources'].append(dict(id='solar_rooftop',name='Rooftop',kind='renewable',capacity=40,cost=.5,co2=0))
    c['zones'].append(dict(id='school',name='School',kind='flexible',minimum=0,feeder=100))
    for row in c['availability']: row.append(40)
    for row in c['tariff_multipliers']: row.append(1)
    for row in c['demand']: row.append(50)
    c['line_limits']={'5:6':5}
    r=solve(c)
    assert np.array(r['allocation'])[:,5,6].max()<=5+1e-4
    assert r['validation']['valid']

def test_surplus_is_not_dispatchable_waste():
    c=make_scenario(hours=1);c['demand']=[[1]*6];c['availability']=[[760,320,1100,180,300]]
    r=solve(c)
    assert r['summary']['renewable_potential_unused']>0
    assert r['summary']['backup']<1e-4
    assert np.array(r['unused_capacity'])[0,2]>0

def test_half_hour_units():
    c=make_scenario(hours=2);c['dt']=.5
    r=solve(c)
    assert r['summary']['demand']==pytest.approx(np.array(c['demand']).sum()*.5)
    assert r['validation']['valid']

@pytest.mark.parametrize('field,value',[('dt',0),('carbon_price',-1),('demand',[[-1]*6]*24)])
def test_invalid_input(field,value):
    c=make_scenario();c[field]=value
    with pytest.raises(ValueError): solve(c)

def test_rolling_horizon_uses_observed_inventory():
    from energy.rolling import reoptimize
    r=solve(make_scenario())
    next_plan=reoptimize(r,elapsed=4,observed_soc=3000.)
    assert next_plan['soc'][0]==pytest.approx(3000.)
    assert len(next_plan['slots'])==20
    assert next_plan['validation']['valid']
