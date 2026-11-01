"""Replan only the unexecuted horizon using the observed thermal inventory."""
from copy import deepcopy
from .dispatch import solve


def reoptimize(previous,elapsed=1,observed_soc=None,availability=None,demand=None):
    c=deepcopy(previous['config'])
    if not 1<=elapsed<c['hours']: raise ValueError('elapsed must leave at least one slot')
    c['hours']-=elapsed
    if 'fixed_dispatch' in c:
        c['fixed_dispatch']={str(int(t)-elapsed):v for t,v in c['fixed_dispatch'].items() if int(t)>=elapsed}
    c['availability']=c['availability'][elapsed:] if availability is None else availability
    c['demand']=c['demand'][elapsed:] if demand is None else demand
    if 'tariff_multipliers' in c: c['tariff_multipliers']=c['tariff_multipliers'][elapsed:]
    c['storage']['initial']=previous['soc'][elapsed] if observed_soc is None else observed_soc
    return solve(c)
