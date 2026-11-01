"""Synthetic, reproducible hourly electrical energy inputs; not a grid power-flow model."""
from copy import deepcopy
import math

SOURCES = [
    dict(id='solar', name='Solar PV', kind='renewable', capacity=760, cost=1.8, co2=0.045),
    dict(id='wind', name='Wind farm', kind='renewable', capacity=320, cost=2.4, co2=0.012),
    dict(id='grid', name='Utility grid', kind='grid', capacity=1100, cost=7.2, co2=0.65),
    dict(id='thermal', name='Thermal storage → power', kind='storage', capacity=180, cost=3.2, co2=0.03),
    dict(id='backup', name='Emergency generator', kind='backup', capacity=300, cost=24., co2=0.9),
]
ZONES = [
    dict(id='hospital', name='Hospital', kind='critical', minimum=1., base=110, feeder=150),
    dict(id='emergency', name='Water & emergency', kind='critical', minimum=1., base=85, feeder=120),
    dict(id='residential', name='Residential', kind='household', minimum=.35, base=310, feeder=650),
    dict(id='commercial', name='Commercial', kind='flexible', minimum=0., base=210, feeder=360),
    dict(id='buildings', name='Office buildings', kind='flexible', minimum=0., base=170, feeder=300),
    dict(id='industry', name='Small industry', kind='flexible', minimum=0., base=170, feeder=300),
]


def make_scenario(mode='weekday', stress='normal', hours=24, overrides=None):
    if mode not in ('weekday', 'weekend', 'holiday', 'event'):
        raise ValueError('Unknown day mode')
    if stress not in ('normal', 'peak', 'cloudy', 'grid_outage', 'severe'):
        raise ValueError('Unknown stress scenario')
    if not 1 <= hours <= 48:
        raise ValueError('hours must be 1..48')
    sources, zones = deepcopy(SOURCES), deepcopy(ZONES)
    availability, demand = [], []
    for t in range(hours):
        h = t % 24
        solar = max(0, math.sin(math.pi * (h - 6) / 12))
        wind = .54 + .20 * math.sin((h + 2) * math.pi / 12)
        availability.append([760 * solar, 320 * wind, 1100., 180., 300.])
        home = .68 + .56 * math.exp(-((h - 20) / 3.3)**2) + .35 * math.exp(-((h - 7) / 2)**2)
        work = .22 + .92 * math.exp(-((h - 13) / 4.8)**4)
        if mode in ('weekend', 'holiday'):
            home *= 1.16
            work *= .57
        event = 1.5 if mode == 'event' and 16 <= h <= 22 else 1.
        demand.append([110., 85., 310 * home * event, 210 * work * event,
                       170 * work, 170 * (.80 if 7 <= h < 20 else .40)])
        if stress == 'peak':
            demand[-1] = [v * 1.45 for v in demand[-1]]
        if stress in ('cloudy', 'severe'):
            availability[-1][0] *= .12
            availability[-1][1] *= .35
        if stress == 'grid_outage' and 16 <= h <= 21:
            availability[-1][2] = 0.
        if stress == 'severe':
            availability[-1][2] = 80.
            availability[-1][4] = 65.
    cfg = dict(mode=mode, stress=stress, hours=hours, dt=1., sources=sources, zones=zones,
               availability=availability, demand=demand,
               storage=dict(initial=4200., capacity=6000., reserve=600., eta_charge=.9,
                            eta_discharge=.4, charge_limit=300.),
               carbon_price=0., line_limits={},
               tariff_multipliers=[[1.,1.,1.5 if 17<=t%24<=22 else (.86 if t%24<7 else 1.),1.,1.] for t in range(hours)])
    if overrides:
        # Overrides are explicit JSON inputs, never executable expressions.
        for key, val in overrides.items():
            if key not in cfg:
                raise ValueError(f'Unknown scenario field: {key}')
            cfg[key] = val
    return cfg
