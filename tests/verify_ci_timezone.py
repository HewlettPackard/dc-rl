"""Checks that carbon intensity (CI files: UTC) is aligned with the weather and the simulation clock (EPW files: local
standard time). Run from the repo root: python tests/verify_ci_timezone.py"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, yaml

from sustaindc_env import SustainDC
from utils.managers import epw_utc_offset
from utils.utils_cf import obtain_paths

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    BASE = yaml.safe_load(f)
AGENTS = ['agent_ls', 'agent_dc', 'agent_bat']
JULY = slice(181 * 24, 212 * 24)  # hourly rows of July (2022 is not a leap year)
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def hourly_profile(x):
    x = np.asarray(x)[JULY]
    return x.reshape(-1, 24).mean(0)


print("1. Time zones read from the weather files")
check("Phoenix EPW is UTC-7", epw_utc_offset('USA_AZ_Phoenix-Sky.Harbor.epw') == -7)
check("LaGuardia EPW is UTC-5", epw_utc_offset('USA_NY_New.York-LaGuardia.epw') == -5)

for loc, offset in (('az', -7), ('ny', -5)):
    print(f"2. {loc.upper()}: CI rolled from UTC to local standard time")
    ci_loc, wea_loc = obtain_paths(loc)
    sdc = SustainDC({**BASE, 'location': loc, 'month': 6, 'agents': AGENTS})
    raw = pd.read_csv(f'data/CarbonIntensity/{ci_loc}_NG_&_avgCI.csv')
    check("shift equals the UTC offset", sdc.ci_m.ci_shift_hours == offset, f"{sdc.ci_m.ci_shift_hours} h")
    utc = raw['avg_CI'].values.astype(float)
    k = np.arange(181 * 24, 212 * 24)  # local hours in July; the env series has 4 steps per hour (row k at index 4k)
    env_hourly = sdc.ci_m.original_data[4 * k]
    # (the manager's 15-min interpolation drifts by a fraction of an hour over the year, so match the best lag)
    errs = {lag: np.abs(env_hourly - utc[k + lag]).mean() for lag in range(-12, 13)}
    best = min(errs, key=errs.get)
    check("env CI at local hour k matches the file's UTC hour k - offset", best == -offset,
          f"best lag {best} h, mean abs diff {errs[best]:.2f} g/kWh")

    # Physical sanity: the env clock, the weather and the CI now share local time.
    temp = sdc.weather_m.temperature_data[::4][:8760]
    hottest = int(hourly_profile(temp).argmax())
    check("hottest hour of July in local afternoon", 12 <= hottest <= 18, f"{hottest} h")
    if loc == 'az':
        sun = np.roll(raw['SUN'].values.astype(float), sdc.ci_m.ci_shift_hours)
        prof = hourly_profile(sun)
        daylight = np.where(prof > 0.5 * prof.max())[0]  # the file has some night-time "SUN" (storage/thermal)
        check("solar generation above half its peak only in local daytime (6-20 h)", daylight.min() >= 6 and daylight.max() <= 20,
              f"hours {daylight.min()}-{daylight.max()}")
        ci_min = int(hourly_profile(np.roll(utc, sdc.ci_m.ci_shift_hours)).argmin())
        check("cleanest grid hour in local daytime (solar)", 10 <= ci_min <= 19, f"{ci_min} h")

    print(f"3. {loc.upper()}: the env reads CI and weather at the same clock step")
    sdc.reset()
    ok = True
    for _ in range(200):
        ok &= sdc.ci_m.time_step == sdc.weather_m.time_step
        ok &= np.isclose(float(sdc.t_m.hour) * 4 % 96, sdc.ci_m.time_step % 96)
        sdc.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2})
    check("CI index, weather index and hour of day agree over 200 steps", ok)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
