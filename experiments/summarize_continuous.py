"""Refresh continuous tables from all 120 service-margin reruns."""
from pathlib import Path
import argparse
import csv
import gzip
import json
import statistics
import bootstrap

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
METHODS = ('RCSR-RHCR', 'RHCR-adapted', 'CC-RHCR')
CONDITIONS = ('fixed', 'moderate')


def scaled(value, scale):
    return None if value is None else value * scale


def read(name):
    with (RESULTS / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (RESULTS / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, default=ROOT / 'reruns/continuous')
    args = parser.parse_args()
    cases = {}
    for path in sorted(args.episodes.glob('*_t1000.json.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            d = json.load(f)
        j = d['experiment_job']
        key = d['display_method'], j['condition'], j['seed']
        assert d['status'] == 'completed' and j['suite'] == 'continuous'
        assert d['summary']['continuous_interface_version'] == 'service_margin_v2'
        assert key not in cases
        cases[key] = d
    expected = {(m, c, s) for m in METHODS for c in CONDITIONS for s in bootstrap.SEEDS}
    assert set(cases) == expected, (len(cases), len(expected))

    def values(method, condition, metric, section='metrics'):
        return {s: cases[method, condition, s][section].get(metric) for s in bootstrap.SEEDS}

    absolute = read('continuous_absolute.csv')
    for r in absolute:
        scale = bootstrap.METRICS[r['metric']][2]
        v = values(r['method'], r['condition'], r['metric'])
        stat = bootstrap.summarize(v, paired=True)
        r.update(mean=scaled(stat['mean'], scale),
                 sample_sd=scaled(stat['sample_sd'], scale),
                 ci95_low=scaled(stat['ci95'][0], scale),
                 ci95_high=scaled(stat['ci95'][1], scale), n_valid=stat['n_valid'])
    paired = read('continuous_paired.csv')
    for r in paired:
        scale = bootstrap.METRICS[r['metric']][2]
        a = values(r['candidate'], r['condition'], r['metric'])
        b = values(r['control'], r['condition'], r['metric'])
        valid = [s for s in bootstrap.SEEDS if a[s] is not None and b[s] is not None]
        stat = bootstrap.summarize({s: a[s]-b[s] if s in valid else None for s in bootstrap.SEEDS}, paired=True)
        r.update(candidate_mean=statistics.mean(a[s] for s in valid)*scale if valid else None,
                 control_mean=statistics.mean(b[s] for s in valid)*scale if valid else None,
                 difference=scaled(stat['mean'], scale),
                 ci95_low=scaled(stat['ci95'][0], scale),
                 ci95_high=scaled(stat['ci95'][1], scale), n_valid=stat['n_valid'])
    diagnostics = read('continuous_summary_diagnostics.csv')
    for r in diagnostics:
        stat = bootstrap.summarize(values(r['method'], r['condition'], r['metric'], 'summary'))
        r.update(mean=stat['mean'], sample_sd=stat['sample_sd'], n_valid=stat['n_valid'])
    detail = read('continuous_diagnostics_per_seed.csv')
    for r in detail:
        value = cases[r['method'], r['condition'], int(r['seed'])]['summary'].get(r['metric'])
        r.update(value=value, status='recorded' if value is not None else 'not_recorded_or_not_applicable')
    core = read('per_seed_core_metrics.csv')
    for r in core:
        if r['suite'] == 'continuous':
            r.update(cases[r['method'], r['condition'], int(r['seed'])]['metrics'])
    for name, rows in [('continuous_absolute.csv', absolute), ('continuous_paired.csv', paired),
                       ('continuous_summary_diagnostics.csv', diagnostics),
                       ('continuous_diagnostics_per_seed.csv', detail), ('per_seed_core_metrics.csv', core)]:
        write(name, rows)

    def mean(method, condition, metric, diagnostic=False):
        rows = diagnostics if diagnostic else absolute
        return next(r['mean'] for r in rows if (r['method'], r['condition'], r['metric']) == (method, condition, metric))

    table7 = []
    for condition in CONDITIONS:
        for method in METHODS:
            p90 = mean(method, condition, 'p90_response_time')
            n = next(r['n_valid'] for r in absolute if (r['method'], r['condition'], r['metric']) == (method, condition, 'p90_response_time'))
            table7.append(f"{'Zero delay' if condition == 'fixed' else 'Moderate'} & {method} & {p90:.2f} & {n} & {mean(method, condition, 'steady_state_throughput'):.4f} & {mean(method, condition, 'mean_continuous_start_error_m', True):.3f} " + r'\\')
    table8 = []
    for condition, method in [(c, m) for c in CONDITIONS for m in METHODS]:
        cells = []
        for event, exposure in [('uav_contact_events', 'uav_contact_pair_substep_fraction'), ('obstacle_contact_events', 'obstacle_contact_uav_substep_fraction'), ('boundary_violation_events', 'boundary_violation_uav_substep_fraction')]:
            cells.append(f'{mean(method, condition, event, True):.3f} / {mean(method, condition, exposure, True):.3g}')
        table8.append(' & '.join(['Zero delay' if condition == 'fixed' else 'Moderate', method] + cells) + r' \\')
    for name, body in [('table7_continuous.tex', table7), ('table8_contacts.tex', table8)]:
        p = RESULTS / name
        text = p.read_text(encoding='utf-8').replace('The zero-delay row applies separately to each method. ', '')
        start = text.index('\n', text.index(r'\midrule')) + 1
        end = text.index(r'\bottomrule', start)
        p.write_text(text[:start] + '\n'.join(body) + '\n' + text[end:], encoding='utf-8')
    print('Updated 120 continuous episodes, paired intervals, diagnostics, and Tables 7–8.')


if __name__ == '__main__':
    main()
