#!/usr/bin/env python3
"""Tail OH breakdown analysis for PTO2 scheduler.

Analyzes tail overhead from two sources:
  1. Per-task perf profiling data (perf_swimlane_*.json)
  2. AICPU scheduler loop breakdown (device log)

Usage:
    python tail_oh_breakdown.py                          # auto-select latest files
    python tail_oh_breakdown.py --perf-json <path>       # specify perf data
    python tail_oh_breakdown.py --device-log <path>      # specify device log
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path


def auto_select_perf_json():
    """Find the latest perf_swimlane_*.json in outputs/ directory."""
    outputs_dir = Path(__file__).parent.parent / 'outputs'
    files = sorted(outputs_dir.glob('perf_swimlane_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print(f"Error: No perf_swimlane_*.json files found in {outputs_dir}", file=sys.stderr)
        sys.exit(1)
    return files[0]


def auto_select_device_log():
    """Find the latest .log in ~/ascend/log/debug/device-0/."""
    log_dir = Path.home() / 'ascend' / 'log' / 'debug' / 'device-0'
    if not log_dir.exists():
        print(f"Error: Device log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)
    files = sorted(log_dir.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print(f"Error: No .log files found in {log_dir}", file=sys.stderr)
        sys.exit(1)
    return files[0]


def parse_scheduler_threads(log_path):
    """Parse device log for PTO2 scheduler stats per thread."""
    threads = {}
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = re.search(r'Thread (\d+): PTO2 scheduler stats: loops=(\d+), completed=(\d+), total=([\d.]+)us', line)
            if m:
                tid = int(m.group(1))
                threads[tid] = {
                    'loops': int(m.group(2)),
                    'completed': int(m.group(3)),
                    'total_us': float(m.group(4))
                }
            m = re.search(r'Thread (\d+):   scan=([\d.]+)us \(([\d.]+)%\), orch_drain=([\d.]+)us \(([\d.]+)%\), complete=([\d.]+)us \(([\d.]+)%\), dispatch=([\d.]+)us \(([\d.]+)%\)', line)
            if m:
                tid = int(m.group(1))
                threads[tid]['scan_us'] = float(m.group(2))
                threads[tid]['scan_pct'] = float(m.group(3))
                threads[tid]['orch_drain_us'] = float(m.group(4))
                threads[tid]['orch_drain_pct'] = float(m.group(5))
                threads[tid]['complete_us'] = float(m.group(6))
                threads[tid]['complete_pct'] = float(m.group(7))
                threads[tid]['dispatch_us'] = float(m.group(8))
                threads[tid]['dispatch_pct'] = float(m.group(9))
            m = re.search(r'Thread (\d+):   yield=([\d.]+)us \(([\d.]+)%, (\d+) calls', line)
            if m:
                tid = int(m.group(1))
                threads[tid]['yield_us'] = float(m.group(2))
                threads[tid]['yield_pct'] = float(m.group(3))
                threads[tid]['yield_calls'] = int(m.group(4))
            m = re.search(r'Thread (\d+):   lock\(ready_q\): wait=(\d+)us hold=(\d+)us \(scan=([\d]+)/([\d]+) orch=([\d]+)/([\d]+) complete=([\d]+)/([\d]+) dispatch=([\d]+)/([\d]+)\)', line)
            if m:
                tid = int(m.group(1))
                threads[tid]['lock_wait_us'] = int(m.group(2))
                threads[tid]['lock_hold_us'] = int(m.group(3))
                threads[tid]['lock_scan_wait'] = int(m.group(4))
                threads[tid]['lock_scan_hold'] = int(m.group(5))
                threads[tid]['lock_complete_wait'] = int(m.group(8))
                threads[tid]['lock_complete_hold'] = int(m.group(9))
                threads[tid]['lock_dispatch_wait'] = int(m.group(10))
                threads[tid]['lock_dispatch_hold'] = int(m.group(11))
            m = re.search(r'Thread (\d+):   fanout: total_traversed=(\d+), max_len=(\d+), avg=([\d.]+)', line)
            if m:
                tid = int(m.group(1))
                threads[tid]['fanout_total'] = int(m.group(2))
                threads[tid]['fanout_max'] = int(m.group(3))
                threads[tid]['fanout_avg'] = float(m.group(4))
            m = re.search(r'Thread (\d+):   lock\(fanout\): spin=(\d+)us hold=(\d+)us', line)
            if m:
                tid = int(m.group(1))
                threads[tid]['fanout_spin_us'] = int(m.group(2))
                threads[tid]['fanout_hold_us'] = int(m.group(3))
    return threads


def main():
    parser = argparse.ArgumentParser(
        description='Tail OH breakdown analysis for PTO2 scheduler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # auto-select latest files
  %(prog)s --perf-json outputs/perf_swimlane_*.json
  %(prog)s --device-log ~/ascend/log/debug/device-0/device-*.log
        """
    )
    parser.add_argument('--perf-json', help='Path to perf_swimlane_*.json file. If not specified, uses the latest in outputs/')
    parser.add_argument('--device-log', help='Path to device log file. If not specified, uses the latest in ~/ascend/log/debug/device-0/')
    args = parser.parse_args()

    # Resolve input paths
    perf_path = Path(args.perf_json) if args.perf_json else auto_select_perf_json()
    log_path = Path(args.device_log) if args.device_log else auto_select_device_log()

    if not perf_path.exists():
        print(f"Error: Perf JSON not found: {perf_path}", file=sys.stderr)
        return 1
    if not log_path.exists():
        print(f"Error: Device log not found: {log_path}", file=sys.stderr)
        return 1

    print(f"Perf data:  {perf_path}")
    print(f"Device log: {log_path}")

    # === Part 1: Per-task time breakdown from perf data ===
    with open(perf_path) as f:
        data = json.load(f)
    tasks = data['tasks']
    n_total = len(tasks)

    all_exec = sum(t['duration_us'] for t in tasks)
    all_head = sum(t['start_time_us'] - t['dispatch_time_us'] for t in tasks)
    all_tail = sum(t['finish_time_us'] - t['end_time_us'] for t in tasks)
    min_disp = min(t['dispatch_time_us'] for t in tasks)
    max_fin = max(t['finish_time_us'] for t in tasks)
    wall = max_fin - min_disp

    all_latency = all_exec + all_head + all_tail

    print()
    print('=' * 90)
    print('Part 1: Per-task time breakdown (from perf profiling data)')
    print('=' * 90)
    print(f'Total tasks: {n_total}')
    print(f'Wall-clock:  {wall:.1f} us')
    print()
    fmt = "  {:<35} {:>12} {:>14} {:>13}"
    print(fmt.format('Component', 'Total (us)', 'Avg/task (us)', '% of Latency'))
    print('  ' + '-' * 78)
    print(fmt.format('Kernel Exec (end-start)', f'{all_exec:.1f}', f'{all_exec/n_total:.2f}', f'{all_exec/all_latency*100:.1f}%'))
    print(fmt.format('Head OH (start-dispatch)', f'{all_head:.1f}', f'{all_head/n_total:.2f}', f'{all_head/all_latency*100:.1f}%'))
    print(fmt.format('Tail OH (finish-end)', f'{all_tail:.1f}', f'{all_tail/n_total:.2f}', f'{all_tail/all_latency*100:.1f}%'))
    print()

    # === Part 2: AICPU scheduler loop breakdown from device log ===
    threads = parse_scheduler_threads(log_path)
    n_threads = len(threads)

    print('=' * 90)
    print('Part 2: AICPU scheduler loop breakdown (from device log)')
    print(f'  {n_threads} scheduler threads')
    print('=' * 90)
    print()
    fmt2 = "  {:<10} {:>7} {:>10} {:>11}"
    print(fmt2.format('Thread', 'Loops', 'Completed', 'Total (us)'))
    print('  ' + '-' * 42)
    for tid in sorted(threads.keys()):
        t = threads[tid]
        print(fmt2.format('T'+str(tid), t['loops'], t['completed'], f"{t['total_us']:.1f}"))
    total_us = sum(t['total_us'] for t in threads.values())
    total_completed = sum(t['completed'] for t in threads.values())
    total_loops = sum(t['loops'] for t in threads.values())
    print(fmt2.format('SUM', total_loops, total_completed, f'{total_us:.1f}'))
    print()

    phases = ['scan', 'orch_drain', 'complete', 'dispatch', 'yield']
    phase_labels = {
        'scan':       'Scan (discover new root tasks)',
        'orch_drain': 'Orch drain (wait for orchestrator)',
        'complete':   'Complete (poll handshake, resolve fanout)',
        'dispatch':   'Dispatch (pop queue, build payload, flush)',
        'yield':      'Yield (no progress, thread_yield)',
    }

    fmt3 = "  {:<50} {:>11} {:>10} {:>14}"
    print(fmt3.format('Phase', 'Total (us)', '% of total', 'Avg/task (us)'))
    print('  ' + '-' * 89)
    phase_totals = {}
    for p in phases:
        key = p + '_us'
        tot = sum(t.get(key, 0) for t in threads.values())
        phase_totals[p] = tot
        pct = tot / total_us * 100 if total_us > 0 else 0
        avg = tot / total_completed if total_completed > 0 else 0
        print(fmt3.format(phase_labels[p], f'{tot:.1f}', f'{pct:.1f}%', f'{avg:.2f}'))

    print()

    # Lock contention breakdown
    fmt4 = "  {:<50} {:>11} {:>10}"
    print(fmt4.format('Lock contention (ready_q)', 'Total (us)', '% of total'))
    print('  ' + '-' * 75)
    lock_wait = sum(t.get('lock_wait_us', 0) for t in threads.values())
    lock_hold = sum(t.get('lock_hold_us', 0) for t in threads.values())
    print(fmt4.format('  wait (spinning for lock)', str(lock_wait), f'{lock_wait/total_us*100:.1f}%' if total_us > 0 else '0.0%'))
    print(fmt4.format('  hold (inside critical section)', str(lock_hold), f'{lock_hold/total_us*100:.1f}%' if total_us > 0 else '0.0%'))
    print()

    # Lock wait breakdown by phase
    print('  Lock wait by phase:')
    for p in ['scan', 'complete', 'dispatch']:
        w = sum(t.get(f'lock_{p}_wait', 0) for t in threads.values())
        h = sum(t.get(f'lock_{p}_hold', 0) for t in threads.values())
        print(f'    {p:<12}  wait={w:>6} us  hold={h:>6} us')
    print()

    # Fanout
    fanout_total = sum(t.get('fanout_total', 0) for t in threads.values())
    fanout_max = max((t.get('fanout_max', 0) for t in threads.values()), default=0)
    fanout_spin = sum(t.get('fanout_spin_us', 0) for t in threads.values())
    fanout_hold = sum(t.get('fanout_hold_us', 0) for t in threads.values())
    print(f'  Fanout traversal: total={fanout_total}, max_len={fanout_max}, lock spin={fanout_spin}us hold={fanout_hold}us')

    print()
    print('=' * 90)
    print('Part 3: Tail OH distribution & cause analysis')
    print('=' * 90)
    print()

    tails = [t['finish_time_us'] - t['end_time_us'] for t in tasks]
    tails.sort()
    n = len(tails)
    print(f'  Tail OH distribution (N={n}):')
    for pct_val in [10, 25, 50, 75, 90, 95, 99]:
        idx = min(int(n * pct_val / 100), n - 1)
        print(f'    P{pct_val:<4}  {tails[idx]:>7.1f} us')
    print(f'    Max:   {tails[-1]:>7.1f} us')
    print(f'    Mean:  {sum(tails)/n:>7.1f} us')
    print()

    # Scheduler loop time
    avg_loop_us = total_us / total_loops if total_loops > 0 else 0
    avg_tail_oh = sum(tails) / n
    loop_ratio = avg_tail_oh / avg_loop_us if avg_loop_us > 0 else 0
    print(f'  Avg scheduler loop iteration: {avg_loop_us:.1f} us (\u2248 avg polling interval per loop)')
    print(f'  With {n_threads} threads sharing {total_loops} loops over {total_us/n_threads:.0f} us wall each:' if n_threads > 0 else '')
    print()
    print(f'  Scheduler CPU time breakdown (per completed task):')

    # Build phase data with sub-items for sorting
    phase_details = {
        'dispatch': {
            'label': 'Dispatch phase (build payload + cache flush)',
            'total': phase_totals['dispatch'],
            'sub_items': [
                ('Lock wait (ready_q pop)', sum(t.get('lock_dispatch_wait', 0) for t in threads.values())),
                ('Lock hold + build + dc cvac/civac + dsb sy', phase_totals['dispatch'] - sum(t.get('lock_dispatch_wait', 0) for t in threads.values())),
            ]
        },
        'complete': {
            'label': 'Complete phase (poll + fanout resolve)',
            'total': phase_totals['complete'],
            'sub_items': [
                ('Lock wait (ready_q push)', sum(t.get('lock_complete_wait', 0) for t in threads.values())),
                ('Fanout traversal + atomic ops', phase_totals['complete'] - sum(t.get('lock_complete_wait', 0) for t in threads.values())),
            ]
        },
        'scan': {
            'label': 'Scan phase (new task discovery)',
            'total': phase_totals['scan'],
            'sub_items': []
        },
        'yield': {
            'label': 'Yield (idle)',
            'total': phase_totals['yield'],
            'sub_items': []
        },
    }

    # Sort by total descending
    for p, detail in sorted(phase_details.items(), key=lambda x: x[1]['total'], reverse=True):
        per_task = detail['total'] / total_completed if total_completed > 0 else 0
        pct = detail['total'] / total_us * 100 if total_us > 0 else 0
        print(f'    - {detail["label"]:<50} {per_task:.2f} us/task  ({pct:.1f}% of scheduler CPU)')
        for sub_label, sub_total in detail['sub_items']:
            sub_per_task = sub_total / total_completed if total_completed > 0 else 0
            print(f'        {sub_label:<48} {sub_per_task:.2f} us/task')

    print()
    print(f'  Avg Tail OH = {avg_tail_oh:.1f} us \u2248 {loop_ratio:.1f} \u00d7 avg loop iteration ({avg_loop_us:.1f} us)')
    print(f'  \u2192 on average, a completed task waits ~{loop_ratio:.1f} loop iterations before being detected')
    print()

    # Data-driven insight: find the dominant phase (excluding yield)
    work_phases = {p: phase_totals[p] for p in ['scan', 'orch_drain', 'complete', 'dispatch']}
    dominant_phase = max(work_phases, key=work_phases.get)
    dominant_pct = work_phases[dominant_phase] / total_us * 100 if total_us > 0 else 0
    print(f'  Key insight: {phase_labels[dominant_phase].split(" (")[0]} phase consumes ~{dominant_pct:.0f}% of scheduler CPU.')
    if dominant_phase == 'dispatch':
        print(f'  Within dispatch, cache flush (dc cvac + dsb sy) is the dominant cost.')
        print(f'  Each dsb sy stalls the AICPU pipeline until all prior dc ops complete.')
    elif dominant_phase == 'complete':
        print(f'  Within complete, handshake polling and fanout resolution dominate.')
        print(f'  Reordering the scheduler loop to process completed tasks first may help.')
    print('=' * 90)

    return 0


if __name__ == '__main__':
    sys.exit(main())
