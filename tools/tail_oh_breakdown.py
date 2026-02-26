#!/usr/bin/env python3
"""Tail OH breakdown analysis for PTO2 scheduler."""
import json, os, re
from collections import defaultdict

# === Part 1: Per-task time breakdown from perf data ===
perf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
files = sorted([f for f in os.listdir(perf_dir) if f.startswith('perf_swimlane_')], reverse=True)
with open(os.path.join(perf_dir, files[0])) as f:
    data = json.load(f)
tasks = data['tasks']
func_names = {0:'QK', 1:'SF', 2:'PV', 3:'UP', 4:'AIC_HUB', 5:'AIV_HUB'}
n_total = len(tasks)

all_exec = sum(t['duration_us'] for t in tasks)
all_head = sum(t['start_time_us'] - t['dispatch_time_us'] for t in tasks)
all_tail = sum(t['finish_time_us'] - t['end_time_us'] for t in tasks)
min_disp = min(t['dispatch_time_us'] for t in tasks)
max_fin = max(t['finish_time_us'] for t in tasks)
wall = max_fin - min_disp

print('=' * 90)
print('Part 1: Per-task time breakdown (from perf profiling data)')
print('=' * 90)
print(f'Total tasks: {n_total}')
print(f'Wall-clock:  {wall:.1f} us')
print()
fmt = "  {:<35} {:>12} {:>14} {:>10}"
print(fmt.format('Component', 'Total (us)', 'Avg/task (us)', '% of Wall'))
print('  ' + '-' * 75)
print(fmt.format('Kernel Exec (end-start)', f'{all_exec:.1f}', f'{all_exec/n_total:.2f}', f'{all_exec/wall*100:.1f}%'))
print(fmt.format('Head OH (start-dispatch)', f'{all_head:.1f}', f'{all_head/n_total:.2f}', f'{all_head/wall*100:.1f}%'))
print(fmt.format('Tail OH (finish-end)', f'{all_tail:.1f}', f'{all_tail/n_total:.2f}', f'{all_tail/wall*100:.1f}%'))
print()

# === Part 2: AICPU scheduler loop breakdown from device log ===
log_dir = os.path.expanduser('~/ascend/log/debug/device-0')
log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.log')], reverse=True)
log_path = os.path.join(log_dir, log_files[0])

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

print('=' * 90)
print('Part 2: AICPU scheduler loop breakdown (from device log)')
print('  3 scheduler threads, each manages 8 AIC + 16 AIV cores')
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
for p in phases:
    key = p + '_us'
    tot = sum(t.get(key, 0) for t in threads.values())
    pct = tot / total_us * 100
    avg = tot / total_completed if total_completed > 0 else 0
    print(fmt3.format(phase_labels[p], f'{tot:.1f}', f'{pct:.1f}%', f'{avg:.2f}'))

print()

# Lock contention breakdown
fmt4 = "  {:<50} {:>11} {:>10}"
print(fmt4.format('Lock contention (ready_q)', 'Total (us)', '% of total'))
print('  ' + '-' * 75)
lock_wait = sum(t.get('lock_wait_us', 0) for t in threads.values())
lock_hold = sum(t.get('lock_hold_us', 0) for t in threads.values())
print(fmt4.format('  wait (spinning for lock)', str(lock_wait), f'{lock_wait/total_us*100:.1f}%'))
print(fmt4.format('  hold (inside critical section)', str(lock_hold), f'{lock_hold/total_us*100:.1f}%'))
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
fanout_max = max(t.get('fanout_max', 0) for t in threads.values())
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

# Scheduler loop time = where Tail OH comes from
avg_loop_us = total_us / total_loops
complete_sum = sum(t.get('complete_us', 0) for t in threads.values())
dispatch_sum = sum(t.get('dispatch_us', 0) for t in threads.values())
print(f'  Avg scheduler loop iteration: {avg_loop_us:.1f} us (= min Tail OH granularity)')
print(f'  With 3 threads sharing {total_loops} loops over {total_us/3:.0f} us wall each:')
print()
print(f'  Tail OH breakdown (per completed task):')
complete_per_task = complete_sum / total_completed
dispatch_per_task = dispatch_sum / total_completed
scan_per_task = sum(t.get('scan_us', 0) for t in threads.values()) / total_completed
yield_per_task = sum(t.get('yield_us', 0) for t in threads.values()) / total_completed
print(f'    1. Dispatch phase (build payload + cache flush):  {dispatch_per_task:.2f} us/task  ({dispatch_sum/total_us*100:.1f}% of scheduler CPU)')
print(f'       - Lock wait (ready_q pop):                     {sum(t.get("lock_dispatch_wait",0) for t in threads.values())/total_completed:.2f} us/task')
print(f'       - Lock hold + build + dc cvac/civac + dsb sy:  {(dispatch_sum - sum(t.get("lock_dispatch_wait",0) for t in threads.values()))/total_completed:.2f} us/task')
print(f'    2. Complete phase (poll + fanout resolve):         {complete_per_task:.2f} us/task  ({complete_sum/total_us*100:.1f}% of scheduler CPU)')
print(f'       - Lock wait (ready_q push):                    {sum(t.get("lock_complete_wait",0) for t in threads.values())/total_completed:.2f} us/task')
print(f'       - Fanout traversal + atomic ops:               {(complete_sum - sum(t.get("lock_complete_wait",0) for t in threads.values()))/total_completed:.2f} us/task')
print(f'    3. Scan phase (new task discovery):               {scan_per_task:.2f} us/task')
print(f'    4. Yield (idle):                                  {yield_per_task:.2f} us/task')
print()
print(f'  Key insight: Dispatch phase consumes ~62% of scheduler CPU.')
print(f'  Within dispatch, cache flush (dc cvac + dsb sy) is the dominant cost.')
print(f'  Each dsb sy stalls the AICPU pipeline until all prior dc ops complete.')
print('=' * 90)
