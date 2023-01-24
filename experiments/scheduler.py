import os
import sys
import argparse
import subprocess
import shlex
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=['10x_optuna.py'])
    parser.add_argument('--models', nargs='+', default=['moe'])
    parser.add_argument('--tcr_embs', nargs='*', default=[])
    parser.add_argument('--no_tcr_emb', action='store_true')
    parser.add_argument('--timeout', type=float, default=48., help='max optimization time in hours per optimization run.')
    parser.add_argument('--runs', type=int, default=1, help='Number of optimization runs of time <--timeout> to be conducted for each experiment.')
    parser.add_argument('--cont', action='store_true', help='continue optuna study if it already exist instead of overwriting it. If <--runs> is greater than 1, for the consecutive runs, the studies are always continued.')
    parser.add_argument('--filters', nargs='*', default=['all'])
    parser.add_argument('--donors', nargs='*', default=['1'])
    args = parser.parse_args()
    for exp in args.experiments:
        if not os.path.exists(exp):
            print(f"ERROR: experiment {exp} does not exist")
            exit(1)
        elif os.path.basename(exp) not in ['10x_optuna.py']:
            print(f"ERROR: experiment {exp} currently not supported")
            exit(1)
    for model in args.models:
        if model not in ['poe', 'moe'] and args.tcr_embs:
            print(f"ERROR: model {model} currently not supported for use with --tcr_emb argument")
            exit(1)
    return args

def main():
    commands = []
    for exp in args.experiments:
        for donor in args.donors:
            for model in args.models:
                for f in args.filters:
                    for tcr_emb in args.tcr_embs:
                        cmd_str = f"{PYTHONPATH} {exp} --model {model} --tcr_emb {tcr_emb} --timeout {args.timeout} --filter_non_binder {f} --donor {donor}"
                        commands.append(cmd_str)
                    if args.no_tcr_emb:
                        cmd_str = f"{PYTHONPATH} {exp} --model {model} --timeout {args.timeout} --filter_non_binder {f} --donor {donor}"
                        commands.append(cmd_str)
    print(f"{len(commands)} commands planned:")
    for cmd_str in commands:
        print(f"command '{cmd_str}'")
    print()
    print(f"These commands are run >>> {args.runs} <<< times each, in the order given above.")
    runtime = len(commands) * args.runs * args.timeout
    print(f"Estimated total runtime: {runtime:.1f} hours (= {runtime/24.:.1f} days)")
    
    eng_num = {1:'st', 2:'nd', 3:'rd'}
    failed = []
    successful = []
    for i in range(args.runs):
        for cmd_str in commands:
            if (i == 0 and args.cont) or i > 0:
                cmd_str += " --continue"
            print(f"INFO: Running command '{cmd_str}' the {i+1}{eng_num.get(i+1, 'th')} time")
            cmd = shlex.split(cmd_str)
            rc = run_cmd(cmd)
            if rc != 0:
                print(f"WARNING: command '{cmd_str}', execution {i+1} stopped with non-zero return code {rc}")
                failed.append((i, cmd_str, rc))
            else:
                successful.append((i, cmd_str))
    
    print("##########################\nAll done.\n")
    print("Successful commands:")
    for i, cmd_str in successful:
        print(f"command '{cmd_str}', execution {i+1} finished successfully")
    print()
    print("Failed commands:")
    for i, cmd_str, rc in failed:
        print(f"command '{cmd_str}', execution {i+1} stopped with non-zero return code {rc}")
    print()
    

def run_cmd(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b""):
        sys.stdout.buffer.write(c)
    time.sleep(3) # wait till process ended
    process.poll() # update return code
    rc = process.returncode
    return rc

if __name__ == '__main__':
    PYTHONPATH = os.path.join(sys.exec_prefix, "bin", "python")
    args = parse_args()
    main()

