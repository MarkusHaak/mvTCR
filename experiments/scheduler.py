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
    args = parser.parse_args()
    for exp in args.experiments:
        if not os.path.exists(exp):
            print(f"ERROR: experiment {exp} does not exist")
            exit(1)
    for model in args.models:
        if model not in ['poe', 'moe'] and args.tcr_embs:
            print(f"ERROR: model {model} currently not supported for use with --tcr_emb argument")
            exit(1)
    return args

def main():
    commands = []
    for exp in args.experiments:
        for model in args.models:
            for tcr_emb in args.tcr_embs:
                cmd_str = f"{PYTHONPATH} {exp} --model {model} --tcr_emb {tcr_emb}"
                commands.append(cmd_str)
            if args.no_tcr_emb:
                cmd_str = f"{PYTHONPATH} {exp} --model {model}"
                commands.append(cmd_str)
    print(f"{len(commands)} commands planned:")
    for cmd_str in commands:
        print(f"command '{cmd_str}'")
    print()
    
    failed = []
    successful = []
    for cmd_str in commands:
        print(f"INFO: Running command '{cmd_str}'")
        cmd = shlex.split(cmd_str)
        rc = run_cmd(cmd)
        if rc != 0:
            print(f"WARNING: command '{cmd_str}' stopped with non-zero return code {rc}")
            failed.append((cmd_str, rc))
        else:
            successful.append(cmd_str)
    
    print("##########################\nAll done. Failed commands:")
    for cmd_str, rc in failed:
        print(f"command '{cmd_str}' stopped with non-zero return code {rc}")
    print()
    print("Successful commands:")
    for cmd_str in successful:
        print(f"command '{cmd_str}' finished successfully")

def run_cmd(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b""):
        sys.stdout.buffer.write(c)
    time.sleep(2)
    process.poll()
    rc = process.returncode
    return rc

if __name__ == '__main__':
    PYTHONPATH = os.path.join(sys.exec_prefix, "bin", "python")
    args = parse_args()
    main()

