import argparse
import sys
import json
import os
import zipfile
from .run_native_platform import run_platform
from .run_state import (
    find_latest_feedback_trace, 
    tail_jsonl, 
    load_memory_summary, 
    clear_memory,
    backup_file
)

def run_command(args):
    summary = run_platform(
        num_frames=args.frames,
        num_nodes=args.nodes,
        engine_steps_per_frame=args.engine_steps,
        feedback_enabled=not args.no_feedback
    )
    print("\nRun Summary:")
    print(json.dumps(summary, indent=2))

def scope_command(args):
    path = args.path or find_latest_feedback_trace()
    if not path:
        print("No feedback trace found.")
        return
    
    print(f"Tailing {path} (last {args.tail} frames):")
    records = tail_jsonl(path, args.tail)
    
    header = f"{'T':>4} | {'Hex':<25} | {'C':>4} | {'E':>4} | {'Ctn':>4} | {'Rec':>4} | {'Status':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for r in records:
        status = "IMPRINTED" if r.get("residue_committed") else "SKIPPED"
        print(f"{r['t']:4d} | {r['hex']:<25} | {r['C']:4.2f} | {r['E']:4.2f} | {r['caution']:4.2f} | {r['recovery']:4.2f} | {status}")

def memory_command(args):
    summary = load_memory_summary(args.path)
    print("\nResidue Memory Summary:")
    print(json.dumps(summary, indent=2))

def reset_memory_command(args):
    if clear_memory(args.path):
        print(f"Memory reset successful (backup created).")
    else:
        print(f"Failed to reset memory at {args.path}")

def export_run_command(args):
    import glob
    out_path = args.out or f"run_export_{args.run_id or 'latest'}.zip"
    
    with zipfile.ZipFile(out_path, 'w') as zipf:
        # Add logs
        log_files = glob.glob(os.path.join(args.logs_dir, f"*{args.run_id or ''}*.jsonl"))
        for f in log_files:
            zipf.write(f, os.path.join("logs", os.path.basename(f)))
            
        # Add sessions
        session_files = glob.glob(os.path.join("sessions", f"*{args.run_id or ''}*.*"))
        for f in session_files:
            zipf.write(f, os.path.join("sessions", os.path.basename(f)))
            
        # Add config and docs
        if os.path.exists("config"):
            for f in glob.glob("config/*.json"):
                zipf.write(f, os.path.join("config", os.path.basename(f)))
        if os.path.exists("docs"):
            for f in glob.glob("docs/*.md"):
                zipf.write(f, os.path.join("docs", os.path.basename(f)))
                
    print(f"Run exported to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Native Wave-Residue Platform CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Run
    run_p = subparsers.add_parser("run", help="Start a new analog field run")
    run_p.add_argument("--frames", type=int, default=100)
    run_p.add_argument("--nodes", type=int, default=100)
    run_p.add_argument("--engine-steps", type=int, default=20)
    run_p.add_argument("--no-feedback", action="store_true")

    # Scope
    scope_p = subparsers.add_parser("scope", help="Inspect latest trajectory")
    scope_p.add_argument("--tail", type=int, default=20)
    scope_p.add_argument("--path", type=str)

    # Memory
    mem_p = subparsers.add_parser("memory", help="View residue memory summary")
    mem_p.add_argument("--path", type=str, default="sessions/native_memory.json")

    # Reset Memory
    reset_p = subparsers.add_parser("reset-memory", help="Clear residue memory")
    reset_p.add_argument("--path", type=str, default="sessions/native_memory.json")

    # Export
    export_p = subparsers.add_parser("export-run", help="Bundle logs and state")
    export_p.add_argument("--run-id", type=str)
    export_p.add_argument("--logs-dir", type=str, default="logs")
    export_p.add_argument("--out", type=str)

    args = parser.parse_args()
    if args.command == "run":
        run_command(args)
    elif args.command == "scope":
        scope_command(args)
    elif args.command == "memory":
        memory_command(args)
    elif args.command == "reset-memory":
        reset_memory_command(args)
    elif args.command == "export-run":
        export_run_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
