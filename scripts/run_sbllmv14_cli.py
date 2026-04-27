import argparse
import os
import sys

# Ensure repo root is on sys.path when executed as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from interfaces.chat_shell import ChatShell


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config_v14_terminal.json")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    shell = ChatShell(config_path=args.config, seed=args.seed)
    shell.run()


if __name__ == "__main__":
    main()
