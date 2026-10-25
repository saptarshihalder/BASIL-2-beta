"""Deprecated entry point, kept so old commands still work.

Runs the fixed comparison (PPO only / thinking only / adaptive, plus the same-rate
confidence and random gates) from evaluate_v2.py. See GUIDE.md.
"""
import sys

from evaluate_v2 import main

if __name__ == "__main__":
    main(sys.argv[1:])
