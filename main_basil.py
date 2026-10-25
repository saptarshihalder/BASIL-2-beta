"""Deprecated entry point, kept so old commands still work.

The old version always moved first, used the TicTacToe policy for Connect4 and a JSD
threshold above the reachable maximum. It now runs the fixed evaluation in evaluate_v2.py
with the adaptive mode only. See GUIDE.md.
"""
import sys

from evaluate_v2 import main

if __name__ == "__main__":
    main(["--modes", "adaptive"] + sys.argv[1:])
