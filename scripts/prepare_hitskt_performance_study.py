#!/usr/bin/env python3
"""CPU-only declaration builder. There is deliberately no train/execute option."""
import json
from ktbench.favor_plus_performance_study import prepare

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
