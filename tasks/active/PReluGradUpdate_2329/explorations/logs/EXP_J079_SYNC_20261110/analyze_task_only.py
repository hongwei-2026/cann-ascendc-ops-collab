#!/usr/bin/env python3
"""J079: reuse the frozen analyzer; only omit unavailable pipeline metrics."""
from pathlib import Path
import importlib.util, sys
p=Path('/mnt/workspace/2329-JadeYara-20261104-1812/J078/abc/control/campaign.py')
spec=importlib.util.spec_from_file_location('frozen_campaign', p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.METRICS=['Task Duration(us)']
m.analyze(*(Path(x) for x in sys.argv[1:]))
