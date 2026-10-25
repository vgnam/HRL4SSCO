"""Training-free SA planner (open and closed loop, see kill2_rpp.py) on the exact Flickr test instance of
`RPP/Main.py 0 --train-dataset flickr-C --test-dataset flickr-C` (whole city, default drift), 100 rounds."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'RPP'))
import numpy as np
from real_data import instance_sampler
import kill2_rpp_lib as L

for city in sys.argv[1:]:
    inst = instance_sampler(f'flickr-{city}', drift='default')()
    for mode in ['open', 'closed']:
        vals = [L.episode(inst, inst[2], mode, s) for s in range(100)]
        print(f"flickr-{city} {mode}: {np.mean(vals):.2f} +- {np.std(vals) / 10:.2f}", flush=True)
