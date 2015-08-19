#! /usr/bin/env python

import json, sys
from utils import Bunch

# get parameters from config file
fname = sys.argv[1]
f = open(fname, 'r')
params = Bunch(json.loads(f.read()))
