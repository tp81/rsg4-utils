import os
import traceback

from sys import argv

import tifffile
import numpy as np

# Expects a stack of ribbons, outputs the corrected stack of ribbons
def bkgcorr2(stk):
  m = np.median(stk,axis=0)
  p = np.median(m,axis=0)
  p = p/p[len(p)//2]
  stkc = np.stack([pl/p for pl in stk]).astype("uint16")

  return stkc

def processPref(pref,op):
	tifs = [os.path.join(pref,f) for f in sorted(os.listdir(pref)) if f.endswith(".tif")]
        
	stk = np.stack([tifffile.imread(t) for t in tifs])
	
	stkc = bkgcorr2(stk)

	for p,s in zip(tifs,stkc):
		if not os.path.isdir(op):
			os.makedirs(op)

		tifffile.imwrite(p.replace(pref,op), s)      
    
if len(argv)<2:
	error("Usage: remove_background <path_to_layer_directory>")

inpref,outpref = argv[1].split(",")

if os.path.isdir(inpref):
	print(f"Processing {inpref}")
	try:
		processPref(inpref,outpref)
	except Exception:
		print(f"Error processing {inpref}")
		print(traceback.format_exc())

