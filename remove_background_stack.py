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

def processPref(stack_path,op):
	pref=os.path.split(stack_path)[0]
	
	stk = tifffile.imread(stack_path)
	
	stkc = bkgcorr2(stk)

	tifffile.imwrite(stack_path.replace(pref,op), stkc)      
    
if len(argv)<2:
	error("Usage: remove_background <path_to_stack>,<path_to_output_directory>")

inpath,outpref = argv[1].split(",")

if os.path.isfile(inpath):
	print(f"Processing {inpath}")
	try:
		processPref(inpath,outpref)
	except Exception:
		print(f"Error processing {inpath}")
		print(traceback.format_exc())

