import os

#indir = '2318.3'
indir = # FULL PATH TO THE "stack1" DIRECTORY 
#channels = ['488','561']
channels = ['488','561','640']

outdir = indir+'_bkgCorr'

layers = [ p for p in os.listdir(indir) if p.startswith('layer')]

def getPrefix(layerdir,channel):
    return os.path.join(indir,layerdir,channel,'images')
#    return os.path.join(indir,layerdir)

prefs = [ getPrefix(l,c) for c in channels for l in layers]
#prefs = [ getPrefix(l) for l in layers ]

for p in prefs:
    print(p+","+p.replace(indir,outdir))

