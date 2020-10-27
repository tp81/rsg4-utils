import tifffile
from rsg4 import Reconstruct

indir = '/data/tpengo/Natalie/Natalie_2803_20x/stitch_test/layer000/561/images'
outpath =  '/data/tpengo/Natalie/Natalie_2803_20x/stitch_test/layer000.tif'

m1 = Reconstruct(indir)

tifffile.imwrite(outpath, m1)
