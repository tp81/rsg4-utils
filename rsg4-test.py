import tifffile
from rsg4 import ReconstructLayer
import logging

infile_zipped = 'https://s3.msi.umn.edu/tpengo-public/layer000.zip'
samplezipfile = 'sample.zip'

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO)

# Download the samples file and unzip
logging.info("Downloading sample file...")
import requests
r = requests.get(infile_zipped)
with open(samplezipfile,'wb') as of:
    of.write(r.content)
logging.info("Unzipping...")
import zipfile
with zipfile.ZipFile(samplezipfile, 'r') as zip_ref:
    zip_ref.extractall()


indir = './layer000/561/images'
outpath =  './layer000.tif'

logging.info("Reconstructing no GPU...")
m1 = ReconstructLayer(indir)

logging.info("Reconstructing with GPU...")
m1 = ReconstructLayer(indir)

logging.info(f"Saving to {outpath}")
tifffile.imwrite(outpath, m1)

logging.info("Reconstructing without BkgSub...")
m1 = ReconstructLayer(indir, do_bkgsub=False)

outpath2 = outpath.replace('.tif','_nobkgSub.tif')
logging.info(f"Saving to {outpath2}")
tifffile.imwrite(outpath2, m1)
