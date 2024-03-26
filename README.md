# Utilities for the RSG4 ribbon scanning confocal

This assumes running at the Minnesota Supercomputing Institute

## Stitching

TODO

## Background removal

Create a list of directories to process. Edit the `list_prefixes.py` file.

```
module load python3/3.10.9_anaconda2023.03_libmamba
python list_prefixes.py >pref.txt
```

Run in parallel

```
module load parallel
<pref.txt parallel bar jobs 32 python remove_background.py
```



