# slope2noise

Everything is still under development. As of now only the broadband case (f_bands = None) using Gaussian noise shaping (synthesis_type = 'wgn') seems to kinda work (Bayesian Decay Analysis gives similar T values, but there's something off with the ampltidue). 

packages used 
- numpy 
- scipy
- ppyaml
- pydantic 
- torch 
- h5py
- soundfile
- loguru

For local installation: clone and install dependencies on a new pyton virtual environment `slope2noise` 
```
git clone https://github.com/gdalsanto/slope2noise
cd slope2noise
python3.10 -m venv .slope2noise-env
source .slope2noise-env/bin/activate
pip install -e .
```

Requires DecayFitNet submodule so when cloning please add the `--recurse-submodules` flag 
