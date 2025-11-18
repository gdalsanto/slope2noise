# slope2noise

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
