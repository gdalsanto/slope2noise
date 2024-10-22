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

Requires DecayFitNet submodule to run Bayesian Decay Analysis so when cloning please add the `--recurse-submodules` flag 
