# ElegansBot

[![PyPI Version](https://img.shields.io/pypi/v/elegansbot.svg)](https://pypi.org/project/elegansbot/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/elegansbot.svg)](https://pypi.org/project/elegansbot/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/elegansbot.svg)](https://anaconda.org/conda-forge/elegansbot)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/elegansbot.svg)](https://anaconda.org/conda-forge/elegansbot)

Newtonian Mechanics Model for C. elegans Locomotion  

https://github.com/TaegonChung/elegansbot/assets/29942136/ca554471-4397-4e19-bbf3-e83e2ce5ed81

(Left: Experimental video from [1], Right: ElegansBot)
```
[1] Broekmans, O. D., Rodgers, J. B., Ryu, W. S., & Stephens, G. J. (2016). Resolving coiled shapes reveals new reorientation behaviors in C. elegans. ELife, 5, e17227. https://doi.org/10.7554/eLife.17227
```
## Web Demo
- https://taegonchung.github.io/elegansbot/
    - Use the sliders to observe the worm's movement changes.
        - Water-Agar Slider: Adjust this to modify the ground's frictional coefficients.
        - Swim-Crawl Slider: This slider alters the period and linear wave number of C. elegans' locomotion.

## Requirements
- Python (version 3)
- NumPy
- Numba
    - SciPy (required as a dependency of numba)
- Matplotlib  

**(Optional)**
- FFmpeg (for saving a video.)
- Jupyter Notebook (for interactive block coding)

## Tested Environment
Please, check "https://github.com/TaegonChung/ElegansBot/tested_environments.txt".  

## Usage
1. Install library by `pip install elegansbot` or `conda install conda-forge::elegansbot`.
2. Use `from elegansbot import Worm` to import the library.
3. Refer to the detailed instructions in the docstring of the "Worm" class. Below is a brief overview of potential use-cases:
    - If you want to determine $\theta_{\mathrm{ctrl}}$ dynamically, it's advised to update "act" (equivalent to theta_ctrl) manually and then invoke the "steps" method on an instance of the "Worm" class.
    - If you wish to use ElegansBot with a pre-determined $\theta_{\mathrm{ctrl}}$ (kymogram), it's recommended to utilize the "run" method of an instance of the "Worm" class.

## Examples
You may want to check out examples in the "examples/" directory.  
For instance, you can execute the example code with:  
`python examples/example01.py`

Files:
- examples/example01.py : Dynamic input  
- examples/example02.py : Kymogram input  
- examples/example03.py : Kymogram input (Omega-turn)
- examples/example04.py : Kymogram input (Delta-turn)
- examples/example05.py : Frictional forces, power, and behavioral classification

## Local Demo
Execute the following command:
```
python -m elegansbot.elegansbot
```

## File Description
- `elegansbot.py` : Main code
- `Video_S1_omega_turn.avi` : Video from the supplementary section of our paper.
- `Video_S2_delta_turn.avi` : Another video from the supplementary section of our paper.

## Citing ElegansBot
- If ElegansBot has been significant in your research, and you would like to acknowledge this work in your academic publication, we suggest citing [the following paper](https://elifesciences.org/articles/92562):
```
Taegon Chung, Iksoo Chang, Sangyeol Kim (2024) Development of equation of motion deciphering locomotion including omega turns of Caenorhabditis elegans eLife 12:RP92562 https://doi.org/10.7554/eLife.92562
```
