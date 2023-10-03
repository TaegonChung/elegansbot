# ElegansBot
Newtonian Mechanics Model for C. elegans Locomotion  

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

## Tested Environment
- Windows 10
    - Python 3.8.18
    - NumPy 1.19.0
    - Numba 0.54.0
        - SciPy 1.5.0
    - Matplotlib 3.4.2

## Usage
1. Install library by `pip install elegansbot` or Download the `elegansbot.py` file into your project directory.
2. Use `from elegansbot import Worm` to import the library.
3. Refer to the detailed instructions in the docstring of the "Worm" class. Below is a brief overview of potential use-cases:
    - If you want to determine $\theta_{\mathrm{ctrl}}$ dynamically, it's advised to update "act" (equivalent to theta_ctrl) manually and then invoke the "steps" method on an instance of the "Worm" class.
    - If you wish to use ElegansBot with a pre-determined $\theta_{\mathrm{ctrl}}$ (kymogram), it's recommended to utilize the "run" method of an instance of the "Worm" class.

## Examples
You may want to check out examples in the "examples/" directory.  
Try running `python examples/example01.py` for dynamic input  
or `python examples/example02.py` for kymogram input.  

## Local Demo
Execute the following command:
```
python elegansbot.py
```

## File Description
- `elegansbot.py` : Main code
- `Video_S1_omega_turn.avi` : Video from the supplementary section of our paper.
- `Video_S2_delta_turn.avi` : Another video from the supplementary section of our paper.

## Citing ElegansBot
- If ElegansBot has been significant in your research, and you would like to acknowledge this work in your academic publication, please consider citing the following paper (this citation may be updated in the future):
```
Chung, T., Chang, I., & Kim, S. (2023). ElegansBot: Development of equation of motion deciphering locomotion including omega turns of Caenorhabditis elegans (p. 2023.09.26.559644). bioRxiv. https://doi.org/10.1101/2023.09.26.559644
```
