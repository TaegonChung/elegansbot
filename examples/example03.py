from elegansbot import Worm
import numpy as np

fps = 32
kymogram = np.load('kymogram_omega-turn.npy')

env = Worm(scale_friction=0.01)
env.run(kymogram, 1/fps)
env.play_animation()