from elegansbot import Worm
import numpy as np

fps = 32
kymogram = np.load('kymogram_delta-turn.npy')

env = Worm(
    simTime=(kymogram.shape[0]-1)/fps,
    dt_snapshot=1/fps,
    scale_friction=0.01,
    polarity_clockwise=True,
)

env.run(kymogram, 1/fps)
env.play_animation()