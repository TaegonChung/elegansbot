from elegansbot import Worm, set_bbox_inches_tight
import numpy as np
import matplotlib.pyplot as plt

# kymogram
fps = 30
kymogram = np.zeros((fps*5, 24))
for i in range(kymogram.shape[0]):
    t = (1/fps) * i
    kymogram[i] = 0.5 * np.cos(2 * np.pi * (1.832 * np.linspace(0, 1, 24) - t / 1.6))

# kymogram figure
matrix = kymogram.transpose()
shape = np.shape(matrix)

fig, ax = plt.subplots(dpi=120)
im = ax.imshow(matrix, aspect=.3*(shape[1]/shape[0]), cmap='bwr')
ax.set_title('Kymogram')
ax.set_xlabel('frame number')
ax.set_ylabel('body position')
ax.set_yticks([0, shape[0]-1])
ax.set_yticklabels(['head', 'tail'])

[l, b, w, h] = ax.get_position().bounds
cbound = [l+w*1.05, b, w*0.05, h]
cax = fig.add_axes(cbound)
cbar = fig.colorbar(im, ax=ax, cax=cax)
cax.set_title('dorsal', fontsize=8)
cax.set_xlabel('ventral', fontsize=8)

set_bbox_inches_tight(fig)
plt.show()

# ElegansBot Simulation
env = Worm(scale_friction=0.01)
env.run(kymogram, 1/fps)

env.plot_overview()
env.plot_speed_graph()

env.play_animation(speed_playback=0.5)

## If you are using jupyter-notebook 6, use the following instead.
# %matplotlib notebook
# env.play_animation(speed_playback=0.5)
# %matplotlib inline

## If you are using jupyter-notebook 7, use the following instead.
# %matplotlib tk
# env.play_animation(speed_playback=0.5)
# %matplotlib inline

env.save_animation('demo.mp4') # FFMPEG is required for saving animation.
