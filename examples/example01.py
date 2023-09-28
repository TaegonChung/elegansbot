from elegansbot import Worm
import numpy as np

print('Agar plate')
env = Worm(dt=0.00005) # Numba JIT spends compile-time here, once.
for _ in range(env.n_snapshot):
    env.act = 0.7 * np.cos(2 * np.pi * (1.832 * np.linspace(0, 1, env.n - 1) - env.t / 1.6))
    env.steps()
env.plot_overview()
env.plot_speed_graph()
env.play_animation()
# env.save_animation('demo_crawl.mp4')

print('Water')
env = Worm(dt=0.00005, bT=5.2e3, bII=5.2e3/1.5) # You can calculate faster with less accurate result by increasing dt.
for _ in range(env.n_snapshot):
    env.act = 0.5 * np.cos(2 * np.pi * (0.667 * np.linspace(0, 1, env.n - 1) - env.t / 0.4))
    env.steps()
env.plot_overview()
env.plot_speed_graph()
env.play_animation()
# env.save_animation('demo_swim.mp4')
