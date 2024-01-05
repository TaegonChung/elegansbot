# -*- coding: utf-8 -*-

# %%
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.patches as mpatches
from elegansbot import Worm, set_bbox_inches_tight


# %% [markdown]
# ## figure controller

# %%
# l, b, w, h for custom position definition
# _, _, W, H for definition in inches
# ll, bb, ww, hh for definition in relative values. (matplotlib position)

class FigureController:
    def __init__(self, fig, ax_main):
        """
        Call methods(add_axes, etc) of this instance after ax_main is fully drawn.
        Shorter side of main axes(ax_main) will be unit length (length=1).
        
        --- methods ---
        add_axes : add axes where you want
        add_axes_label : add axes a label, label will be 'A', 'B', 'C', ... and so on everytime you call this.
        
        --- property ---
        w : width of ax_main
        h : height of ax_main
        
        Axes control with this class should have super flat structure.
        If you try to make any kind of group object and build hierarchy with it, It will mess with your mind.
        """
        self.fig = fig
        self.ax_main = ax_main
        assert fig is ax_main.get_figure() # Be careful that fig must not be dropped by garbage collecter of Python.
        self.did_lazy_init = False
        
        def getter_l(self_ax):
            self.lazy_init()
            return self_ax.__l
        
        def getter_b(self_ax):
            self.lazy_init()
            return self_ax.__b
        
        def getter_w(self_ax):
            self.lazy_init()
            return self_ax.__w
        
        def getter_h(self_ax):
            self.lazy_init()
            return self_ax.__h
        
        def setter_l(self_ax, l):
            self_ax.__l = l
            
        def setter_b(self_ax, b):
            self_ax.__b = b
            
        def setter_w(self_ax, w):
            self_ax.__w = w
            
        def setter_h(self_ax, h):
            self_ax.__h = h
        
        class_name = self.ax_main.__class__.__name__ + 'Child'
        child_class = type(class_name, (self.ax_main.__class__,), {'l': property(getter_l, fset=setter_l)})
        self.ax_main.__class__ = child_class
        child_class = type(class_name, (self.ax_main.__class__,), {'b': property(getter_b, fset=setter_b)})
        self.ax_main.__class__ = child_class
        child_class = type(class_name, (self.ax_main.__class__,), {'w': property(getter_w, fset=setter_w)})
        self.ax_main.__class__ = child_class
        child_class = type(class_name, (self.ax_main.__class__,), {'h': property(getter_h, fset=setter_h)})
        self.ax_main.__class__ = child_class
        
        if 'figure_controller' not in dir(fig):
            fig.figure_controller = self
            
    def lazy_init(self):
        if self.did_lazy_init == False:
            self.ll, self.bb, self.ww, self.hh = self.ax_main.get_position().bounds
            self.W = self.ax_main.bbox.width
            self.H = self.ax_main.bbox.height
            self.S = min(self.W, self.H) # size
            self.r_w = self.ww * self.S / self.W
            self.r_h = self.hh * self.S / self.H

            self.__w = self.W/self.S
            self.__h = self.H/self.S
            self.__xy_ = {
                self.ax_main : {
                    'l' : 0,
                    'b' : 0,
                    'w' : self.__w,
                    'h' : self.__h
                },
            }
            self.ax_main.l = 0
            self.ax_main.b = 0
            self.ax_main.w = self.__w
            self.ax_main.h = self.__h

            self.i_label_ax_ = {
                self.ax_main : 0,
            }
            self.ax_ = [self.ax_main]
            self.label_count = 1
        
        self.did_lazy_init = True
    
    @property
    def w(self):
        if self.did_lazy_init == False:
            self.lazy_init()
        return self.__w
    
    @property
    def h(self):
        if self.did_lazy_init == False:
            self.lazy_init()
        return self.__h
    
    @property
    def xy_(self):
        if self.did_lazy_init == False:
            self.lazy_init()
        return self.__xy_
    
    
    def add_axes(self, l, b, w, h, ax_origin=None, should_label=False):
        """
        shorter side of main axes will be unit (length=1)
        """
        if self.did_lazy_init == False:
            self.lazy_init()
        if ax_origin is not None:
            l += self.__xy_[ax_origin]['l']
            b += self.__xy_[ax_origin]['b']

        ax = self.fig.add_axes(
            [
                self.ll + self.r_w * l, 
                self.bb + self.r_h * b, 
                self.r_w * w, 
                self.r_h * h,
            ]
        )
        self.__xy_[ax] = {
            'l' : l,
            'b' : b,
            'w' : w,
            'h' : h,
        }
        self.ax_.append(ax)
        
        ax.l = l
        ax.b = b
        ax.w = w
        ax.h = h
        
        if should_label == True:
            self.i_label_ax_[ax] = self.label_count
            self.label_count += 1    
        
        return ax
    
    def add_axes_label(
        self,
        ax,
        l,
        b,
        fontsize=24,
        ha='right',
        va='top',
        label=None,
    ):
        """
        ha : left / right
        va : baseline / top
        """
        if self.did_lazy_init == False:
            self.lazy_init()
        l0 = self.__xy_[ax]['l']
        b0 = self.__xy_[ax]['b']
        i_label = self.i_label_ax_[ax]
        ax_label = self.add_axes(l0+l, b0+b, .01, .01, should_label=False)
        if label is None:
            label = chr(65+i_label)
        ax_label.text(0, 0, label, fontsize=fontsize, ha=ha, va=va)
        ax_label.set_axis_off()
    
    def add_axes_label_all(
        self,
        l=0,
        b=0,
        fontsize=24,
        ha='right',
        va='top',
    ):
        """
        ha : left / right
        va : baseline / top
        """
        if self.did_lazy_init == False:
            self.lazy_init()
        for ax in self.i_label_ax_.keys():
            l0 = self.__xy_[ax]['l']
            t0 = self.__xy_[ax]['b'] + self.__xy_[ax]['h']
            i_label = self.i_label_ax_[ax]
            ax_label = self.add_axes(l0+l, t0+b, .01, .01, should_label=False)
            ax_label.text(0, 0, chr(65+i_label), fontsize=fontsize, ha=ha, va=va)
            ax_label.set_axis_off()
    
    
    @property
    def all_lbwh(self):
        if self.did_lazy_init == False:
            self.lazy_init()
        xy = self.__xy_[self.ax_[0]] 
        l = xy['l']
        b = xy['b']
        r = l + xy['w']
        t = b + xy['h']
        for i in range(1, len(self.ax_)):
            xy = self.__xy_[self.ax_[i]] 
            l2 = xy['l']
            b2 = xy['b']
            r2 = l + xy['w']
            t2 = b + xy['h']
            l = l2 if l2 < l else l
            b = b2 if b2 < b else b
            r = r2 if r2 > r else r
            t = t2 if t2 > t else t
        
        w = r - l
        h = t - b
        return l, b, w, h


# %% [markdown]
# ## else

# %%
def get_parameter_from_kymogram(theta_t_):
    assert len(theta_t_.shape) == 2
    n_t = theta_t_.shape[0]
    n_joint = theta_t_.shape[1]
    
    x = np.linspace(0, 1, n_joint)
    def func_sine(x, A, l, x_0, y_0):
        return A *np.sin((2 *np.pi /l) *(x -x_0)) +y_0
    
    # initial guess of first snapshot
    y = theta_t_[0]
    A = max(y.max(), -y.min()) /2
    l = 2 *np.abs(np.argmax(y) -np.argmin(y)) /(len(y)-1)
    x_0 = np.argmax(y) /(len(y)-1) -l /4
    y_0 = np.mean(y)

    # sine fitting of first snapshot
    (A, l, x_0, y_0), _ = curve_fit(func_sine, x, y, (A, l, x_0, y_0))

    if A < 0:
        A = -A
        x_0 -= l /2
    x_0 -= (x_0 //l) *l
    
    p__ = np.zeros(
        n_t,
        dtype=[
            ('A', np.float64),
            ('l', np.float64),
            ('x_0', np.float64),
            ('y_0', np.float64),
            ('x_0_tilde', np.float64),
        ],
    )
    
    # sine fitting of lasting snapshots
    p__[0] = A, l, x_0, y_0, x_0
    for i in range(1, theta_t_.shape[0]):
        y = theta_t_[i]
        
        (A, l, x_0, y_0), _ = curve_fit(func_sine, x, y, (A, l, x_0, y_0))
        
        if A < 0:
            A = -A
            x_0 -= l /2
        x_0 -= (x_0 //l) *l
        
        p__[i] = A, l, x_0, y_0, x_0
        
    for i in range(1, theta_t_.shape[0]):
        d_x_0 = p__['x_0'][i] - p__['x_0'][i-1]
        if d_x_0 > p__['l'][i] /2:
            p__['x_0_tilde'][i:] -= p__['l'][i]
        elif d_x_0 < -p__['l'][i] /2:
            p__['x_0_tilde'][i:] += p__['l'][i]
    
    return p__, func_sine

def continuous_range_(bool_):
    diff = np.diff(bool_.astype(int))
    start_index_ = np.where(diff == 1)[0] +1
    end_index_ = np.where(diff == -1)[0]
    if bool_[0] == True:
        start_index_ = np.insert(start_index_, 0, 0)
    if bool_[-1] == True:
        end_index_ = np.append(end_index_, len(bool_) -1)
    range_ = np.column_stack((start_index_, end_index_))
    return range_

def get_smoothed_parameter_from_kymogram(theta_t_, dt):
    p__raw_, func_sine = get_parameter_from_kymogram(theta_t_)
    theta_sine_t_ = np.zeros_like(theta_t_)
    for i in range(theta_t_.shape[0]):
        p_ = p__raw_[i]
        theta_sine_t_[i] = func_sine(np.linspace(0, 1, theta_t_.shape[-1]), p_['A'], p_['l'], p_['x_0'], p_['y_0'])

    p__ = np.zeros_like(p__raw_) # proportional
    d__ = np.zeros_like(p__raw_) # derivative
    for name in p__raw_.dtype.names:
        d__[name] = savgol_filter(p__raw_[name], window_length=int((1 /dt) /4) *2 +1, polyorder=2, deriv=1) # time window of 0.5 sec length
        p__[name] = np.cumsum(np.concatenate([[0], (d__[name][:-1] +d__[name][1:]) /2]))
        p__[name] += -p__[name].mean() +p__raw_[name].mean()
        d__[name] *= 1 /dt
        
    return p__raw_, func_sine, theta_sine_t_, p__, d__

    
def get_behavior_log_(p__, d__, y_0_turn=-.07):
    n_t = p__.shape[0]
    assert n_t == d__.shape[0]
    bool_turn_ = p__['y_0'] < y_0_turn
    bool_forward_ = np.logical_and(
        p__['y_0'] >= y_0_turn,
        d__['x_0_tilde'] > 0,
    )
    bool_backward_ = np.logical_and(
        p__['y_0'] >= y_0_turn,
        d__['x_0_tilde'] <= 0,
    )

    range_ = {
        'turn' : continuous_range_(bool_turn_),
        'forward' : continuous_range_(bool_forward_),
        'backward' : continuous_range_(bool_backward_),
    }
    
    ethogram_ = np.zeros(n_t, dtype='<U8')
    for key in range_.keys():
        rang = range_[key]
        for t_min, t_max in rang:
            ethogram_[t_min:t_max+1] = key

    t_window_ = continuous_range_(np.concatenate([
        [ethogram_[i] == ethogram_[i+1] for i in range(n_t-1)],
        [True],
    ]))

    behavior_log_ = [(t_window[0], ethogram_[t_window[0]]) for t_window in t_window_]
    return behavior_log_

def stream_get_behavior_log_(
    env,
    y_0_turn=-.07,
):
    _, _, _, p__, d__ = get_smoothed_parameter_from_kymogram(env.theta_log, dt=env.dt_snapshot)
    behavior_log_ = get_behavior_log_(p__, d__, y_0_turn=y_0_turn)
    return behavior_log_

def plot_ethogram_behavior_log(ax, behavior_log_, n_t):
    color_ = {
        'forward' : (0, 0, 1),
        'turn' : (0, .5, 0),
        'backward' : (1, 0, 0),
    }

    for i in range(len(behavior_log_)):
        i_snapshot, behavior = behavior_log_[i]
        if i == 0:
            i_snapshot += -.5 # -.5 for half of heatmap pixel size
            i_snapshot_2, _ = behavior_log_[i +1]
        if i < len(behavior_log_) -1:
            i_snapshot_2, _ = behavior_log_[i +1]
        else:
            i_snapshot_2 = n_t
        ax.axvspan(i_snapshot, i_snapshot_2, color=color_[behavior])
        if (i_snapshot_2-i_snapshot)/n_t > .1:
            ax.text(x=(i_snapshot+i_snapshot_2)/2, y=.5, s=behavior[0].upper()+behavior[1:], color=(1, 1, 1), ha='center', va='center', weight='bold')
        
    ax.set_yticks([])

def heatmap(ax, mat, vm, fixed_yxratio):
    fig = ax.get_figure()
    cmap = clrs.LinearSegmentedColormap.from_list(
        "",
        [[.0, (0, 0, 1)], [.5, (1, 1, 1)], [1., (1, 0, 0)]]
    )
    
    image = ax.imshow(
        mat.transpose(),
        cmap=cmap,
        aspect=mat.shape[0]/mat.shape[1]*fixed_yxratio,
        vmin=-vm,
        vmax=vm,
        interpolation='nearest',
    )
    
    h = mat.shape[1]
    w = mat.shape[0]
    ax.set_xlim([-0.02*w-0.5, w+0.02*w-0.5]) # 0.5 for half of pixel size
    ax.set_ylim([h+0.02*h-0.5, -0.02*h-0.5])
    
    l, b, w, h = ax.get_position().bounds
    cax = fig.add_axes([l+w*1.05, b, w*0.05, h])
    cbar = fig.colorbar(
        image,
        ax=ax,
        cax=cax,
    )
    
def plot_kymogram_parameter_n_behavior_log(ax, dt, theta_t_, theta_sine_t_, p__raw_, p__, d__, behavior_log_, y_0_turn=-.07, should_plot_diff=True, fixed_yxratio=.2):
    fig = ax.get_figure()
    
    n_t = theta_t_.shape[0]
    xticklabels = np.arange(0, (n_t-1)*dt, 5, dtype=int)
    xticks = np.round(xticklabels / dt)
    
    vm = max(
        abs(theta_t_.min()),
        abs(theta_t_.max()),
        abs(theta_sine_t_.min()),
        abs(theta_sine_t_.max()),
    )

    heatmap(ax, theta_t_, vm, fixed_yxratio)
    ax.set_ylabel(r'$\theta_{i}^{(t)}$', rotation=0, ha='right', va='center')
    ax.yaxis.set_label_coords(-.11, .5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_yticks([0, theta_t_.shape[-1]-1])
    ax.set_yticklabels(['Head', 'Tail'])

    l, b, w, h = ax.get_position().bounds
    ax = fig.add_axes([l, b -1.1 *h, w, h])
    heatmap(ax, theta_sine_t_, vm, fixed_yxratio)
    ax.set_ylabel(r'$\hat{\theta}_{i}^{(t)}$', rotation=0, ha='right', va='center')
    ax.yaxis.set_label_coords(-.11, .5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_yticks([0, theta_t_.shape[-1]-1])
    ax.set_yticklabels(['Head', 'Tail'])

    display_name_ = {
        'A' : r'$A^{(t)}$',
        'l' : r'$\lambda^{(t)}$',
        'x_0' : r'$\xi_{0}^{(t)}$',
        'y_0' : r'$\theta_{0}^{(t)}$',
        'x_0_tilde' : r'$\tilde{\xi}_{0}^{(t)}$',
    }
    for name in p__.dtype.names:
        l, b, w, h = ax.get_position().bounds
        ax = fig.add_axes([l, b -1.1 *h, w, h])
        ax.plot(p__raw_[name], linewidth=.5, color=(0, 0, 0))
        if name not in ['A', 'l', 'x_0']:
            ax.plot(p__[name], linewidth=.5, color=(1, 0, 0))
        ax.set_box_aspect(.2)
        ax.set_xlim(-.02 *n_t, 1.02 *n_t)
        ax.set_ylabel(display_name_[name])
        ax.yaxis.set_label_coords(-.11, .5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
        if name == 'y_0':
            ax.axhline(y=0, linewidth=.5, linestyle='--', color=(0, 0, 0))
            ax.axhline(y=y_0_turn, linewidth=.5, linestyle='--', color=(0, 0, 1))

    name = 'x_0_tilde'
    l, b, w, h = ax.get_position().bounds
    ax = fig.add_axes([l, b -1.1 *h, w, h])
    ax.axhline(y=0, linewidth=.5, color=(0, 0, 0), linestyle='--')
    ax.plot(np.diff(p__raw_[name]) *(1 /dt), linewidth=.5, color=(0, 0, 0))
    ax.plot(d__[name], linewidth=.5, color=(1, 0, 0))
    ax.set_box_aspect(.2)
    ax.set_xlim(-.02 *n_t, 1.02 *n_t)
    ax.set_ylim(-2, 2)
    ax.set_ylabel(r"$\frac{\mathrm{d} \tilde{\xi}_0^{(t)}}{\mathrm{d} t}$", fontsize=14)
    ax.yaxis.set_label_coords(-.11, .5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([])

    l, b, w, h = ax.get_position().bounds
    xlim = ax.get_xlim()
    ax = fig.add_axes([l, b -(.1 +.5) *h, w, .5 *h])
    ax.set_xlim(xlim)
    plot_ethogram_behavior_log(ax, behavior_log_, n_t)

    ax.set_yticks([])
    ax.set_ylabel('Behavior', rotation=0, ha='right', va='center')
    ax.yaxis.set_label_coords(-.08, .5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('t (sec)')


# %%
def plot_color_wheel(
    ax,
    n_phi=32,
    n_rho=10,
    title='',
    y_title=-.4,
    yticklabels=(.5, 1.0),
    should_quick_draw=False,
):
    if should_quick_draw == True:
        n_phi = 8
        n_rho = 4
        
    fig = ax.get_figure()
    
    phi_ = np.linspace(0, 2*np.pi, n_phi+1)[:-1]
    rho_ = np.linspace(0, 1, n_rho)[1:]
    dphi = np.diff(phi_).mean()
    drho = np.diff(rho_).mean()

    bounds = ax.get_position().bounds
    fig.delaxes(ax)
    ax = fig.add_axes(bounds, projection='polar')
    for phi in phi_:
        for rho in rho_:
            rgb = clrs.hsv_to_rgb((phi/(2*np.pi), 3*rho**2-2*rho**3, 1))
            artist = mpatches.Rectangle((phi-dphi/2, rho-drho/2), dphi, drho, facecolor=rgb, edgecolor='none')
            ax.add_artist(artist)
    ax.bar(0, 1).remove()
    ax.set_title(title, y=y_title)
    ax.set_rlabel_position(135)
    ax.set_rticks([.5, 1])
    ax.set_yticklabels(yticklabels, va='center')


# %%
def plot_process_behavior_classification(
    env,
    dt=1/32,
    y_0_turn=-.07,
):
    fig, ax = plt.subplots(figsize=(5.5, 2))
    ctrl = FigureController(fig, ax)
    theta_t_ = env.theta_log
    p__raw_, func_sine, theta_sine_t_, p__, d__ = get_smoothed_parameter_from_kymogram(theta_t_, dt=dt)
    behavior_log_ = get_behavior_log_(p__, d__, y_0_turn=y_0_turn)
    plot_kymogram_parameter_n_behavior_log(ax, dt, theta_t_, theta_sine_t_, p__raw_, p__, d__, behavior_log_, y_0_turn, should_plot_diff=False)
    set_bbox_inches_tight(fig)


# %%
def plot_heatmap_rod_force(env, f_s_max=None, i_snapshot_arrow_=None, fixed_yxratio=.2, y_title=-.4, should_quick_draw=False, verbose=False, dpi=72, figsize=(7, 1.5)):
    """
    set should_quick_draw=True for faster drawing with simpler color wheel
    """
    
    fig, ax_main = plt.subplots(dpi=dpi, figsize=figsize)
    ctrl = FigureController(fig, ax_main)
    
    P_ = np.zeros((env.n_snapshot))
    f_xy_ = np.zeros((env.n_snapshot, 2, env.n))
    f_ = np.zeros((env.n_snapshot, env.n))
    rgb_f_ = np.zeros((env.n, env.n_snapshot, 3))

    n = env.n
    t_log = env.t_log
    xy_tip = env.xy_tip_log
    s_log = env.s_log
    bII = env.bII
    bT = env.bT

    dt = np.diff(t_log).mean()

    for i in range(env.n_snapshot):
        vx_tip = (xy_tip[i, 0, :] - xy_tip[max(0, i-1), 0, :]) / dt
        vy_tip = (xy_tip[i, 1, :] - xy_tip[max(0, i-1), 1, :]) / dt
        vx = (vx_tip[:-1] + vx_tip[1:]) / 2
        vy = (vy_tip[:-1] + vy_tip[1:]) / 2
        cos_s = np.cos(s_log[i, :])
        sin_s = np.sin(s_log[i, :])

        vII = cos_s*vx + sin_s*vy  # parallel(to i-rod) component of velocity
        vT = -sin_s*vx + cos_s*vy  # perpendicular component
        fII, fT = -(bII/n)*vII, -(bT/n)*vT
        fx = cos_s*fII - sin_s*fT
        fy = sin_s*fII + cos_s*fT
        f_xy_[i, 0, :] = fx
        f_xy_[i, 1, :] = fy

        P_[i] = -((fx*vx).sum() + (fy*vy).sum())
        
        f = np.sqrt(fx**2 + fy**2)
        
        f_[i, :] = f
        
    var_ = {
        'P_' : P_,
        'f_xy_' : f_xy_,
        'f_' : f_,
    }

    f_mean_ = f_.mean(axis=-1)
    
    if f_s_max is None:
        f_s_max = (f_.mean()+2.56*f_.std() + np.quantile(f_.max(axis=-1), .75))/2
        f_s_max = float(f"{f_s_max:.1g}")
        
    for i in range(env.n_snapshot):
        fx = f_xy_[i, 0, :]
        fy = f_xy_[i, 1, :]

        h = np.arctan2(fy, fx) / (2*np.pi) # force acting on body by ground
        h[h<0] += 1
        s = np.clip(np.sqrt(fx**2 +fy**2)/f_s_max, 0, 1)

        for j in range(env.n):
            rgb_f_[j, i, :] = clrs.hsv_to_rgb((h[j], s[j], 1))
            
    P_behavior_ = dict()
    behavior_log_ = stream_get_behavior_log_(env)
    for i in range(len(behavior_log_)):
        i_snapshot, behavior = behavior_log_[i]
        if i < len(behavior_log_) -1:
            i_snapshot_2, _ = behavior_log_[i +1]
            P_behavior_[(i, behavior)] = P_[i_snapshot:i_snapshot_2].mean()
        else:
            P_behavior_[(i, behavior)] = P_[i_snapshot:].mean()
    
    if verbose == True:
        print(f"Power : ", end='')
        for key in P_behavior_.keys():
            print(f"({key}, {P_behavior_[key]:.0f} fW), ", end='')
        print()
    
    xticklabels = np.arange(0, (env.n_snapshot-1)*env.dt_snapshot, 5, dtype=int)
    xticks = np.round(xticklabels / env.dt_snapshot)

    ax = ax_main
    ax.imshow(
        rgb_f_,
        aspect=rgb_f_.shape[1]/rgb_f_.shape[0]*fixed_yxratio,
        interpolation='nearest',
    )
    ax.set_ylabel(r'$\mathbf{F}_{b,i}^{(t)}$', color=(0, 0, 0), rotation=0, va='center', ha='right')
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_yticks([0, env.n-1])
    ax.tick_params(axis='y')
    ax.set_yticklabels(['Head', 'Tail'])

    ax_to_add_arrow_ = [ax]
    
    tmp = 0
    ax = ctrl.add_axes(ctrl.w +.4+tmp, 0+tmp, 1-2*tmp, 1-2*tmp, ax_origin=ax_main)
    plot_color_wheel(
        ax,
        yticklabels=(
            rf'${{{f_s_max/2:.0f}}} \mathrm{{pN}}$' if np.log10(f_s_max/2)>=0 else rf'${{{f_s_max/2}}} \mathrm{{pN}}$',
            rf'${{{f_s_max:.0f}}} \mathrm{{pN}}$' if np.log10(f_s_max/2)>=0 else rf'${{{f_s_max}}} \mathrm{{pN}}$',
        ),
        y_title=y_title,
        should_quick_draw=should_quick_draw,
    )
    
    ax = ctrl.add_axes(0, -1.35*ctrl.h, ctrl.w, ctrl.h, ax_origin=ax_main)
    ax_to_add_arrow_.append(ax)
    cmap = clrs.LinearSegmentedColormap.from_list(
        "",
        [[.0, (1, 1, 1)], [.2, (0, 0, 1)], [.6, (0, 1, 0)], [.8, (1, 1, 0)], [1., (1, 0, 0)]]
    )

    image = ax.imshow(
        f_.transpose(),
        cmap=cmap,
        aspect=f_.shape[0]/f_.shape[1]*fixed_yxratio,
        vmin=0,
        vmax=f_s_max,
        interpolation='nearest',
    )
    
    ax.set_ylabel(r'$|\mathbf{F}_{b,i}^{(t)}|$', color=(0, 0, 0), rotation=0, va='center', ha='right')
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_yticks([0, env.n-1])
    ax.tick_params(axis='y')
    ax.set_yticklabels(['Head', 'Tail'])

    cax = ctrl.add_axes(ctrl.xy_[ax]['w'] +.1, 0, .2, 1, ax_origin=ax)
    cbar = fig.colorbar(
        image,
        ax=ax,
        cax=cax,
    )
    cax.set_title(r'$(\mathrm{pN})$', fontsize=10)    

    ax_keep = ax    
    ax = ctrl.add_axes(0, -1.35*ctrl.h, ctrl.w, ctrl.h, ax_origin=ax_keep)
    ax_to_add_arrow_.append(ax)
    ax.plot(f_mean_, color=(0, 0, 0))
    ax.set_ylabel(r'$\left< |\mathbf{F}_{b,i}^{(t)}| \right>_{i}$'+f' '+r'$(\mathrm{pN})$', color=(0, 0, 0))
    ax.set_xlim(ax_main.get_xlim())
    ax.set_xticks(xticks)
    ax.set_xticklabels([])

    ax2 = ctrl.add_axes(0, -1.35*ctrl.h, ctrl.w, ctrl.h, ax_origin=ax_keep)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.plot(P_, '--', color=(1, 0, 0), label='P')
    ax2.set_ylabel(f'Power\n'+r'($\mathrm{fW}$)', color=(1, 0, 0), rotation=270, va='bottom')
    ax2.tick_params(axis='y', colors=(1, 0, 0))
    ax2.set_xlim(ax_main.get_xlim())
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([])
    ax2.set_facecolor('none')

    xlim = ax.get_xlim()
    ax = ctrl.add_axes(0, -(.35 +.3) *ctrl.h, ctrl.w, .3*ctrl.h, ax_origin=ax)
    ax_bottom = ax
    ax_to_add_arrow_.append(ax)
    plot_ethogram_behavior_log(ax, behavior_log_, env.n_snapshot)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('Behavior', rotation=0, ha='right', va='center', labelpad=26.5)
        
    for ax_to_add_arrow in ax_to_add_arrow_:
        if i_snapshot_arrow_ is not None:
            r = .025
            dx, _ = f_.shape
            
            ax = ctrl.add_axes(
                ctrl.w *(-r /2),
                ax_to_add_arrow.h +.003*ctrl.w,
                ctrl.w *(1 +r),
                ctrl.w *(r),
                ax_origin=ax_to_add_arrow,
            )
            ax.set_xlim(-(r /2) *dx -.5, (1 +r /2) *dx -.5) # -.5 for half of pixel size of heatmap
            ax.set_ylim(0, 1)
            
            ax.set_facecolor('none')
            ax.set_axis_off()

            xy_arrow = np.array([[0, 0], [-r *dx /2, 1], [r *dx /2, 1]])
            for i_snapshot in i_snapshot_arrow_:
                x_point = i_snapshot
                xy_point = (x_point, 0)
                patch = mpatches.Polygon(xy=xy_arrow +xy_point, alpha=.8, facecolor=(0, 0, 0))
                ax.add_patch(patch)
    
    set_bbox_inches_tight(fig)

    return ax_bottom.l, ax_bottom.b, var_



# %% [markdown]
# # main

# %%
try:
    env = Worm()
    env.load('env_omega-turn')
except:
    env = Worm(scale_friction=.01, angle_init=2*np.pi*(.43))
    theta_t_ = np.load('kymogram_omega-turn.npy')
    env.run(theta_t_, dt=1/32)
    env.save('env_omega-turn')

env.plot_overview()

plot_heatmap_rod_force(env, should_quick_draw=True)
plt.show()

plot_process_behavior_classification(env)
plt.show()

# %%
try:
    env = Worm()
    env.load('env_delta-turn')
except:
    env = Worm(scale_friction=.01, angle_init=2*np.pi*(-.015), polarity_clockwise=True)
    theta_t_ = np.load('kymogram_delta-turn.npy')
    env.run(theta_t_, dt=1/32)
    env.save('env_delta-turn')

env.plot_overview()

plot_heatmap_rod_force(env, should_quick_draw=True)
plt.show()

plot_process_behavior_classification(env)
plt.show()

# %%

# %%

# %%
