# -*- coding: utf-8 -*-
# %%
import time
import numpy as np
from numba import njit
from numba import int64, float64, boolean
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from matplotlib.transforms import Bbox

# %%
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False

def set_bbox_inches_tight(fig, ratio_margin=0.01):
    assert 0 <= ratio_margin <= 1
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = bbox.width, bbox.height # inches

    bb = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in fig.axes]
    tight_bbox_raw = Bbox.union(bb)
    tight_bbox = fig.transFigure.inverted().transform_bbox(tight_bbox_raw)

    l, b, w, h = tight_bbox.bounds # 0~1 value (relative)
    margin = ratio_margin * max(width, height)
    w_m = margin / width
    h_m = margin / height
    l -= w_m
    b -= h_m
    w += w_m * 2
    h += h_m * 2
    for ax in fig.axes:
        l2, b2, w2, h2 = ax.get_position().bounds
        r2, t2 = l2+w2, b2+h2
        l3 = (l2 - l) / w
        r3 = (r2 - l) / w
        b3 = (b2 - b) / h
        t3 = (t2 - b) / h
        w3 = r3 - l3
        h3 = t3 - b3
        ax.set_position((l3, b3, w3, h3))

    fig.set_size_inches((width+2*margin, height+2*margin))

def plot_outline_n_trajectory(
    ax, 
    i, 
    env,
    draw=False, # whether to run fig.canvas.draw()
    show_legend=True,
    init_fig=False,
    dpi=None,
):
    """
    Drawing single snapshot image
    Body outline + head/tail track
    """
    
    if init_fig == True:
        xy_tip_log = env.xy_tip_log

        width = xy_tip_log[:, 0, :].max() - xy_tip_log[:, 0, :].min() + 1
        height = xy_tip_log[:, 1, :].max() - xy_tip_log[:, 1, :].min() + 1
        yx_ratio = height / width
        width = 3 / np.sqrt(yx_ratio) + 1
        width = max(width, 4.2)
        height = 3 * np.sqrt(yx_ratio)

        fig, ax = plt.subplots(dpi=dpi, figsize=(width, height))

        return fig, ax

    t_log = env.t_log
    xy_tip_log = env.xy_tip_log
    
    x_tip = xy_tip_log[i, 0, :]
    y_tip = xy_tip_log[i, 1, :]
    track_head = xy_tip_log[:i, :, 0]
    track_tail = xy_tip_log[:i, :, -1]
    time = t_log[i]
    
    fig = ax.get_figure()
    
    n = env.n
    thickness = (env.L/20)*np.sin(np.arccos(np.linspace(-.98, .98, n+1)))

    atan2 = np.arctan2(np.diff(y_tip), np.diff(x_tip))
    thetaDiff = np.diff(atan2)
    thetaDiff[thetaDiff < -np.pi] += 2*np.pi
    thetaDiff[thetaDiff > np.pi] -= 2*np.pi
    
    theta = np.cumsum(np.concatenate([[atan2[0]], thetaDiff]))

    midang = theta[:-1]+theta[1:]
    difang = np.abs(theta[:-1]-theta[1:]) < np.pi
    midang = midang*difang + (midang+2*np.pi)*np.logical_not(difang)
    angle = np.concatenate([[theta[0]], midang/2, [theta[-1]]])

    sign = 1 if env.polarity_clockwise == True else -1
    dx, dy = -thickness*np.sin(angle)*sign, thickness*np.cos(angle)*sign
    x_dorsal, y_dorsal = x_tip+dx, y_tip+dy
    x_ventral, y_ventral = x_tip-dx, y_tip-dy

    if ax.lines:
        ax.lines[0].set_data(x_tip[0], y_tip[0])
        ax.lines[1].set_data(x_dorsal, y_dorsal)
        ax.lines[2].set_data(x_ventral, y_ventral)
        ax.lines[3].set_data([x_dorsal[0], x_ventral[0]], [y_dorsal[0], y_ventral[0]])
        ax.lines[4].set_data([x_dorsal[-1], x_ventral[-1]], [y_dorsal[-1], y_ventral[-1]])
        ax.lines[5].set_data(track_head[:, 0], track_head[:, 1])
        ax.lines[6].set_data(track_tail[:, 0], track_tail[:, 1])
        ax.texts[0].set_text(f'Time: {time:.3f} (sec)')
        
        if draw == True:
            fig.canvas.draw()
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        left = np.min(xy_tip_log[:, 0, :], axis=None) - 0.5
        right = np.max(xy_tip_log[:, 0, :], axis=None) + 0.5
        top = np.max(xy_tip_log[:, 1, :], axis=None) + 0.5
        bottom = np.min(xy_tip_log[:, 1, :], axis=None) - 0.5
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect(1)
        
        lw = 1
        ax.plot(x_tip[0], y_tip[0], 'o', color=[1, 1, 0], label='Head position', linewidth=lw, markersize=lw*10)
        ax.plot(x_dorsal, y_dorsal, '-', color=[.8, 0, 0], label='Dorsal', linewidth=lw)
        ax.plot(x_ventral, y_ventral, '-', color=[0, 0, 1], label="Ventral", linewidth=lw)
        ax.plot([x_dorsal[0], x_ventral[0]], [y_dorsal[0], y_ventral[0]], '-',
                color=[.5, 0, .5], linewidth=lw)
        ax.plot([x_dorsal[-1], x_ventral[-1]], [y_dorsal[-1], y_ventral[-1]],
                '-', color=[.5, 0, .5], linewidth=lw)
        ax.plot(track_head[:, 0], track_head[:, 1], '-',
                color=[1, .5, 0], linewidth=lw/2, label="Head track")
        ax.plot(track_tail[:, 0], track_tail[:, 1], '-',
                color=[0, .5, 1], linewidth=lw/2, label="Tail track")
        ax.text(left+0.1, bottom+0.1, f'Time: {time:.3f} (sec)')
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        if show_legend == True:
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

    return ax.lines + ax.texts


# %%
@njit(fastmath=True, cache=True) # enables all fast-math flags of llvm
def cholSolver_banded(A, b):
    """
    Matrix equation solving function (Cholesky decomposition)
    This function solves Ax = b, where A is Hermitian matrix.
    (x20 faster than np.linalg.solve and np.linalg.inv)
    If you want to test the accuracy of this function,
    compare the results of this function and np.linalg.solve
    by using D matrix from 'step_numba' function as input matrix A
    and a random vector as input vector b.
    """
    n = A.shape[1]//2
    A[0, 0] = np.sqrt(A[0, 0])
    A[1, 0] /= A[0, 0]
    A[0, 1] = np.sqrt(A[0, 1]-A[1, 0]**2)
    # Cholesky decomposition
    for i in range(n-1):
        k = 2*i
        A[2, 0+k] /= A[0, 0+k]
        A[1, 1+k] = (A[1, 1+k] - A[2, 0+k]*A[1, 0+k])/A[0, 1+k]
        A[0, 2+k] = np.sqrt(A[0, 2+k] - A[2, 0+k]**2 - A[1, 1+k]**2)

        A[3, 0+k] /= A[0, 0+k]
        A[2, 1+k] = (A[2, 1+k] - A[3, 0+k]*A[1, 0+k])/A[0, 1+k]
        A[1, 2+k] = (A[1, 2+k] - A[3, 0+k]*A[2, 0+k] -
                     A[2, 1+k]*A[1, 1+k])/A[0, 2+k]
        A[0, 3+k] = np.sqrt(A[0, 3+k] - A[3, 0+k]**2 -
                            A[2, 1+k]**2 - A[1, 2+k]**2)

    # Down (solving y for (b = Ax = LL^Tx = Ly)).
    b[0] /= A[0, 0]
    b[1] = (b[1] - A[1, 0]*b[0]) / A[0, 1]
    for i in range(n-1):
        k = 2*i
        b[2+k] = (b[2+k] - A[2, 0+k]*b[0+k] - A[1, 1+k]*b[1+k]) / A[0, 2+k]
        b[3+k] = (b[3+k] - A[3, 0+k]*b[0+k] - A[2, 1+k]
                  * b[1+k] - A[1, 2+k]*b[2+k]) / A[0, 3+k]

    # Up (solving x for (L^Tx = y)). Be careful of transpose.
    b[-1] /= A[0, -1]
    b[-2] = (b[-2] - A[1, -2]*b[-1]) / A[0, -2]
    for i in range(n-1):
        k = 2*i
        b[-3-k] = (b[-3-k] - A[2, -3-k]*b[-1-k] -
                   A[1, -3-k]*b[-2-k]) / A[0, -3-k]
        b[-4-k] = (b[-4-k] - A[3, -4-k]*b[-1-k] - A[2, -4-k]
                   * b[-2-k] - A[1, -4-k]*b[-3-k]) / A[0, -4-k]

    return b


# %%
@njit(fastmath=True, cache=True)
def step_numba(
    r, xc, yc, vxc, vyc, s, omega,
    x, y, vx, vy, bII, bT, n, F_b, tau_b,
    F_ck, tau, k, theta_ctrl, tau_c, tau_k, tau_ck, c,
    D, ri, rii, Ni, Nii, Q, m, F, M, tau_joint,
    dt, I,
    I_P_inv,
):
    s -= ((s[0]+2*np.pi)//(4*np.pi)) * (2*np.pi) # preventing overflow
    theta = s[1:] - s[:-1]

    ## calculation
    cos_s = np.cos(s)
    sin_s = np.sin(s)
    # friction
    vx[0] = 0
    vy[0] = 0
    for j in range(n-1):
        vx[j+1] = vx[j] -r*omega[j]*sin_s[j] -r*omega[j+1]*sin_s[j+1]
        vy[j+1] = vy[j] +r*omega[j]*cos_s[j] +r*omega[j+1]*cos_s[j+1]
    vx[:] += -vx.mean() +vxc
    vy[:] += -vy.mean() +vyc

    vII = cos_s*vx + sin_s*vy  # parallel component of velocity
    vT = -sin_s*vx + cos_s*vy  # perpendicular component
    fII, fT = -(bII/n)*vII, -(bT/n)*vT
    fx = cos_s*fII - sin_s*fT
    fy = sin_s*fII + cos_s*fT
    F_b[0, :] = fx
    F_b[1, :] = fy
    tau_b[:] = -(bT*r*r/3/n)*omega[:]

    # muscle (damped tortion spring)
    tau[:] = c*(omega[1:]-omega[:-1])
    tau_c[:] = 0
    tau_c[:-1] += tau[:]
    tau_c[1:] -= tau[:]

    tau[:] = k*(theta-theta_ctrl)
    tau_k[:] = 0
    tau_k[:-1] += tau[:]
    tau_k[1:] -= tau[:]

    tau_ck[:] = tau_c + tau_k

    tau[:] = c*(omega[1:]-omega[:-1]) +k*(theta-theta_ctrl)
    tmp = tau * np.sin(theta/2) /(r*np.cos(theta/2)**2)
    phi = (s[1:] + s[:-1])/2
    tmp_cos_phi = tmp * np.cos(phi)
    tmp_sin_phi = tmp * np.sin(phi)
    F_ck[:, :] = 0
    F_ck[0, :-1] -= tmp_cos_phi
    F_ck[1, :-1] -= tmp_sin_phi
    F_ck[0, 1:] += tmp_cos_phi
    F_ck[1, 1:] += tmp_sin_phi

    # Chain dynamics
    D[:, :] = 0
    s_i, c_i = sin_s[0], cos_s[0]
    ri[0] = c_i
    ri[1] = s_i
    Ni[0] = -s_i
    Ni[1] = c_i

    s_ii, c_ii = sin_s[1], cos_s[1]
    rii[0] = c_ii
    rii[1] = s_ii
    Nii[0] = -s_ii
    Nii[1] = c_ii

    D[0, 0] = 3*s_i**2 + 3*s_ii**2 + 2
    D[1, 0] = -3*s_i*c_i - 3*s_ii*c_ii
    D[0, 1] = 3*c_i**2 + 3*c_ii**2 + 2
    Q[:2] = -F_b[:, 0] -F_ck[:, 0] +F_b[:, 1] +F_ck[:, 1] -(3/r)*((tau_b[0]+tau_ck[0])*Ni + (
        tau_b[1]+tau_ck[1])*Nii) + m*r*(omega[0]**2*ri+omega[1]**2*rii)
    for j in range(1, n-1):
        s_i = s_ii
        c_i = c_ii
        ri[:] = rii
        Ni[:] = Nii
        s_ii, c_ii = sin_s[j+1], cos_s[j+1]
        rii[0] = c_ii
        rii[1] = s_ii
        Nii[0] = -s_ii
        Nii[1] = c_ii
        # Ai
        D[2, 2*j-2] = 3*s_i**2 - 1
        D[3, 2*j-2] = -3*s_i*c_i
        D[1, 2*j-1] = -3*s_i*c_i
        D[2, 2*j-1] = 3*c_i**2 - 1
        # Bi
        D[0, 2*j] = 3*s_i**2 + 3*s_ii**2 + 2
        D[1, 2*j] = -3*s_i*c_i - 3*s_ii*c_ii
        D[0, 2*j+1] = 3*c_i**2 + 3*c_ii**2 + 2

        Q[2*j:2*(j+1)] = -F_b[:, j] -F_ck[:, j] +F_b[:, j+1] +F_ck[:, j+1] - (3/r)*(
            (tau_b[j]+tau_ck[j])*Ni + (tau_b[j+1]+tau_ck[j+1])*Nii
        ) + m*r*(omega[j]**2*ri + omega[j+1]**2*rii)
    tmp = cholSolver_banded(D, Q)
    for j in range(n-1):
        F[:, j] = tmp[2*j:2*(j+1)]
    tau_joint[:] = 0
    tau_joint[:-1] += r * (-sin_s[:-1]*F[0, :] + cos_s[:-1]*F[1, :])
    tau_joint[1:] += -r * (-sin_s[1:]*-F[0, :] + cos_s[1:]*-F[1, :])

    ## Correction
    x[0] = 0
    y[0] = 0
    for j in range(n-1):
        x[j+1] = x[j] +r*cos_s[j] +r*cos_s[j+1]
        y[j+1] = y[j] +r*sin_s[j] +r*sin_s[j+1]
    x[:] += -x.mean() +xc
    y[:] += -y.mean() +yc

    x_bar = x - xc
    y_bar = y - yc
    I_body = m *np.sum(x_bar**2 +y_bar**2) + I*n
    tau_body = np.sum(x_bar*fy -y_bar*fx) /(1+bT*dt/M)
    L_body = m *np.sum(x_bar*vy -y_bar*vx)

    omega_next = I_P_inv @ (omega +(dt/I)*(tau_k+tau_joint))
    s_next = s + omega_next*dt

    # variable increase [xc, yc, vxc, vyc]
    axc = fx.sum()/M /(1+bT*dt/M)
    ayc = fy.sum()/M /(1+bT*dt/M)
    vxc += axc * dt
    vyc += ayc * dt
    xc += vxc * dt
    yc += vyc * dt

    cos_s = np.cos(s_next)
    sin_s = np.sin(s_next)

    x[0] = 0
    y[0] = 0
    for j in range(n-1):
        x[j+1] = x[j] +r*cos_s[j] +r*cos_s[j+1]
        y[j+1] = y[j] +r*sin_s[j] +r*sin_s[j+1]
    x[:] += -x.mean() +xc
    y[:] += -y.mean() +yc

    vx[0] = 0
    vy[0] = 0
    for j in range(n-1):
        vx[j+1] = vx[j] -r*omega_next[j]*sin_s[j] -r*omega_next[j+1]*sin_s[j+1]
        vy[j+1] = vy[j] +r*omega_next[j]*cos_s[j] +r*omega_next[j+1]*cos_s[j+1]
    vx[:] += -vx.mean() +vxc
    vy[:] += -vy.mean() +vyc

    x_bar = x - xc
    y_bar = y - yc
    I_body_next = m *np.sum(x_bar**2 +y_bar**2) + I*n
    L_body_next = m *np.sum(x_bar*vy -y_bar*vx)

    ## variable increase [s, omega]
    I_body_mean = (I_body_next + I_body) / 2
    omega[:] = omega_next + (-(L_body_next - L_body) + tau_body*dt) / I_body_mean
    s[:] += omega * dt

    return xc, yc, vxc, vyc

@njit(fastmath=True, cache=True)
def m_steps_numba(
    i, m,
    r, xc, yc, vxc, vyc, s, omega,
    x, y, vx, vy, bII, bT, n, F_b, tau_b,
    F_ck, tau, k, theta_ctrl, tau_c, tau_k, tau_ck, c,
    D, ri, rii, Ni, Nii, Q, F, M, tau_joint,
    dt, I,
    I_P_inv,
    i_snapshot, n_snapshot, dt_snapshot, t_log, xyc_log, s_log, vc_log,
):
    for j in range(m):
        xc, yc, vxc, vyc = step_numba(
            r, xc, yc, vxc, vyc, s, omega,
            x, y, vx, vy, bII, bT, n, F_b, tau_b,
            F_ck, tau, k, theta_ctrl, tau_c, tau_k, tau_ck, c,
            D, ri, rii, Ni, Nii, Q, M/n, F, M, tau_joint,
            dt, I,
            I_P_inv,
        )
        i += 1
        if i % round(dt_snapshot/dt) == 0:
            i_snapshot += 1
            if i_snapshot < n_snapshot:
                t_log[i_snapshot] = i * dt
                xyc_log[i_snapshot, 0] = xc
                xyc_log[i_snapshot, 1] = yc
                s_log[i_snapshot, :] = s
                vc_log[i_snapshot, 0] = vxc
                vc_log[i_snapshot, 1] = vyc
            else:
                break
    return xc, yc, vxc, vyc, i, i_snapshot


# %%
class Worm:
    """
    Caenorhabditis elegans kinetic simulator.
    """
    __is_new_attr_forbidden = False
    def __init__(
        self,
        dt=0.00001,
        n=25,
        M=2,
        L=1,
        bT=1.28e8,
        bII=1.28e8/40,
        k=1.75e5,
        c=1.75e5/5.6,
        simTime=5.0,
        dt_snapshot=0.001,
        angle_init=0,
        theta=None,
        polarity_clockwise=False,
        scale_friction=1,
        scale_muscle=1,
    ):
        """
        Caenorhabditis elegans kinetic simulator.
        
        Quick Example
        -------------
        from elegansbot import Worm
        import numpy as np

        env = Worm()
        for _ in range(env.n_snapshot):
            env.act = 0.7 * np.cos(2 * np.pi * (1.832 * np.linspace(0, 1, env.n - 1) - env.t / 1.6))
            env.steps()
        env.plot_overview()


        Parameters
        ----------
        dt : Time-step (sec) [float] (default: 0.00001)
             Lower dt means higher accuracy and longer run-time.
        n : The number of segment [int] (default: 25)
        M : Mass (ug) [float] (default: 2)
        L : Length (mm) [float] (default: 1)
        bT : Perpendicular frictional constant (ug / sec) [float] (default: 1.28e8) (agar: 1.28e8, water: 5.2e3)
        bII : Parallel frictional constant (ug / sec) [float] (default: 1.28e8/40) (agar: 1.28e8/40, water: 5.2e3/1.5)
        k : Tortional elastic constant (ug * mm^2 / (sec^2 * rad)) [float] (default: 1.75e5)
        c : Tortional damping constant (ug * mm^2 / (sec * rad)) [float] (default: 1.75e5/5.6)
        simTime : Total simulation-time (sec) [float] (default: 5.0)
        dt_snapshot : Time-step between snapshots (sec) [float] (default: 0.001)
                      If you don't want to use memory for saving trajectory,
                      set simTime and dt_snapshot as None.
        angle_init : Initial tail to head direction angle (radian) [float] (default: 0)
        theta : Initial angles between rods (radian) [np.array(dtype=float)] (default: None)
        polarity_clockwise : Dorsoventral polarity. [bool] (default: False)
                             Changing this variable flips worm dorsoventrally.
                             This variable is False when cross product of a vector from posterior end
                             to anterior end of i-rod and a vector from ventral side to dorsal side of i-rod
                             directs out of the screen, which means counter-clockwise direction.
        scale_friction : scaling factor for both perpendicular and parallel frictional constant (default: 1)
        scale_muscle : scaling factor for both tortional elastic and tortional damping constant (default: 1)


        Attributes
        ----------
        n_snapshot : Total number of snapshots.
        t : Current time within simulation.
        t_log : Record of time
        xyc_log : Record of x-y coordinates of the worm.
        s_log : Record of angle from x+ axis of each segment.
        vc_log : Record of velocity of the worm.
        xy_tip_log : Record of x-y coordinates of each tip of segments.


        Example usage 1: Real-time input
        --------------------------------
        from elegansbot import Worm
        import numpy as np

        env = Worm(simTime=5, dt_snapshot=0.001)
        for _ in range(env.n_snapshot):
            env.act = 0.7 * np.cos(2 * np.pi * (1.832 * np.linspace(0, 1, env.n - 1) - env.t / 1.6))
            env.steps()

        env.plot_overview()
        env.plot_speed_graph()
        # %matplotlib notebook # uncomment this line if the code runs in jupyter-notebook.
        env.play_animation(speed_playback=0.5)
        # %matplotlib inline # uncomment this if it's in jupyter-notebook.
        # env.save_animation('demo.mp4') # FFMPEG is required for saving animation.


        Example usage 2: Kymogram input
        -------------------------------
        from elegansbot import Worm
        import numpy as np

        fps = 30
        kymogram = np.zeros((fps*5, 24))
        for i in range(kymogram.shape[0]):
            t = (1/fps) * i
            kymogram[i] = 0.5 * np.cos(2 * np.pi * (1.832 * np.linspace(0, 1, 24) - t / 1.6))

        env = Worm(scale_friction=0.01)
        env.run(kymogram, 1/fps)

        env.plot_overview()
        env.plot_speed_graph()
        # %matplotlib notebook # uncomment this line if the code runs in jupyter-notebook.
        env.play_animation(speed_playback=0.5)
        # %matplotlib inline # uncomment this if it's in jupyter-notebook.
        # env.save_animation('demo.mp4') # FFMPEG is required for saving animation.
        """
        
        # time
        self.__dt = dt
        
        # physical values and variables
        self.n = n
        self.M = M
        self.L = L
        
        m = M/n
        r = L/(2*n)
        I = (1/3)*m*r**2
        self.m = m
        self.r = r
        self.I = I
        
        self.bT = bT * scale_friction
        self.bII = bII * scale_friction
        self.k = k * scale_muscle
        self.c = c * scale_muscle

        self.angle_init = angle_init
        
        # pre-allocated memory
        self.s = np.zeros(n)
        self.omega = np.zeros(n)
        self.P = np.zeros((n, n))
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        self.ri = np.zeros(2)
        self.rii = np.zeros(2)
        self.Ni = np.zeros(2)
        self.Nii = np.zeros(2)
        self.D = np.zeros((4, 2*(n-1)))
        self.Q = np.zeros(2*(n-1))
        self.F_b = np.zeros((2, n))
        self.F_ck = np.zeros((2, n))
        self.F = np.zeros((2, n-1))
        self.tau = np.zeros(n-1)
        self.tau_b = np.zeros(n)
        self.tau_c = np.zeros(n)
        self.tau_k = np.zeros(n)
        self.tau_ck = np.zeros(n)
        self.tau_joint = np.zeros(n)
        self.__theta_ctrl = np.zeros(n-1)
        
        # records
        self.simTime = 0.0
        self.dt_snapshot = 0.0
        self.n_snapshot = 0
        self.i_snapshot = 0
        self.t_log = np.zeros(0)
        self.xyc_log = np.zeros((0, 0))
        self.s_log = np.zeros((1, n))
        self.vc_log = np.zeros((0, 0))
        self.__xy_tip_log = np.zeros((0, 0, 0))
        self.__polarity_clockwise = polarity_clockwise

        self.reset()
        if simTime != None and dt_snapshot != None:
            self.record_init(simTime, dt_snapshot)
        if type(theta) != type(None):
            self.theta = theta
        
        # disabling creation of new attribute for avoiding making any related mistakes.
        self.__is_new_attr_forbidden = True

    def __setattr__(self, key, value):
        if self.__is_new_attr_forbidden and not hasattr(self, key):
            raise TypeError('Creating new attribute is forbidden.')
        super().__setattr__(key, value)

    @property
    def dt(self):
        return self.__dt
    
    @property
    def t(self):
        return self.i * self.__dt
        
    @property
    def polarity_clockwise(self):
        return self.__polarity_clockwise

    @property
    def state(self):
        return np.concatenate([[self.xc, self.yc, self.vxc, self.vyc], self.s, self.omega])
    
    @property
    def theta(self):
        _theta = np.diff(self.s)
        return _theta if self.polarity_clockwise == True else -_theta

    @theta.setter
    def theta(self, _theta):
        _theta = _theta if self.polarity_clockwise == True else -_theta
        _s = np.cumsum(np.concatenate([[0], _theta]))
        self.s = _s -_s.mean() +self.s.mean()
        self.s_log[self.i_snapshot, :] = self.s

    @property
    def theta_log(self):
        _theta_log = np.diff(self.s_log, axis=-1)
        return _theta_log if self.polarity_clockwise == True else -_theta_log

    @property
    def theta_ctrl(self):
        return self.__theta_ctrl if self.polarity_clockwise == True else -self.__theta_ctrl
    
    @theta_ctrl.setter
    def theta_ctrl(self, _theta_ctrl):
        self.__theta_ctrl = _theta_ctrl if self.polarity_clockwise == True else -_theta_ctrl

    @property
    def act(self):
        return self.theta_ctrl
    
    @act.setter
    def act(self, _act):
        self.theta_ctrl = _act
        
    def reset(self):
        """
        Reseting variables to the initial state of worm.
        i : 0, t : 0 sec, center of mass at the origin, every rod at initial angle, zero velocity, zero angular velocity.
        """
        self.i = 0
        self.xc, self.yc = 0.0, 0.0
        self.vxc, self.vyc = 0.0, 0.0
        self.s[:] = (self.angle_init + np.pi)
        self.omega[:] = 0.0
        
        n = self.n
        dt = self.dt
        bT = self.bT
        c = self.c
        k = self.k
        r = self.r
        I = self.I
        
        P = self.P
        P[0, 0] = -(1/3)*(bT/n)*r*r -c -k*dt
        P[0, 1] = c +k*dt
        for j in range(1, n-1):
            P[j, j-1] = c +k*dt
            P[j, j] = -(1/3)*(bT/n)*r*r -2*c -2*k*dt
            P[j, j+1] = c +k*dt
        P[n-1, n-2] = c +k*dt
        P[n-1, n-1] = -(1/3)*(bT/n)*r*r -c -k*dt
        self.I_P_inv = np.linalg.inv(np.eye(n) - P*(dt/I))

    def step(self):
        """
        One step of numerical procedure of simulation.
        """
        self.xc, self.yc, self.vxc, self.vyc = step_numba(
            self.r, self.xc, self.yc, self.vxc, self.vyc, self.s, self.omega,
            self.x, self.y, self.vx, self.vy, self.bII, self.bT, self.n, self.F_b, self.tau_b,
            self.F_ck, self.tau, self.k, self.__theta_ctrl, self.tau_c, self.tau_k, self.tau_ck, self.c,
            self.D, self.ri, self.rii, self.Ni, self.Nii, self.Q, self.m, self.F, self.M, self.tau_joint,
            self.dt, self.I,
            self.I_P_inv,
        )
        self.i += 1
        
    def record_init(
        self,
        simTime,
        dt_snapshot=0.001,
    ):
        """
        Trajectory recording initialization.
        This method calculates total number of snapshot(n_snapshot)
        and allocates record arrays(t_log, xyc_log, s_log, vc_log).
        

        Parameters
        ----------
        simTime : Total simulation-time (sec) [float]
        dt_snapshot : Time-step between snapshots (sec) [float] (default: 0.001)
        

        Example usage
        -------------
        env = Worm()
        env.record_init(simTime=5, dt_snapshot=0.001)
        """
        assert dt_snapshot >= self.__dt
        self.i_snapshot = 0
        self.simTime = simTime
        self.dt_snapshot = dt_snapshot
        dt = self.__dt
        n = self.n
        
        if dt_snapshot < dt:
            dt_snapshot = dt

        n_snapshot = round(simTime/dt_snapshot) + 1
        self.n_snapshot = n_snapshot
        self.t_log = np.zeros(n_snapshot)
        self.xyc_log = np.zeros((n_snapshot, 2))
        self.s_log = np.zeros((n_snapshot, n))
        self.vc_log = np.zeros((n_snapshot, 2))
        self.__xy_tip_log = np.zeros((0, 0, 0))

        # 1st snapshot (keep in mind that these variables may not be 0.)
        self.t_log[self.i_snapshot] = self.t
        self.xyc_log[self.i_snapshot, 0] = self.xc
        self.xyc_log[self.i_snapshot, 1] = self.yc
        self.s_log[self.i_snapshot, :] = self.s
        self.vc_log[self.i_snapshot, 0] = self.vxc
        self.vc_log[self.i_snapshot, 1] = self.vyc
        
    def record_step(self):
        """
        Saving records of current step.
        """
    
        if self.i % round(self.dt_snapshot/self.__dt) == 0:
            self.i_snapshot += 1
            if self.i_snapshot < self.n_snapshot:
                self.t_log[self.i_snapshot] = self.t
                self.xyc_log[self.i_snapshot, 0] = self.xc
                self.xyc_log[self.i_snapshot, 1] = self.yc
                self.s_log[self.i_snapshot, :] = self.s
                self.vc_log[self.i_snapshot, 0] = self.vxc
                self.vc_log[self.i_snapshot, 1] = self.vyc
            else:
                return False
        return True

    def m_steps(self, m): # 17% faster.
        """
        Multiple steps of numerical procedure of simulation.
        """
        self.xc, self.yc, self.vxc, self.vyc, self.i, self.i_snapshot = m_steps_numba(
            self.i, m,
            self.r, self.xc, self.yc, self.vxc, self.vyc, self.s, self.omega,
            self.x, self.y, self.vx, self.vy, self.bII, self.bT, self.n, self.F_b, self.tau_b,
            self.F_ck, self.tau, self.k, self.__theta_ctrl, self.tau_c, self.tau_k, self.tau_ck, self.c,
            self.D, self.ri, self.rii, self.Ni, self.Nii, self.Q, self.F, self.M, self.tau_joint,
            self.dt, self.I,
            self.I_P_inv,
            self.i_snapshot, self.n_snapshot, self.dt_snapshot, self.t_log, self.xyc_log, self.s_log, self.vc_log,
        )
            
    def steps(self):
        """
        Multiple steps of numerical procedure of simulation for 'dt_snapshot' simulation time.
        """
        self.m_steps(round(self.dt_snapshot / self.__dt))

    def run(self, kymogram, dt, theta_init=None):
        """
        Running whole simulation from a kymogram.

        Keep in mind that dt should be the same value as time-step of kymogram.
        (If the FPS(frame per second) is 30, then dt=(1/30).)


        Parameters
        ----------
        kymogram : Kymogram with the first dimension as the time dimension. [np.array(dtype=float)]
        dt : Time-step between each snapshot of input kymogram. (sec) [float]
        theta_init : theta of first snapshot. [np.array(dtype=float)] (Default: None)
        """
        self.reset()
        self.record_init(simTime=(kymogram.shape[0]-1)*dt, dt_snapshot=dt)
        if type(theta_init) == type(None):
            self.theta = kymogram[0]
        else:
            self.theta = theta_init
        for i_snapshot in range(self.n_snapshot):
            self.theta_ctrl = kymogram[i_snapshot]
            self.steps()
    
    @property
    def xy_tip_log(self):
        """
        x-y coordinates of each rod tip at time 't', which is a trajectory of a worm.
        """
        if len(self.__xy_tip_log) > 0:
            pass
        else:
            n = self.n
            r = self.r
            xyc_log = self.xyc_log
            s_log = self.s_log
            
            n_snapshot = self.n_snapshot
            assert n_snapshot == xyc_log.shape[0]
            assert n_snapshot == s_log.shape[0]

            x_tip = np.zeros(n+1)
            y_tip = np.zeros(n+1)
            xy_tip_log = np.zeros((n_snapshot, 2, n+1))
            for i in range(n_snapshot):
                xc, yc = xyc_log[i, :]
                s = s_log[i, :]
                cos_s = np.cos(s)
                sin_s = np.sin(s)
                x_tip[0] = 0
                y_tip[0] = 0
                for j in range(n):
                    x_tip[j+1] = x_tip[j] + 2*r*cos_s[j]
                    y_tip[j+1] = y_tip[j] + 2*r*sin_s[j]
                x = (x_tip[1:]+x_tip[:-1])/2 
                y = (y_tip[1:]+y_tip[:-1])/2
                x_tip[:] = x_tip - np.mean(x) + xc
                y_tip[:] = y_tip - np.mean(y) + yc

                xy_tip_log[i, 0, :] = x_tip
                xy_tip_log[i, 1, :] = y_tip
                
            self.__xy_tip_log = xy_tip_log

        return self.__xy_tip_log
    
    def play_animation(
        env,
        func_plot=plot_outline_n_trajectory,  # drawing function
        speed_playback=1,
        dpi=120,
        bbox_inches_tight=True,
        ax=None,
    ):
        """
        Playing animation of worm's movement from the records.


        Example usage
        -------------
        env.play_animation()


        Example usage in Jupyter notebook
        ---------------------------------
        %matplotlib notebook
        env.play_animation()
        %matplotlib inline
        """

        assert speed_playback > 0
        dt = env.dt_snapshot / speed_playback

        running_in_notebook = is_notebook()
        if not(running_in_notebook):
            plt.ion()

        if type(ax) != type(None):
            fig = ax.get_figure()
        else:
            fig, ax = func_plot(None, 0, env, init_fig=True, dpi=dpi)

        if not(running_in_notebook):
            plt.show()

        func_plot(ax, 0, env, draw=True)

        if bbox_inches_tight == True:
            set_bbox_inches_tight(fig)

        i = 0
        delay = 0
        func_plot(ax, i, env, draw=True)
        while True:
            tmp = time.time()
            step = int(delay/dt)
            if step > 0:
                i += step
                if i >= env.n_snapshot:
                    break
                func_plot(ax, i, env, draw=True)
                if not(running_in_notebook):
                    fig.canvas.flush_events()
                delay -= step*dt
            else:
                time.sleep(dt/10)
            delay += time.time()-tmp

        func_plot(ax, env.n_snapshot-1, env, draw=True)

        if not(running_in_notebook):
            plt.ioff()
            plt.show()
        else:
            plt.close(fig)

    def save_animation(
        env,
        file_name="animation.mp4",
        speed_playback=1,
        dpi=120,
        fps=60,
        bbox_inches_tight=True,
        func_plot=plot_outline_n_trajectory,  # drawing function
        ax=None,
    ):
        """
        Saving animation of worm's movement into MP4 video file.


        This function requires FFMPEG to be installed on your computer.
        and 'ffmpeg' command to be accessible by setting PATH environment variable properly.
        Check out 'https://ffmpeg.org'.


        Example usage
        -------------
        env.save_animation(file_name=f"example.mp4")
        """
        assert speed_playback > 0
        dt = env.dt_snapshot / speed_playback

        running_in_notebook = is_notebook()
        if not(running_in_notebook):
            plt.ion()

        if type(ax) != type(None):
            fig = ax.get_figure()
        else:
            fig, ax = func_plot(None, 0, env, init_fig=True, dpi=dpi)

        func_plot(ax, 0, env) # Drawing object(line, text, ...) initialization

        if bbox_inches_tight == True:
            set_bbox_inches_tight(fig)

        def init(): # Video initialization
            return func_plot(ax, 0, env)

        idx_ = [0]
        i = 0
        delay = 0
        time_elapsed = 0
        while True:
            time_elapsed += 1/fps
            delay += 1/fps
            step = int(delay / dt)
            delay -= step * dt
            i += step
            if i < env.n_snapshot-1:
                idx_.append(i)
            else:
                idx_.append(env.n_snapshot-1)
                break
                
        def update(i):
            return func_plot(ax, idx_[i], env)

        anim = mpl_animation.FuncAnimation(
            fig,
            func=update,
            init_func=init,
            frames=len(idx_),
            interval=1000/fps,
            blit=True,
        )
        anim.save(file_name)
        plt.close(fig)

    def plot_overview(env, n_row=6, dpi=80):
        """
        Plotting overview of worm's trajectory
        """

        xy_tip_log = env.xy_tip_log
        left = np.min(xy_tip_log[:, 0, :], axis=None) - 0.5
        right = np.max(xy_tip_log[:, 0, :], axis=None) + 0.5
        top = np.max(xy_tip_log[:, 1, :], axis=None) + 0.5
        bottom = np.min(xy_tip_log[:, 1, :], axis=None) - 0.5
        width = right - left
        height = top - bottom
        yx_ratio = height / width

        fig, ax_ = plt.subplots(
            nrows=n_row,
            dpi=dpi,
            figsize=(5, n_row * 2),
        )

        for i, idx in enumerate(np.linspace(0, env.n_snapshot-1, n_row).astype(int)):
            ax = ax_[i]
            if i == 0:
                plot_outline_n_trajectory(ax, idx, env, show_legend=True)
            else:
                plot_outline_n_trajectory(ax, idx, env, show_legend=False)

            if i < n_row - 1:
                ax_[i].set_xticklabels([])
                ax_[i].set_xlabel('')

            if yx_ratio > 1/1.7: # to keep the time information in the box
                x_center = (left + right) / 2
                ax.set_xlim(x_center -height*.85, x_center +height*.85)
                ax.texts[0].set_position((x_center -height*.85 +0.1, bottom+0.1))

        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        fig.set_size_inches((bbox.width, bbox.height))
        set_bbox_inches_tight(fig)
        plt.show()

    def plot_speed_graph(env, dpi=80):
        """
        Plot speed graph


        Example usage
        -------------
        env.velocity_graph()
        """
        t_log = env.t_log
        vc_log = env.vc_log

        speed_log = np.sqrt((vc_log**2).sum(axis=-1))
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.axhline(y=0, linestyle='--', linewidth=.5, color=(.3, .3, .3))
        ax.plot(t_log, speed_log, color=[0, 0, 1], label='|v|')
        ax.plot(t_log, vc_log[:, 0], '--', color=[1, 0, 0], label=r'$v_x$')
        ax.plot(t_log, vc_log[:, 1], '--', color=[0, .5, 0], label=r'$v_y$')
        ax.set_title("worm locomotion speed")
        ax.set_xlabel("time (sec)")
        ax.set_ylabel("worm speed (mm/sec)")
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        fig.set_size_inches((bbox.width, bbox.height))
        set_bbox_inches_tight(fig)
        plt.show()
        print(f"average worm speed: {np.mean(speed_log):.5f} (mm/sec)")

    def save(self, file_name):
        """
        Saving data attributes into npz file


        Parameter
        ---------
        file_name : name of npz file
        """
        np.savez_compressed(
            file=file_name,
            __dt=self.__dt,
            n=self.n,
            M=self.M,
            L=self.L,
            m=self.m,
            r=self.r,
            I=self.I,
            bT=self.bT,
            bII=self.bII,
            k=self.k,
            c=self.c,
            angle_init=self.angle_init,
            xc=self.xc,
            yc=self.yc,
            vxc=self.vxc,
            vyc=self.vyc,
            s=self.s,
            omega=self.omega,
            P=self.P,
            I_P_inv=self.I_P_inv,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            ri=self.ri,
            rii=self.rii,
            Ni=self.Ni,
            Nii=self.Nii,
            D=self.D,
            Q=self.Q,
            F_b=self.F_b,
            F_ck=self.F_ck,
            F=self.F,
            tau=self.tau,
            tau_b=self.tau_b,
            tau_c=self.tau_c,
            tau_k=self.tau_k,
            tau_ck=self.tau_ck,
            tau_joint=self.tau_joint,
            __theta_ctrl=self.__theta_ctrl,
            simTime=self.simTime,
            dt_snapshot=self.dt_snapshot,
            n_snapshot=self.n_snapshot,
            i=self.i,
            i_snapshot=self.i_snapshot,
            t_log=self.t_log,
            xyc_log=self.xyc_log,
            s_log=self.s_log,
            vc_log=self.vc_log,
            __xy_tip_log=self.__xy_tip_log,
            __polarity_clockwise=self.__polarity_clockwise,
            __is_new_attr_forbidden=self.__is_new_attr_forbidden,
        )

    def load(self, file_name):
        """
        Loading data attributes from npz file


        Parameter
        ---------
        file_name : name of npz file
        """
        file_name = file_name if file_name[-4:] == '.npz' else file_name + '.npz'
        npz = np.load(file=file_name)

        self.__dt = npz['__dt']
        self.n = npz['n']
        self.M = npz['M']
        self.L = npz['L']
        self.m = npz['m']
        self.r = npz['r']
        self.I = npz['I']
        self.bT = npz['bT']
        self.bII = npz['bII']
        self.k = npz['k']
        self.c = npz['c']
        self.angle_init = npz['angle_init']
        self.xc = npz['xc']
        self.yc = npz['yc']
        self.vxc = npz['vxc']
        self.vyc = npz['vyc']
        self.s = npz['s']
        self.omega = npz['omega']
        self.P = npz['P']
        self.I_P_inv = npz['I_P_inv']
        self.x = npz['x']
        self.y = npz['y']
        self.vx = npz['vx']
        self.vy = npz['vy']
        self.ri = npz['ri']
        self.rii = npz['rii']
        self.Ni = npz['Ni']
        self.Nii = npz['Nii']
        self.D = npz['D']
        self.Q = npz['Q']
        self.F_b = npz['F_b']
        self.F_ck = npz['F_ck']
        self.F = npz['F']
        self.tau = npz['tau']
        self.tau_b = npz['tau_b']
        self.tau_c = npz['tau_c']
        self.tau_k = npz['tau_k']
        self.tau_ck = npz['tau_ck']
        self.tau_joint = npz['tau_joint']
        self.__theta_ctrl = npz['__theta_ctrl']
        self.simTime = npz['simTime']
        self.dt_snapshot = npz['dt_snapshot']
        self.n_snapshot = npz['n_snapshot']
        self.i = npz['i']
        self.i_snapshot = npz['i_snapshot']
        self.t_log = npz['t_log']
        self.xyc_log = npz['xyc_log']
        self.s_log = npz['s_log']
        self.vc_log = npz['vc_log']
        self.__xy_tip_log = npz['__xy_tip_log']
        self.__polarity_clockwise = npz['__polarity_clockwise']
        self.__is_new_attr_forbidden = npz['__is_new_attr_forbidden']


# %%
# Demo
if __name__ == '__main__':
    print('Agar plate')
    env = Worm(dt=0.00005) # Numba JIT spends compile-time here, once.
    for _ in range(env.n_snapshot):
        env.act = 0.7 * np.cos(2 * np.pi * (1.832 * np.linspace(0, 1, env.n - 1) - env.t / 1.6))
        env.steps()
    env.plot_overview()
    env.plot_speed_graph()
    env.play_animation()
#     env.save_animation('demo_crawl.mp4')
    
    print('Water')
    env = Worm(dt=0.00005, bT=5.2e3, bII=5.2e3/1.5) # You can calculate faster with less accurate result by increasing dt.
    for _ in range(env.n_snapshot):
        env.act = 0.5 * np.cos(2 * np.pi * (0.667 * np.linspace(0, 1, env.n - 1) - env.t / 0.4))
        env.steps()
    env.plot_overview()
    env.plot_speed_graph()
    env.play_animation()
#     env.save_animation('demo_swim.mp4')

