import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
rcParams['figure.figsize'] = (10, 6)

_FT2M = 0.3048

_LBF2N = 4.4482216

THR_DATA = {'mach_bp': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8], dtype=np.float64),
            'alt_bp': np.array([0.0, 5.0E3, 10.0E3, 15.0E3, 20.0E3, 25.0E3, 30.0E3, 40.0E3, 50.0E3, 70.0E3]) * _FT2M,
            'thrust': np.array([  # ! alt=0
                24200.0, 28000.0, 28300.0, 30800.0, 34500.0, 37900.0, 36100.0, 34300.0, 32500.0, 30700.0,
                #! alt=5000
                24000.0, 24600.0, 25200.0, 27200.0, 30300.0, 34300.0, 38000.0, 36600.0, 35200.0, 33800.0,
                # ! alt=10000
                20300.0, 21100.0, 21900.0, 23800.0, 26600.0, 30400.0, 34900.0, 38500.0, 42100.0, 45700.0,
                #! alt=15000
                17300.0, 18100.0, 18700.0, 20500.0, 23200.0, 26800.0, 31300.0, 36100.0, 38700.0, 41300.0,
                #! alt=20000
                14500.0, 15200.0, 15900.0, 17300.0, 19800.0, 23300.0, 27300.0, 31600.0, 35700.0, 39800.0,
                #! alt=25000
                12200.0, 12800.0, 13400.0, 14700.0, 16800.0, 19800.0, 23600.0, 28100.0, 32000.0, 34600.0,
                #! alt=30000
                10200.0, 10700.0, 11200.0, 12300.0, 14100.0, 16800.0, 20100.0, 24200.0, 28100.0, 31100.0,
                #! alt=40000
                5700.0, 6500.0, 7300.0,   8100.0,  9400.0, 11200.0, 13400.0, 16200.0, 19300.0, 21700.0,
                #! alt=50000
                3400.0, 3900.0, 4400.0,   4900.0,  5600.0,  6800.0, 8300.0, 10000.0, 11900.0, 13300.0,
                #! alt=70000
                100.0,  200.0,  400.0,    800.0,  1100.0,  1400.0, 1700.0,  2200.0,  2900.0,  3100.0]).reshape((10, 10)) * _LBF2N}

def make_plot(im_cubic, gradients, mn, alt, thrust, title):
    plt.figure()
    plt.suptitle(title)
    plt.subplot(2, 2, 1)
    plt.title('orig. data')
    im = plt.imshow(thrust[::-1], extent=(mn[0], mn[-1],
                                          alt[0], alt[-1]), aspect='auto')
    plt.colorbar(im, cax=make_axes_locatable(
        plt.gca()).append_axes("right", size="5%", pad=0.05))
    plt.xlabel('MN')
    plt.ylabel('Alt')
    plt.subplot(2, 2, 2)
    plt.title('cubic')
    im = plt.imshow(im_cubic[::-1], extent=(mn[0], mn[-1],
                                            alt[0], alt[-1]), aspect='auto')
    plt.colorbar(im, cax=make_axes_locatable(
        plt.gca()).append_axes("right", size="5%", pad=0.05))
    plt.xlabel('MN')
    plt.ylabel('Alt')

    plt.subplot(2, 2, 3)
    plt.title('dthrust/dmn')
    im = plt.imshow(gradients[:, :, 0][::-1], extent=(mn[0], mn[-1],
                                                      alt[0], alt[-1]), aspect='auto')
    plt.colorbar(im, cax=make_axes_locatable(
        plt.gca()).append_axes("right", size="5%", pad=0.05))
    plt.subplot(2, 2, 4)
    plt.title('dthrust/dalt')
    im = plt.imshow(gradients[:, :, 1][::-1], extent=(mn[0], mn[-1],
                                                      alt[0], alt[-1]), aspect='auto')
    plt.colorbar(im, cax=make_axes_locatable(
        plt.gca()).append_axes("right", size="5%", pad=0.05))

    plt.tight_layout()

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def thrust_problem(n, norm=True):
    mn = THR_DATA['mach_bp']
    alt = THR_DATA['alt_bp']
    thrust = THR_DATA['thrust']

    mn_test = np.linspace(mn[0], mn[-1], n)
    alt_test = np.linspace(alt[0], alt[-1], n)

    if norm:
        mn_test = normalize(mn_test)
        alt_test = normalize(alt_test)
        mn = normalize(mn)
        alt = normalize(alt)
        thrust = normalize(thrust)

    mnt, altt = np.meshgrid(mn_test, alt_test, indexing='ij')

    pts_test = np.array([mnt.ravel(), altt.ravel()]).T

    return mn, alt, thrust, pts_test
