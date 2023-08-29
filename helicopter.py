import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, PlotWithSliders
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
import open3d as o3d
from scipy.interpolate import CubicSpline
import cupy as cupy
import cupyx.scipy.signal
from numba import cuda, njit
from numba.cuda.random import create_xoroshiro128p_states
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal.windows import taylor
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

fs = 1e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

# The antenna can run Ka and Ku
# on individual pulses
bg_file = '/data5/SAR_DATA/2022/03282022/SAR_03282022_122555.sar'
fc = 32e9
bandwidth = 500e6
az_bw = 20.
el_bw = 10.
nbg_pts = 300
prf = 3300
bg_extent = (8000, 8000)
collect_time = 2.
init_pt = (30.895723, -86.529416)
init_height = 450.
init_dir = azelToVec(328. * DTR, np.pi / 2)
dep_angle = 30.
squint_angle = -90.
init_spd = 89.4
pts_per_tri = 5
cpi_len = 128
plp = .8
upsample = 1
debug = True
use_sdr = True
noise_sig = 1e-12
ant_offsets = np.array([[0, 0, 1.5]])
n_pulses = int(collect_time * prf)

# Generate the background for simulation
init_llh = (*init_pt, getElevation(init_pt) + init_height)
bg = SDREnvironment(bg_file, num_vertices=500000)
init_enu = llh2enu(*init_llh, bg.origin)
vertices = bg.vertices
triangles = bg.triangles
normals = bg.normals

# Generate the path of the copter
# For no SDR file
tt = np.linspace(0, collect_time, n_pulses)
if use_sdr:
    rp = SDRPlatform(bg_file, origin=bg.origin, fs=fs)
    init_height = rp.pos(rp.gpst[0])[2] - bg.origin[2]
    tt += rp.gpst[0]
else:
    path_e = init_enu[0] + init_dir[0] * init_spd * tt + np.random.normal(0, 1e-6, n_pulses)
    path_n = init_enu[1] + init_dir[1] * init_spd * tt + np.random.normal(0, 1e-6, n_pulses)
    path_u = init_enu[2] + init_dir[2] * init_spd * tt + np.random.normal(0, 1e-6, n_pulses)
    path_r = np.ones_like(path_e) * np.pi / 2
    path_p = np.zeros_like(path_e)
    path_y = np.arctan2(path_n, path_e)
    rp = RadarPlatform(path_e, path_n, path_u, path_r, path_p, path_y, tt, ant_offsets, dep_angle, squint_angle, az_bw,
                       el_bw, fs)
rp.dep_ang = 8 * DTR

# Generate the animation plot function
nrange, frange = rp.calcRanges(init_height)
ptb_rngs = np.array([nrange, nrange, nrange, nrange, frange, frange, frange, frange, nrange, nrange, frange, frange,
                     nrange, nrange, nrange, frange, frange])
anim_plt = PlotWithSliders(vertices, ntraces=1)

# Get a rotation scheme for the radar
ang = np.zeros(tt.shape)
rot_lims = ((-3 + squint_angle) * DTR , (3 + squint_angle) * DTR)
rot_spd = 50 * DTR / prf
rot_dir = 1.
ang[0] = rot_lims[0]
curr_rot_lim = 1
for n in range(1, n_pulses):
    ang[n] = ang[n - 1] + rot_dir * rot_spd * tt[n]
    if ang[n] + rot_dir * rot_spd * tt[n] > rot_lims[curr_rot_lim]:
        rot_dir *= -1
        curr_rot_lim = 0 if curr_rot_lim == 1 else 1
pan = CubicSpline(tt, ang)
el = CubicSpline(tt, rp.att(tt)[0, :])

# General calculations for slant ranges, etc.
nr = rp.calcPulseLength(init_height, plp, use_tac=True)
nsam = rp.calcNumSamples(init_height, plp)
ranges = rp.calcRanges(init_height)
fft_len = findPowerOf2(nsam + nr)
up_fft_len = fft_len * upsample

# Chirp and matched filter calculations
taytay = taylor(up_fft_len)
tayd = np.fft.fftshift(taylor(cpi_len))
taydopp = np.fft.fftshift(np.ones((nsam * upsample, 1)).dot(tayd.reshape(1, -1)), axes=1)
chirp = np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), nr, fs, fc, bandwidth) * 1e5, up_fft_len)
mfilt = chirp.conj()
mfilt[:up_fft_len // 2] *= taytay[up_fft_len // 2:]
mfilt[up_fft_len // 2:] *= taytay[:up_fft_len // 2]
chirp_gpu = cupy.array(np.tile(chirp, (cpi_len, 1)).T, dtype=np.complex128)
mfilt_gpu = cupy.array(np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)

# Generate a test strip of data
tri_vert_indices = cupy.array(triangles, dtype=np.int32)
vert_xyz = cupy.array(vertices, dtype=np.float64)
vert_norms = cupy.array(normals, dtype=np.float64)
scattering_coef = cupy.array(bg.scat_coefs, dtype=np.float64)
ref_coef_gpu = cupy.array(bg.ref_coefs, dtype=np.float64)

if debug:
    pts_debug = cupy.zeros((triangles.shape[0], 3), dtype=np.float64)
    angs_debug = cupy.zeros((triangles.shape[0], 2), dtype=np.float64)
else:
    pts_debug = cupy.zeros((1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1), dtype=np.float64)

# GPU device calculations
threads_per_block = getMaxThreads()
blocks_per_grid = (max(1, triangles.shape[0] // threads_per_block[0] + 1), cpi_len // threads_per_block[1] + 1)
rng_states = create_xoroshiro128p_states(threads_per_block[0] * blocks_per_grid[0], seed=10)

# Data blocks for imaging
dblock = np.zeros((nsam * upsample, cpi_len, n_pulses // cpi_len))

# Run through loop to get data simulated
for pulse_n, t_n in tqdm(enumerate(np.arange(rp.gpst[0], n_pulses, cpi_len))):
    if t_n + cpi_len > n_pulses:
        break
    ts = np.arange(t_n, t_n + cpi_len) / prf
    heading = rp.heading(ts)
    panrx_gpu = cupy.array(pan(ts) + heading, dtype=np.float64)
    elrx_gpu = cupy.array(el(ts), dtype=np.float64)
    posrx_gpu = cupy.array(rp.pos(ts), dtype=np.float64)
    data_r = cupy.zeros((nsam, cpi_len), dtype=np.float64)
    data_i = cupy.zeros((nsam, cpi_len), dtype=np.float64)
    genRangeWithoutIntersection[blocks_per_grid, threads_per_block](rng_states, tri_vert_indices, vert_xyz, vert_norms,
                                                                    scattering_coef, ref_coef_gpu,
                                                                    posrx_gpu, posrx_gpu, panrx_gpu, elrx_gpu,
                                                                    panrx_gpu,
                                                                    elrx_gpu, data_r, data_i, pts_debug, angs_debug,
                                                                    c0 / fc, rp.calcRanges(init_height)[0] / c0,
                                                                    rp.fs, rp.az_half_bw, rp.el_half_bw, pts_per_tri,
                                                                    debug)

    cupy.cuda.Device().synchronize()
    rtdata = cupy.fft.fft(data_r + 1j * data_i, fft_len, axis=0)
    upsample_data = cupy.zeros((up_fft_len, cpi_len), dtype=np.complex128)
    upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
    upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
    rtdata = cupy.fft.ifft(upsample_data * chirp_gpu * mfilt_gpu, axis=0)
    cupy.cuda.Device().synchronize()

    data = np.random.normal(0, noise_sig / np.sqrt(2), (nsam * upsample, cpi_len)) + 1j * \
           np.random.normal(0, noise_sig / np.sqrt(2), (nsam * upsample, cpi_len)) + rtdata.get()[:nsam * upsample, :]

    # Memory management
    del rtdata
    del upsample_data
    del data_r
    del data_i
    del panrx_gpu
    del elrx_gpu
    del posrx_gpu
    cupy.get_default_memory_pool().free_all_blocks()

    dblock[:, :, pulse_n] = db(np.fft.fft(data, axis=1))
    a = rp.heading(tt[0]) + pan(tt[0]) - az_bw * DTR / 2
    b = rp.heading(tt[0]) + pan(tt[0]) + az_bw * DTR / 2
    ptb_theta = np.array([a, a, b, b, b, b, a, a, a, a, a, b, b, b, a, a, b])
    a = el(tt[0]) + el_bw * DTR / 2
    b = el(tt[0]) - el_bw * DTR / 2
    ptb_phi = np.array([a, b, b, a, a, b, b, a, a, b, b, b, b, a, a, a, a])
    ptb_pos = rp.pos(tt[0])[:, None] + azelToVec(ptb_theta, ptb_phi) * ptb_rngs[None, :]
    anim_plt.addFrame(ptb_pos.T, trace=1)

d_pts = pts_debug.get()
d_angs = angs_debug.get()

del tri_vert_indices
del vert_xyz
del vert_norms
del scattering_coef
del chirp_gpu
del mfilt_gpu
del pts_debug
del angs_debug
cupy.get_default_memory_pool().free_all_blocks()

datafig = px.imshow(dblock[:, :, 0], aspect='auto')
datafig.show()

anim_plt.render()
