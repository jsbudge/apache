from simulib.platform_helper import RadarPlatform
import numpy as np
import yaml

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254


class ApachePlatform(RadarPlatform):
    def __init__(self,
                 params: dict,
                 e: np.ndarray = None,
                 n: np.ndarray = None,
                 u: np.ndarray = None,
                 r: np.ndarray = None,
                 p: np.ndarray = None,
                 y: np.ndarray = None,
                 t: np.ndarray = None,
                 tx_offset: np.ndarray = None,
                 rx_offset: np.ndarray = None,
                 gimbal: np.ndarray = None,
                 gimbal_offset: np.ndarray = None,
                 gimbal_rotations: np.ndarray = None,
                 dep_angle: float = 45.,
                 fs: float = 2e9,
                 gps_t: np.ndarray = None,
                 gps_az: np.ndarray = None,
                 gps_rxpos: np.ndarray = None,
                 gps_txpos: np.ndarray = None,
                 tx_num: int = 0,
                 rx_num: int = 0,
                 wavenumber: int = 0):
        super().__init__(e, n, u, r, p, y, t, tx_offset, rx_offset, gimbal, gimbal_offset, gimbal_rotations, dep_angle,
                         0., params['az_min_bw'], params['el_min_bw'], fs, gps_t, gps_az, gps_rxpos,
                         gps_txpos, tx_num, rx_num,
                         wavenumber)

        self.params = params

    def getValidPulseTimings(self, prf, pulse_time, cpi_len):
        tcpi = self.gpst[0] + np.arange(self.gpst[0], self.gpst[-1], 1 / prf)
        th_b = self.att(tcpi)[:, 1]
        u = self.params['wheel_height_m'] * np.cos(th_b) - self.params['phase_center_offset_m'] * np.sin(th_b)
        u /= -np.sin(self.tilt(tcpi)) * np.sin(th_b) - np.cos(self.tilt(tcpi)) * np.cos(th_b)
        cp0 = np.matmul(rotateY(self.tilt(tcpi)),
                        np.array([[-u * np.sin(self.el_half_bw) + self.params['phase_center_offset_m']],
                                  [u * np.tan(self.az_half_bw)],
                                  [u * np.cos(self.el_half_bw) + self.params['wheel_height_m']]]).swapaxes(0,
                                                                                                           2).swapaxes(
                            1, 2)).squeeze(2)
        cp1 = np.matmul(rotateY(self.tilt(tcpi)),
                        np.array([[-u * np.sin(self.el_half_bw) + self.params['phase_center_offset_m']],
                                  [-u * np.tan(self.az_half_bw)],
                                  [u * np.cos(self.el_half_bw) + self.params['wheel_height_m']]]).swapaxes(0,
                                                                                                           2).swapaxes(
                            1, 2)).squeeze(2)

        arclen = np.linalg.norm(cp0, axis=1) * np.arccos(1 - (np.linalg.norm(cp0 - cp1, axis=1)) / (2 * np.linalg.norm(cp0, axis=1)**2))
        sweep_radius = 2 * np.pi * np.linalg.norm(cp0)
        tb = (sweep_radius - 4 * (2 * arclen + self.params['blade_chord_m']))
        tb /= 4 * np.linalg.norm(cp0) * self.params['rotor_velocity_rad_s']
        ti = ((2 * arclen + self.params['blade_chord_m']) /
              (4 * np.linalg.norm(cp0) * self.params['rotor_velocity_rad_s']))
        Npi = int(min(tb * prf))
        extra_pulses = Npi % cpi_len
        Npi -= extra_pulses
        Npo = (int(max(np.ceil(ti * prf))) +
               extra_pulses)

        valids = np.concatenate([np.arange(n * Npi + n * Npo, (n + 1) * Npi + n * Npo) for n in range(len(tcpi) // Npi)])
        valids = valids[valids < len(tcpi)]

        return tcpi[valids]


def rotateY(theta):
    z = np.zeros(len(theta))
    return np.array([[np.cos(theta), z, np.sin(theta)], [z, z + 1, z], [-np.sin(theta), z, np.cos(theta)]]).T


if __name__ == '__main__':
    with open('./vae_config.yaml', 'r') as file:
        try:
            wave_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    apache_params = wave_config['apache_params']
    # Get parameters for the Apache specs
    req_slant_range = apache_params['vehicle_slant_range_max']
    req_alt = apache_params['alt_max']
    ground_range = np.sqrt(req_slant_range ** 2 - req_alt ** 2)
    req_dep_ang = np.arccos(req_alt / req_slant_range) + apache_params['el_min_bw'] * DTR * 9
    ngpssam = 500
    e = np.linspace(ground_range + .01, ground_range, ngpssam)
    n = np.linspace(0, 0, ngpssam)
    u = np.linspace(req_alt, req_alt, ngpssam)
    r = np.zeros_like(e)
    p = np.zeros_like(e)
    y = np.zeros_like(e) + np.pi / 2
    t = np.arange(ngpssam) / 100
    gim_pan = np.zeros(ngpssam)
    gim_el = np.zeros_like(gim_pan) + np.arccos(req_alt / req_slant_range)
    goff = np.array([apache_params['phase_center_offset_m'], 0., apache_params['wheel_height_m']])
    grot = np.array([0., 0., 0.])
    apache = ApachePlatform(apache_params, e, n, u, r, p, y, t, dep_angle=req_dep_ang / DTR,
                            gimbal=np.array([gim_pan, gim_el]).T, gimbal_rotations=grot, gimbal_offset=goff, fs=2e9)

    check = apache.getValidPulseTimings(1382, apache.calcPulseLength(u.mean()), 32)
