import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from .utils import full_path


class Constants:
    G_f = 1.166e-11  # MeV^{-2}
    m_e = 0.511  # MeV
    s_w = np.sqrt(0.232)  # PDG
    N_a = 6.02e23
    hbar = 6.58e-22  # MeV*s
    c = 3e10  # cm/s
    alpha = 1 / 137
    mu_b = np.sqrt(4 * np.pi * alpha) / (2 * m_e)  # MeV^{-1}


class XeShell:
    # (n_electron, energy [keV])
    shells = [(54, 34.561),
              (52, 5.453),
              (50, 5.107),
              (47 ,4.786),
              (44, 1.148),
              (42, 1.002),
              (39, 0.941),
              (36, 0.689),
              (31, 0.676),
              (26, 0.213),
              (24, 0.147),
              (21, 0.145),
              (18, 0.070),
              (13, 0.068),
              (8, 0.023),
              (6, 0.013),
              (3, 0.012)
              ]
    k_shell = shells[0][1]
    outer_shell = shells[-1][1]

    def __init__(self):
        pass

    def num_accessible(self, E):
        if E >= self.k_shell:
            return 54
        elif E < self.outer_shell:
            return 0
        else:
            next_shell = self.k_shell
            for n, e in self.shells:
                if e < E <= next_shell:
                    return n
            return 3


C = Constants()
Xe = XeShell()

survival_df = pd.read_csv(full_path('data/msw/LMA_borexino_2018.txt'), sep='\t')
nue_survival = interp1d(survival_df.x, survival_df.y, bounds_error=False,
                        fill_value=(survival_df.y.values[0], survival_df.y.values[-1]))


class SolarNu:
    """Base class for all solar neutrino objects"""
    label = "None"

    def __init__(self):
        self._flux = self.load_flux()

    def load_flux(self):
        raise NotImplementedError

    def flux(self, E):
        return self._flux(E)

    def dRdE_SM(self, Er):
        """SM differential rate, in units events/N_a/keV/year"""

        # so that it can handle arrays
        if hasattr(Er, "__len__"):
            return np.array([self.dRdE_SM(E) for E in Er])

        # E_r in keV, but m_e in MeV above
        # so convert E_r to MeV
        Enu_min = np.sqrt(C.m_e * Er / 1000 / 2)

        def xsec(E_nu):
            if Er > max_recoil_energy(E_nu):
                return 0
            p = nue_survival(E_nu)
            return p * SM_diff_xsec(Er, E_nu) + (1 - p) * SM_diff_xsec(Er, E_nu, nu='mu')

        # Es = df.E[df.E>Enu_min].values
        integral = quad(lambda x: xsec(x) * self.flux(x), Enu_min, np.inf,
                        epsabs=1e-12, limit=200)

        return integral[0] * C.N_a * 3600 * 24 * 365 / 1000

    def dRdE_numu(self, Er, mu):
        """Nu magnetic moment differential rate, in units events/N_a/keV/year"""

        # so that it can handle arrays
        if hasattr(Er, "__len__"):
            return np.array([self.dRdE_numu(E, mu) for E in Er])

        # E_r in keV, but m_e in MeV above
        # so convert E_r to MeV
        Enu_min = np.sqrt(C.m_e * Er / 1000 / 2)

        def xsec(E_nu):
            if Er > max_recoil_energy(E_nu):
                return 0
            return numu_diff_xsec(Er, E_nu, mu)

        # Es = df.E[df.E>Enu_min].values
        integral = quad(lambda x: xsec(x) * self.flux(x), Enu_min, np.inf,
                        epsabs=1e-12, limit=200)

        return integral[0] * C.N_a * 3600 * 24 * 365 / 1000

    def dRdE_millicharge(self, Er, q):

        # so that it can handle arrays
        if hasattr(Er, "__len__"):
            return np.array([self.dRdE_millicharge(E, q) for E in Er])

        # so convert E_r to MeV
        Enu_min = np.sqrt(C.m_e * Er / 1000 / 2)

        def xsec(E_nu):
            if Er > max_recoil_energy(E_nu):
                return 0
            return nu_millicharge_diff_xsec(Er, q)

        # Es = df.E[df.E>Enu_min].values
        integral = quad(lambda x: self.flux(x) * xsec(x), Enu_min, np.inf, epsabs=1e-12, limit=200)
        return integral[0] * C.N_a * 3600 * 24 * 365 / 1000 / 54

    def dRdE_BminusL(self, Er, g, M_A, m_nu=0):
        """Nu magnetic moment differential rate, in units events/N_a/keV/year"""

        # so that it can handle arrays
        if hasattr(Er, "__len__"):
            return np.array([self.dRdE_BminusL(E, g, M_A, m_nu) for E in Er])

        # E_r in keV, but m_e in MeV above
        # so convert E_r to MeV
        Enu_min = np.sqrt(C.m_e * Er / 1000 / 2)

        def xsec(E_nu):
            if Er > max_recoil_energy(E_nu):
                return 0
            return BminusL_diff_xsec(Er, E_nu, g, M_A, m_nu)

        # Es = df.E[df.E>Enu_min].values
        integral = quad(lambda x: xsec(x) * self.flux(x), Enu_min, np.inf,
                        epsabs=1e-12, limit=200)

        return integral[0] * C.N_a * 3600 * 24 * 365 / 1000


class SolarNuLine:
    """For Line sources (pep and Be7)"""
    E_peaks = None

    def __init__(self):
        self._flux = self.load_flux()

    def load_flux(self):
        raise NotImplementedError

    @property
    def flux(self):
        # return Es, flux arrays (in case there are multiple peaks)
        return [(E, flux) for E, flux in zip(self.E_peaks, self._flux)]

    def dRdE_SM(self, Er):
        # return Es, rates arrays (in case there are multiple peaks)
        ret = None
        for (Enu, flux) in self.flux:
            p = nue_survival(Enu)
            xsec = (p*SM_diff_xsec(Er, Enu, nu='e') + (1-p)*SM_diff_xsec(Er, Enu, nu='mu'))
            xsec *= (Er <= max_recoil_energy(Enu))
            rate = xsec * flux * 3600 * 24 * 365 * C.N_a / 1000
            if ret is None:
                ret = rate
            else:
                ret += rate
        return ret

    def dRdE_numu(self, Er, mu):
        # return Es, rates arrays (in case there are multiple peaks)
        ret = None
        for (Enu, flux) in self.flux:
            xsec = numu_diff_xsec(Er, Enu, mu) * (Er <= max_recoil_energy(Enu))
            rate = xsec * flux * 3600 * 24 * 365 * C.N_a / 1000
            if ret is None:
                ret = rate
            else:
                ret += rate
        return ret

    def dRdE_BminusL(self, Er, g, M_A, m_nu=0):
        # return Es, rates arrays (in case there are multiple peaks)
        ret = None
        for (Enu, flux) in self.flux:
            xsec = BminusL_diff_xsec(Er, Enu, g, M_A, m_nu) * (Er <= max_recoil_energy(Enu))
            rate = xsec * flux * 3600 * 24 * 365 * C.N_a / 1000
            if ret is None:
                ret = rate
            else:
                ret += rate
        return ret


class SolarPP(SolarNu):
    """pp neutrinos"""
    label = 'pp'

    def load_flux(self):
        df = pd.read_csv(full_path('data/flux_data/pp.csv'))
        return interp1d(df.E, df.flux, bounds_error=False, fill_value=0)


class SolarPEP(SolarNuLine):
    """pep neutrinos"""
    label = 'pep'

    def load_flux(self):
        df = pd.read_csv(full_path('data/flux_data/pep.csv'))
        self.E_peaks = df.E.values
        return df.flux.values


class SolarBE7(SolarNuLine):
    """beryllium-7 neutrinos"""
    label = '7Be'

    def load_flux(self):
        df = pd.read_csv(full_path('data/flux_data/be7.csv'))
        self.E_peaks = df.E.values
        return df.flux.values


class SolarB8(SolarNu):
    """"boron-8 neutrinos"""
    label = '8B'

    def load_flux(self):
        df = pd.read_csv(full_path('data/flux_data/b8.csv'))
        return interp1d(df.E, df.flux, bounds_error=False, fill_value=0)


class SolarHEP(SolarNu):
    """hep neutrinos"""
    label = 'hep'

    def load_flux(self):
        df = pd.read_csv(full_path('data/flux_data/hep.csv'))
        return interp1d(df.E, df.flux, bounds_error=False, fill_value=0)


class SolarN13(SolarNu):
    """N-13 from CNO"""
    label = '13N'

    def load_flux(self):
        df = pd.read_csv(full_path('data/flux_data/n13.csv'))
        return interp1d(df.E, df.flux, bounds_error=False, fill_value=0)


class SolarO15(SolarNu):
    """O-15 from CNO"""
    label = '15O'

    def load_flux(self):
        df = pd.read_csv(full_path('data/flux_data/o15.csv'))
        return interp1d(df.E, df.flux, bounds_error=False, fill_value=0)


class SolarF17(SolarNu):
    """F-17 from CNO"""
    label = '17F'

    def load_flux(self):
        df = pd.read_csv(full_path('data/flux_data/f17.csv'))
        return interp1d(df.E, df.flux, bounds_error=False, fill_value=0)

class SolarCNO(SolarNu):
    label = 'CNO'

    def load_flux(self):
        nu1 = SolarN13()
        nu2 = SolarO15()
        nu3 = SolarF17()

        def f(E):
            return nu1.flux(E) + nu2.flux(E) + nu3.flux(E)

        return f


def max_recoil_energy(Enu):
    return 2*Enu**2 / (C.m_e + 2*Enu) * 1000 # keV


def load_solars():
    return [SolarPP(), SolarPEP(), SolarBE7(), SolarB8(), SolarHEP(), SolarCNO()]


def SM_diff_xsec(Er_keV, Enu, nu='e'):
    """returns Standard Model differential cross section in units of cm^2/keV"""
    # break into a few parts for readability
    Er = Er_keV / 1000
    a = C.G_f**2 * C.m_e / (2*np.pi*Enu**2)
    b = 4*C.s_w**4 * (2*Enu**2 + Er**2 - Er*(2*Enu+C.m_e))
    c = 2*C.s_w**2*(Er*C.m_e - 2*Enu**2)
    si_corr = (C.hbar*C.c)**2
    if nu == 'e':
        return a*(b - c + Enu**2)*si_corr
    elif nu in ['mu', 'tau']:
        return a*(b + c + Enu**2)*si_corr
    else:
        raise NotImplementedError("nu type needs to be e, mu, tau!")


def numu_diff_xsec(Er_keV, Enu, mu):
    """Neutrino magnetic moment cross section"""
    # Er in keV
    Er = Er_keV / 1000
    # mu is in units of bohr magnetons
    si_corr = (C.hbar * C.c) ** 2
    return (mu * C.mu_b) ** 2 * C.alpha * (1 / Er - 1 / Enu) * si_corr


def nu_millicharge_diff_xsec(Er_keV, q):
    """Doesn't depend on Enu? See https://arxiv.org/pdf/hep-ph/0612203.pdf
    Returns xsec in cm^2/MeV/electron"""

    Er = Er_keV / 1000
    si_corr = (C.hbar * C.c) ** 2
    return 2*np.pi*C.alpha * (1/(C.m_e * Er**2)) * q**2 * si_corr


def BminusL_diff_xsec(Er_keV, Enu, g, M_A, m_nu=0):
    Er = Er_keV / 1000
    si_corr = (C.hbar * C.c) ** 2
    p2 = Enu ** 2 - m_nu ** 2
    return ( ((g**4 * C.m_e)/(4*np.pi*p2*(M_A**2 + 2*Er*C.m_e)**2)) *
              (2*Enu**2 + Er**2 - 2*Enu*Er - Er*C.m_e - m_nu**2) ) * si_corr
