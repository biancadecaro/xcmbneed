import numpy as np
import distutils

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('', parent_package, top_path)

    if distutils.version.StrictVersion(np.version.version) > distutils.version.StrictVersion('1.6.1'):
        config.add_extension('mll', ['curvspec/mll.f90','curvspec/rc3jj.f'],
                             libraries=[], f2py_options=[],
                             extra_f90_compile_args=['-O3'],
                             extra_compile_args=['-std=legacy'], extra_link_args=[],)
    else:
        config.add_extension('mll', ['curvspec/mll.f90','curvspec/rc3jj.f'],
                             libraries=[], f2py_options=[],
                             extra_compile_args=['-std=legacy'], extra_link_args=[],)

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='CurvSpec',
        configuration=configuration,
        version='0.1.0',
        author='Federico Bianchini',
        author_email='federico.bianxini@gmail.com',
        packages=['curvspec','curvspec.test'],
        description='Routines to compute power spectra of curved-sky (Healpix) maps based on the MASTER algorithm',
        url='https://github.com/fbianchini/CurvSpec')

