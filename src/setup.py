import numpy as np
import distutils

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('', parent_package, top_path)

    if distutils.version.StrictVersion(np.version.version) > distutils.version.StrictVersion('1.6.1'):
        config.add_extension('mll', ['mll.f90','rc3jj.f'],
                             libraries=[], f2py_options=[],
                             extra_f90_compile_args=['-O2'],
                             extra_compile_args=['-std=legacy'], extra_link_args=[],)
    else:
        config.add_extension('mll', ['mll.f90','rc3jj.f'],
                             libraries=[], f2py_options=[],
                             extra_compile_args=['-std=legacy'], extra_link_args=[],)

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(name='mll',
        configuration=configuration,
        version='0.1.0',
        author='Federico Bianchini',
        #author_email='federico.bianxini@gmail.com',
        packages=['.'],#['curvspec','curvspec.test'],
        description='Routines to compute power spectra of curved-sky (Healpix) maps based on the MASTER algorithm')#,
        #url='https://github.com/fbianchini/CurvSpec')


    setup(
        name="grid_beta_module",
        version="0.0",
        description="compute grid of needlet coefficients over an array of parameters",
        zip_safe=False,
        packages=['.']#find_packages(),
        #python_requires=">=3.7",
        #install_requires=[
        #"numpy>=1.19",
        #"camb>=1.3.5",
        #"euclid_windows",
        #],
    )

    setup(
        name="likelihood_analysis_module",
        version="0.0",
        description="Compute posterior distribution for cosmological parameters over a grid of spectra",
        zip_safe=False,
        packages=['.']#find_packages(),
        #python_requires=">=3.7",
        #install_requires=[
        #"numpy>=1.19",
        #"camb>=1.3.5",
        #"euclid_windows",
        #],
    )

