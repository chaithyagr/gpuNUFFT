import os
import sys
from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext
from importlib import import_module
import platform
from pprint import pprint
import subprocess


release_info = {}

# ---------------------------------------------------------------------------
# Binary wheel configuration
#
# When GPUNUFFT_WHEEL_CUDA is set (e.g. "13.0", done by cibuildwheel, see
# pyproject.toml), the extension is built for redistribution:
#   * the CUDA runtime is linked statically,
#   * cuFFT (the only CUDA library used) is NOT bundled; it is declared as a
#     dependency on NVIDIA's pip package and found through an RPATH relative
#     to the module ($ORIGIN/nvidia/...).
# When it is unset (plain `pip install` from the sdist), the extension is built
# against the local CUDA toolkit exactly as before, with no extra dependencies.
# ---------------------------------------------------------------------------
WHEEL_CUDA_CONFIGS = {
    "13": {
        "name": "gpuNUFFT",
        "install_requires": ["nvidia-cufft>=12.0,<13"],  # cuFFT of CUDA 13.x
        "rpath": "$ORIGIN/nvidia/cu13/lib",
    },
    "12": {
        "name": "gpuNUFFT-cu12",
        "install_requires": ["nvidia-cufft-cu12>=11.0,<12"],  # cuFFT of CUDA 12.x
        "rpath": "$ORIGIN/nvidia/cufft/lib",
    },
}

WHEEL_CUDA = os.environ.get("GPUNUFFT_WHEEL_CUDA", "").strip()
if WHEEL_CUDA:
    _major = WHEEL_CUDA.split(".")[0]
    if _major not in WHEEL_CUDA_CONFIGS:
        raise RuntimeError(
            "GPUNUFFT_WHEEL_CUDA={} is not supported, use one of {}".format(
                WHEEL_CUDA, sorted(WHEEL_CUDA_CONFIGS)))
    WHEEL_CONFIG = WHEEL_CUDA_CONFIGS[_major]
else:
    WHEEL_CONFIG = None

class CMakeExtension(Extension):

    def __init__(self, name, sourcedir):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    """ Define a cmake build extension.
    """

    @staticmethod
    def _preinstall(package):
        """ Pre-install PyPi packages before running cmake.
        """



    def _set_pybind_path(self):
        """ Set path to Pybind11 include directory.
        """
        self.pybind_path = getattr(import_module('pybind11'), 'get_include')()

    def run(self):
        """ Redifine the run method.
        """
        # Set preinstall requirements
        preinstall = "pybind11"

        # Preinstall packages
        self._preinstall(preinstall)

        # Set Pybind11 path
        self._set_pybind_path()

        # Check cmake is installed and is sufficiently new.
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        # Build extensions
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        """ Build extension with cmake.
        """
        # Define cmake arguments
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
                      "-DPYTHON_EXECUTABLE=" + sys.executable,
                      "-DPython3_EXECUTABLE=" + sys.executable,
                      "-DGEN_PYTHON_FILES=ON",
                      "-DGEN_MEX_FILES=OFF",
                      "-DPYBIND11_INCLUDE_DIR=" + self.pybind_path]
        if WHEEL_CONFIG is not None:
            cmake_args += [
                "-DCUDA_USE_STATIC_CUDA_RUNTIME=ON",
                "-DGPUNUFFT_RPATH=" + WHEEL_CONFIG["rpath"],
                "-DGEN_ATM_LIB=OFF",
            ]
        # Extra user supplied arguments, e.g.
        # GPUNUFFT_CMAKE_ARGS="-DGPUNUFFT_CUDA_ARCHITECTURES=89"
        cmake_args += os.environ.get("GPUNUFFT_CMAKE_ARGS", "").split()
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            cmake_args += ['-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL",
                                  str(os.cpu_count() or 8))
            build_args += ['--', '-j' + jobs]

        # Call cmake in specific environment
        env = os.environ.copy()
        env["CXXFLAGS"] = '{0} -DVERSION_INFO=\\"{1}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version())
        build_temp_dir = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp_dir):
            os.makedirs(build_temp_dir)
        print("Building " + ext.name + " in {0}...".format(build_temp_dir))
        print("Cmake args:")
        pprint(cmake_args)
        print("Cmake build args:")
        pprint(build_args)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args,
                              cwd=build_temp_dir, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args,
                              cwd=build_temp_dir)
        print()

setup(
    name=WHEEL_CONFIG["name"] if WHEEL_CONFIG else "gpuNUFFT",
    version="0.10.8",
    description="gpuNUFFT - An open source GPU Library for 3D Gridding and NUFFT",
    ext_modules=[
        CMakeExtension("gpuNUFFT", sourcedir=os.path.join("CUDA")),
    ],
    cmdclass={
        "build_ext": CMakeBuild,
    },
    install_requires=WHEEL_CONFIG["install_requires"] if WHEEL_CONFIG else [],
    python_requires=">=3.8",
    zip_safe=False
)
