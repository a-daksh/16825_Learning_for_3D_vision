import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
    # Add library directory for linking
    ldflags = []
    if 'CONDA_PREFIX' in os.environ:
        lib_dir = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
        if os.path.exists(lib_dir):
            ldflags.append(f'-L{lib_dir}')
            ldflags.append(f'-Wl,-rpath,{lib_dir}')
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']
    ldflags = []

    # find cl.exe
    def find_cl_path():
        import glob
        for program_files in [r"C:\\Program Files (x86)", r"C:\\Program Files"]:
            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(glob.glob(r"%s\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % (program_files, edition)), reverse=True)
                if paths:
                    return paths[0]
    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

_backend = load(name='_raymarching',
                extra_cflags=c_flags,
                extra_cuda_cflags=nvcc_flags,
                extra_ldflags=ldflags if 'ldflags' in locals() else None,
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'raymarching.cu',
                    'bindings.cpp',
                ]],
                )

__all__ = ['_backend']
