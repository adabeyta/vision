import importlib.machinery
import os

from torch.hub import _get_torch_home


_HOME = os.path.join(_get_torch_home(), "datasets", "vision")
_USE_SHARDED_DATASETS = False
IN_FBCODE = False


def _download_file_from_remote_location(fpath: str, url: str) -> None:
    pass


def _is_remote_location_available() -> bool:
    return False


try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401


# Tracks DLL directories already added on Windows so repeated extension loads
# do not register duplicate search-path entries. Unused on other platforms.
_REGISTERED_DLL_DIRECTORIES = set()


def _add_windows_cuda_dll_directories():
    # Adds the CUDA toolkit bin dir to the Windows DLL search path so a CUDA
    # extension can find the CUDA DLLs it links against. The image_stable
    # extension links nvjpeg, whose nvjpeg64_*.dll lives in the toolkit bin dir.
    # PyTorch only adds the dirs of the CUDA libs it bundles itself, and it
    # bundles cudart but not nvjpeg, so without this nvjpeg64_*.dll is never
    # found and image_stable fails to load. The toolkit is located the same way
    # torch.utils.cpp_extension does: the CUDA_PATH or CUDA_HOME env var first,
    # then nvcc on PATH (nvcc sits in that same bin dir). No-op when none of
    # those resolve. Windows only, so Linux and ROCm (which resolve via RPATH)
    # never reach here.
    import shutil

    cuda_homes = [os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME")]
    if nvcc := shutil.which("nvcc"):
        cuda_homes.append(os.path.dirname(os.path.dirname(nvcc)))
    for cuda_home in cuda_homes:
        if not cuda_home:
            continue
        cuda_bin = os.path.join(cuda_home, "bin")
        if cuda_bin not in _REGISTERED_DLL_DIRECTORIES and os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)
            _REGISTERED_DLL_DIRECTORIES.add(cuda_bin)


def _get_extension_path(lib_name):

    lib_dir = os.path.dirname(__file__)
    if os.name == "nt":
        # Register the main torchvision library location on the default DLL path
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p

        os.add_dll_directory(lib_dir)
        _add_windows_cuda_dll_directories()

        kernel32.SetErrorMode(prev_error_mode)

    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError(f"Could not find module '{lib_name}' in {lib_dir}")

    return ext_specs.origin
