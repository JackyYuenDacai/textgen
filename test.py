import os
import ctypes

# Add common CUDA and system paths
cuda_path = os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
if os.path.exists(cuda_path):
    os.add_dll_directory(os.path.join(cuda_path, "bin"))

# Try to load the DLL manually
try:
    ctypes.CDLL(r"F:\GitHub\textgen\installer_files\env\Lib\site-packages\exllamav3_ext.cp313-win_amd64.pyd")
except OSError as e:
    print(e)