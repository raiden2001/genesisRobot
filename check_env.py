import sys
import subprocess
import importlib
import shutil
import os

def run(cmd):
    """Run a shell command and return its output."""
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return result.decode().strip()
    except Exception as e:
        return f"ERROR: {e}"

print("========== GENESIS ENVIRONMENT CHECK ==========\n")

# -------------------
# Python + pip
# -------------------
print("🔍 Python Version:")
print(sys.version, "\n")

print("🔍 pip Version:")
print(run("pip --version"), "\n")

# -------------------
# Check genesis import
# -------------------
print("🔍 Checking Genesis import...")
try:
    import genesis
    print(f"Genesis imported successfully ✔")
    print(f"Genesis version: {genesis.__version__}")
    print(f"Genesis path: {genesis.__file__}\n")
except Exception as e:
    print(f"❌ Genesis import FAILED: {e}\n")

# -------------------
# Check torch import
# -------------------
print("🔍 Checking Torch...")
try:
    import torch
    print(f"Torch imported ✔")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}\n")
except Exception as e:
    print(f"❌ Torch import FAILED: {e}\n")

# -------------------
# Check GLFW
# -------------------
print("🔍 Checking GLFW...")
try:
    import glfw
    print("GLFW imported ✔\n")
except Exception as e:
    print(f"❌ GLFW import FAILED: {e}\n")

# -------------------
# Vulkan
# -------------------
print("🔍 Checking Vulkan runtime...")
vulkaninfo = shutil.which("vulkaninfo")
if vulkaninfo:
    print("vulkaninfo found ✔")
    print(run("vulkaninfo | head -n 20"), "\n")
else:
    print("⚠️ vulkaninfo NOT found (viewer may still run in CPU mode).\n")

# -------------------
# PATH health
# -------------------
print("🔍 Checking PATH order...")
print(os.environ["PATH"], "\n")

# -------------------
# Compiler (NOT required for pip install)
# -------------------
print("🔍 Checking gcc / g++ (optional)...")
print("gcc:", run("gcc --version | head -n 1"))
print("g++:", run("g++ --version | head -n 1"), "\n")

print("========== END OF CHECK ==========")
