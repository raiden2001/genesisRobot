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
