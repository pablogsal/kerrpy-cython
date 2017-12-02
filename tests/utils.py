import inspect
import subprocess

def compiled_with_openmp(module):
    try:
        elf_out = subprocess.check_output(["readelf","-p",".GCC.command.line",module.__file__])
    except subprocess.CalledProcessError:
        return False
    return "-fopenmp" in elf_out.decode()
