"""
To install vsr prior to running the scripts,
- navigate to this directory in the terminal
- activate wanted conda environnement
- run 'python install.py'
"""
import subprocess
import sys


def python_call(module: str, arguments, **kwargs):
    subprocess.run([sys.executable, "-m", module] + list(arguments), **kwargs)


python_call("pip", ["install", "-e", "./src[dev,test,docs,check]"])
