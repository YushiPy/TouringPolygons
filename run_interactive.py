
import os
import sys
import subprocess

try:
	import pygame # type: ignore
except ImportError:
	subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
	import pygame

current_directory = os.path.dirname(__file__)
os.chdir(current_directory)

path = os.path.join(current_directory, "src", "UnconstrainedTPP", "Interactive", "main.py")

subprocess.run([sys.executable, path])