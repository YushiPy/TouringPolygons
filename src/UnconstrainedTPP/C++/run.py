
import sys
import os
import subprocess
from collections.abc import Callable

COMPILER = "g++"
STANDARD = "c++26"
FLAGS = f"-std={STANDARD} -O2"



def has_modified(filename: str, object_file: bool) -> bool:

	basename = get_basename(filename)

	if object_file:
		if not os.path.isfile(basename + ".o"):
			return True
		recent_time = os.path.getmtime(basename + ".o")
	else:
		if not os.path.isfile(basename):
			return True
		recent_time = os.path.getmtime(basename)
	
	modified_time = os.path.getmtime(filename)
	
	return modified_time > recent_time

def get_basename(filename: str) -> str:
	return os.path.splitext(filename)[0]

def get_dependencies(filename: str) -> list[str]:

	command = f"{COMPILER} {FLAGS} -MMD -c {filename}"
	subprocess.run(command, shell=True, check=True)

	basename = get_basename(filename)

	with open(f"{basename}.d", "r") as f:
		dependencies = f.read().strip().split()

	subprocess.run(f"rm {basename}.d", shell=True, check=True)

	return [dep for dep in dependencies[1:] if dep != filename]

def compile_file(filename: str, object_file: bool = False, report: Callable[..., None] = print) -> bool:

	basename = get_basename(filename)

	if filename.endswith(".h"):
		report(f"Warning: '{filename}' is a header file. Compiling it may not be intended.")
		filename = basename + ".cpp"
	elif not filename.endswith(".cpp"):
		filename += ".cpp"
	
	if not os.path.isfile(filename):
		raise FileExistsError(f"File '{filename}' not found.")

	report(f"👀 Compiling '{filename}'...")

	file_has_modified = has_modified(filename, object_file)

	dependencies = get_dependencies(filename)
	dependency_files = [filename]

	compiled_dependency = False

	for dep in dependencies:

		if not os.path.isfile(dep):
			raise FileExistsError(f"Dependency '{dep}' not found.")

		if get_basename(dep) == get_basename(filename):
			continue

		if dep.endswith(".h"):
			dep = get_basename(dep) + ".cpp"

		compiled_dependency |= compile_file(dep, object_file=True, report=report)

		dependency_files.append(get_basename(dep) + ".o")

	if not compiled_dependency and not file_has_modified:
		report(f"✅ '{filename}' is up to date. Skipping compilation.")
		return False 

	output_file = basename + ".o" * object_file
	command = f"{COMPILER} {FLAGS} {'-c' * object_file} -o {output_file} {' '.join(dependency_files)}"

	report(f"⚙️ Running command: {command}")
	subprocess.run(command, shell=True, check=True)

	if object_file:
		report(f"Compiled '{filename}' to '{output_file}'.")
	else:
		report(f"Compiled '{filename}' to executable '{output_file}'.")

	return True

def main(argv: list[str]) -> int:

	if len(argv) < 2:
		print(ValueError("Usage: run.py <filename> [flags]"))
		return 1

	filename = argv[1]
	object_file = "-c" in argv[2:]
	silent = "-s" in argv[2:]
	report = (lambda *args: None) if silent else print

	compile_file(filename, object_file, report)

	report("Compilation successful.\n")

	return 0
	
if __name__ == "__main__":
	exit(main(sys.argv))