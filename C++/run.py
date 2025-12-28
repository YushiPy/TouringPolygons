
import os
import re
import sys


STANDARD = "c++26"

def is_up_to_date(source: str, target: str) -> bool:
	
	if not os.path.exists(target):
		return False
	
	source_mod_time = os.path.getmtime(source)
	target_mod_time = os.path.getmtime(target)

	return target_mod_time >= source_mod_time

def compile_cpp(file: str, executable: bool = True) -> None:

	if not file.endswith(".cpp"):
		file += ".cpp"

	if not os.path.exists(file):
		raise FileNotFoundError(f"File '{file}' not found.")

	print(f"Compiling {file}...")

	target = file.replace(".cpp", "" if executable else ".o")

	if is_up_to_date(file, target):
	
		if executable:
			print("Executable is up to date. Skipping compilation.")
		else:
			print("Object file is up to date. Skipping compilation.")
	
		return

	with open(file, "r") as f:
		code = f.read()

	code = re.sub(r"//.*?$|/\*.*?\*/", "", code, flags=re.DOTALL | re.MULTILINE)
	imports: list[str] = re.findall(r"#include\s*\"([\w\d\_\.\+\-\/]+)\"", code)

	if imports:
		print(f"Compiling dependencies: {', '.join(imports)}")

	object_files: list[str] = []

	for imp in imports:
		compile_cpp(imp.replace(".h", ".cpp"))
		object_files.append(imp.replace(".h", ".o"))
	
	if executable:
		cmd = f"g++ -std={STANDARD} {file} -o {file.replace('.cpp', '')} {' '.join(object_files)}"
	else:
		cmd = f"g++ -std={STANDARD} -c {file} -o {file.replace('.cpp', '.o')} {' '.join(object_files)}"

	print(f"Running command: {cmd}")
	os.system(cmd)

def main(args: list[str]) -> int:

	flags = [name for name in args if name.startswith("-")]
	file = next((name for name in args if not name.startswith("-")), "")

	if not file:
		print("No file provided.")
		return 1
	
	if not file.endswith(".cpp"):
		file += ".cpp"
	
	if is_up_to_date(file, file.replace(".cpp", "")):
		print("Executable is up to date. No compilation needed.")
		return 0
	
	compile_cpp(file)


if __name__ == "__main__":
	exit(main(sys.argv[1:]))
