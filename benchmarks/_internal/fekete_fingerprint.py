"""Fingerprint of everything the Fekete Python binding is built from.

Run with the Fekete virtual environment's Python (its ABI and prefix are part of the hash):
    python fekete_fingerprint.py FEKETE_SOURCE REVISION PROJECT_ROOT CONAN_PROFILE CMAKE_VERSION CONAN_VERSION
A matching marker lets ``free-compare`` skip the Conan and CMake build.
"""

import hashlib
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import sysconfig

root = pathlib.Path(sys.argv[1]).resolve()
revision = sys.argv[2]
project_root = pathlib.Path(sys.argv[3]).resolve()
conan_profile = sys.argv[4]
cmake_version = sys.argv[5]
conan_version = sys.argv[6]
digest = hashlib.sha256()
excluded = {".git", ".venv", ".conan", "_skbuild", "__pycache__", ".cache", "build", "dist"}
native_suffixes = {".cpp", ".cc", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp", ".tpp", ".cmake", ".txt"}
input_roots = (
	("fekete", root, True),
	("embedded-convex", project_root / "packages/convex-tpp/cpp", False),
	("embedded-geometry", project_root / "packages/common-geometry/cpp", False),
)
for label, source_root, include_metadata in input_roots:
	for directory, subdirectories, filenames in os.walk(source_root):
		subdirectories[:] = sorted(
			name for name in subdirectories
			if name not in excluded and not name.endswith(".egg-info")
		)
		for filename in sorted(filenames):
			path = pathlib.Path(directory) / filename
			relative = path.relative_to(source_root)
			under_conan_recipe = relative.parts[:2] == ("cmake", "conan")
			build_metadata = path.name in {"setup.py", "pyproject.toml", "conanfile.py", "conanfile.txt", "conan.lock", "conandata.yml"}
			if path.suffix.lower() not in native_suffixes and not (include_metadata and build_metadata) and not (include_metadata and under_conan_recipe):
				continue
			digest.update(label.encode())
			digest.update(b"/")
			digest.update(relative.as_posix().encode())
			digest.update(b"\0")
			digest.update(path.read_bytes())
			digest.update(b"\0")
configuration = {
	"revision": revision,
	"python": sys.version_info[:2],
	"python_abi": {key: sysconfig.get_config_var(key) for key in ("SOABI", "EXT_SUFFIX", "INCLUDEPY", "LIBDIR")},
	"python_prefix": sys.prefix,
	"platform": platform.system(),
	"machine": platform.machine(),
	"conan_profile": conan_profile,
	"cmake_version": cmake_version,
	"conan_version": conan_version,
	"environment": {key: os.environ.get(key, "") for key in (
		"CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CMAKE_ARGS",
		"CMAKE_BUILD_TYPE", "CMAKE_GENERATOR", "CMAKE_TOOLCHAIN_FILE", "CMAKE_PREFIX_PATH",
		"CMAKE_OSX_ARCHITECTURES", "MACOSX_DEPLOYMENT_TARGET",
	)},
}
configuration["tool_versions"] = {}
for name, command in (("c_compiler", os.environ.get("CC", "cc")),
		("cxx_compiler", os.environ.get("CXX", "c++"))):
	try:
		version = subprocess.run(shlex.split(command) + ["--version"], capture_output=True, text=True)
		configuration["tool_versions"][name] = (version.stdout + version.stderr).splitlines()[:2]
	except (OSError, ValueError):
		configuration["tool_versions"][name] = command
digest.update(json.dumps(configuration, sort_keys=True).encode())
print(digest.hexdigest())
