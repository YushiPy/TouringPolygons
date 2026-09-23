#!/usr/bin/env python3
"""Serve the static SIICUSP34 directory for local or same-Wi-Fi preview."""

from __future__ import annotations

import argparse
import errno
import ipaddress
import re
import shutil
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


APP = Path(__file__).resolve().parents[1]


def lan_addresses() -> list[tuple[str, str]]:
    """List private IPv4 addresses, preferring the usual macOS Wi-Fi interfaces."""
    if not shutil.which("ifconfig"):
        return []
    result = subprocess.run(["ifconfig", "-a"], capture_output=True, text=True, check=False)
    if result.returncode:
        return []
    found: list[tuple[str, str]] = []
    interface = ""
    for line in result.stdout.splitlines():
        if line and not line[0].isspace():
            interface = line.split(":", 1)[0]
        match = re.search(r"\binet (\d+\.\d+\.\d+\.\d+)\b", line)
        if not match:
            continue
        address = ipaddress.IPv4Address(match.group(1))
        if address.is_private and not address.is_loopback and not address.is_link_local:
            found.append((interface, str(address)))
    return sorted(set(found), key=lambda item: (item[0] not in {"en0", "en1"}, item[0], item[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone", action="store_true", help="aceitar conexões do celular na mesma rede")
    parser.add_argument("--port", type=int, default=8765, help="porta local (padrão: 8765)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("a porta deve estar entre 1 e 65535")

    host = "0.0.0.0" if args.phone else "127.0.0.1"
    handler = partial(SimpleHTTPRequestHandler, directory=str(APP))
    try:
        server = ThreadingHTTPServer((host, args.port), handler)
    except OSError as error:
        hint = " Experimente --port 8766." if error.errno == errno.EADDRINUSE else ""
        parser.error(f"não foi possível abrir a porta {args.port}: {error}.{hint}")

    print(f"Computador: http://127.0.0.1:{args.port}/", flush=True)
    if args.phone:
        addresses = lan_addresses()
        for interface, address in addresses:
            print(f"Celular ({interface}): http://{address}:{args.port}/", flush=True)
        if not addresses:
            print("Não foi possível detectar o IP da rede. Consulte o IPv4 do Wi-Fi e use http://IP:PORTA/.", flush=True)
        print("Conecte o celular à mesma rede Wi-Fi. Pressione Ctrl+C para encerrar.", flush=True)
    else:
        print("Pressione Ctrl+C para encerrar.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPrévia encerrada.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
