from __future__ import annotations

import argparse
import ipaddress
import socket
import subprocess
import sys
from pathlib import Path

import uvicorn

APP_ROOT = Path(__file__).resolve().parents[1]


def lan_addresses() -> list[str]:
    addresses = set()
    if sys.platform == "darwin":
        for interface in ("en0", "en1"):
            result = subprocess.run(
                ["/usr/sbin/ipconfig", "getifaddr", interface], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                addresses.add(result.stdout.strip())
    else:
        try:
            addresses.update(item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
        except socket.gaierror:
            pass
    return sorted(address for address in addresses if not ipaddress.ip_address(address).is_loopback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prévia do SIICUSP acessível na rede local.")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--no-reload", action="store_true", help="Desativar recarga automática ao editar Python.")
    args = parser.parse_args()
    print(f"\nNo Mac: http://127.0.0.1:{args.port}/evento", flush=True)
    for address in lan_addresses():
        print(f"No celular, na mesma rede Wi-Fi: http://{address}:{args.port}/evento", flush=True)
    print("Use o IP do Mac, não localhost, no celular. Mantenha este terminal aberto.\n", flush=True)
    sys.path.insert(0, str(APP_ROOT))
    uvicorn.run(
        "event_server:app",
        host="0.0.0.0",
        port=args.port,
        reload=not args.no_reload,
        reload_dirs=[str(APP_ROOT)] if not args.no_reload else None,
    )


if __name__ == "__main__":
    main()
