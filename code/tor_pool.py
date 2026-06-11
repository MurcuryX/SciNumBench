"""
tor_pool.py — Tor multi-instance proxy pool manager

Creates multiple Tor instances, each providing an independent SOCKS5 proxy port.
Supports parallel crawling with automatic IP rotation.

Usage:
  # Start the proxy pool (default 5 instances)
  python tor_pool.py start --instances 5

  # Stop the proxy pool
  python tor_pool.py stop

  # Test proxies
  python tor_pool.py test
"""

import os
import sys
import time
import signal
import subprocess
import socket
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

# Tor executable path (Linux Expert Bundle preferred, Windows compatible)
TOR_HOME = os.environ.get("TOR_HOME", os.path.expanduser("~/tor-expert"))
TOR_LIB_DIR = os.path.join(TOR_HOME, "tor")   # contains tor binary + libssl/libcrypto/libevent
TOR_GEOIP = os.path.join(TOR_HOME, "data", "geoip")
TOR_GEOIP6 = os.path.join(TOR_HOME, "data", "geoip6")

TOR_EXE_PATHS = [
    os.environ.get("TOR_EXE", ""),            # explicit override takes priority
    os.path.join(TOR_LIB_DIR, "tor"),         # Tor Expert Bundle (Linux / server0)
    r"C:\Tor\Tor\tor.exe",
    r"C:\Program Files\Tor\tor.exe",
    r"C:\Program Files (x86)\Tor\tor.exe",
    r"C:\Users\{}\AppData\Local\Tor\tor.exe".format(os.environ.get("USERNAME", "")),
    "tor",  # if available in PATH
]

# Proxy pool configuration
DEFAULT_INSTANCES = 5          # default number of Tor instances
BASE_SOCKS_PORT = 9050         # starting SOCKS5 port
BASE_CONTROL_PORT = 9150       # starting control port
DATA_DIR = os.environ.get("TOR_POOL_DIR", os.path.join(TOR_HOME, "pool"))  # Tor data directory (absolute path)
STARTUP_TIMEOUT = 60           # startup timeout (seconds)

# Verification URL (httpbin is often unreachable via Tor exits; use Tor's official check endpoint)
CHECK_IP_URL = "https://check.torproject.org/api/ip"
CHECK_IP_TIMEOUT = 30


# ============================================================
# Data structures
# ============================================================

@dataclass
class TorInstance:
    """A single Tor instance."""
    instance_id: int
    socks_port: int
    control_port: int
    data_dir: str
    process: Optional[subprocess.Popen] = None
    is_ready: bool = False
    ip_address: str = ""


class TorPool:
    """Tor multi-instance proxy pool."""

    def __init__(
        self,
        num_instances: int = DEFAULT_INSTANCES,
        base_socks_port: int = BASE_SOCKS_PORT,
        base_control_port: int = BASE_CONTROL_PORT,
        tor_exe: str = None,
    ):
        self.num_instances = num_instances
        self.base_socks_port = base_socks_port
        self.base_control_port = base_control_port
        self.instances: List[TorInstance] = []
        self.current_index = 0
        self.lock = threading.Lock()

        # Locate the Tor executable
        self.tor_exe = tor_exe or self._find_tor_exe()
        if not self.tor_exe:
            raise FileNotFoundError(
                "Tor executable not found. Please install Tor or specify its path.\n"
                "Download: https://www.torproject.org/download/tor/"
            )

        print(f"[TOR] Tor path: {self.tor_exe}")

    def _find_tor_exe(self) -> Optional[str]:
        """Locate the Tor executable."""
        for path in TOR_EXE_PATHS:
            if not path:
                continue
            if os.path.isfile(path):
                return path
            # Check whether it is in PATH
            try:
                result = subprocess.run(
                    ["where", path] if os.name == "nt" else ["which", path],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip().split("\n")[0]
            except:
                pass
        return None

    def _create_torrc(self, instance: TorInstance) -> str:
        """Generate the torrc configuration file."""
        os.makedirs(instance.data_dir, exist_ok=True)

        geoip_lines = ""
        if os.path.isfile(TOR_GEOIP):
            geoip_lines += f"GeoIPFile {TOR_GEOIP}\n"
        if os.path.isfile(TOR_GEOIP6):
            geoip_lines += f"GeoIPv6File {TOR_GEOIP6}\n"

        torrc_content = f"""# Tor instance {instance.instance_id}
SocksPort {instance.socks_port}
ControlPort {instance.control_port}
DataDirectory {instance.data_dir}
Log notice file {os.path.join(instance.data_dir, "tor.log")}
{geoip_lines}
# Performance tuning
NumEntryGuards 3
CircuitBuildTimeout 60
LearnCircuitBuildTimeout 0
NewCircuitPeriod 120
MaxCircuitDirtiness 300

# Avoid being banned
ExitPolicy reject *:*
"""
        torrc_path = os.path.join(instance.data_dir, "torrc")
        with open(torrc_path, "w", encoding="utf-8") as f:
            f.write(torrc_content)

        return torrc_path

    def _start_instance(self, instance: TorInstance) -> bool:
        """Start a single Tor instance."""
        torrc_path = self._create_torrc(instance)

        try:
            # Start the Tor process (Expert Bundle needs LD_LIBRARY_PATH to find libssl/libcrypto/libevent)
            env = os.environ.copy()
            if os.path.isdir(TOR_LIB_DIR):
                env["LD_LIBRARY_PATH"] = TOR_LIB_DIR + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            process = subprocess.Popen(
                [self.tor_exe, "-f", torrc_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            instance.process = process

            # Wait for startup
            start_time = time.time()
            while time.time() - start_time < STARTUP_TIMEOUT:
                # Check whether the process is still running
                if process.poll() is not None:
                    stderr = process.stderr.read().decode() if process.stderr else ""
                    print(f"[TOR] Instance {instance.instance_id} failed to start: {stderr[:200]}")
                    return False

                # Check whether the SOCKS port is available
                if self._check_port(instance.socks_port):
                    instance.is_ready = True
                    return True

                time.sleep(0.5)

            print(f"[TOR] Instance {instance.instance_id} startup timed out")
            return False

        except Exception as e:
            print(f"[TOR] Instance {instance.instance_id} startup error: {e}")
            return False

    def _check_port(self, port: int) -> bool:
        """Check whether a port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            return result == 0
        except:
            return False

    def start(self) -> int:
        """Start the proxy pool. Returns the number of successfully started instances."""
        print(f"[TOR] Starting {self.num_instances} Tor instances...")

        # Create instances
        for i in range(self.num_instances):
            instance = TorInstance(
                instance_id=i,
                socks_port=self.base_socks_port + i,
                control_port=self.base_control_port + i,
                data_dir=os.path.join(DATA_DIR, f"instance_{i}"),
            )
            self.instances.append(instance)

        # Start in parallel
        threads = []
        for instance in self.instances:
            thread = threading.Thread(target=self._start_instance, args=(instance,))
            thread.start()
            threads.append(thread)

        # Wait for all to finish starting
        for thread in threads:
            thread.join(timeout=STARTUP_TIMEOUT + 5)

        # Tally results
        ready_count = sum(1 for inst in self.instances if inst.is_ready)
        print(f"[TOR] Startup complete: {ready_count}/{self.num_instances} instances ready")

        return ready_count

    def stop(self):
        """Stop all Tor instances."""
        for instance in self.instances:
            if instance.process and instance.process.poll() is None:
                try:
                    instance.process.terminate()
                    instance.process.wait(timeout=5)
                except:
                    instance.process.kill()

        self.instances.clear()

    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get the next available proxy."""
        with self.lock:
            # Round-robin selection
            for _ in range(len(self.instances)):
                instance = self.instances[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.instances)

                if instance.is_ready:
                    proxy_url = f"socks5h://127.0.0.1:{instance.socks_port}"
                    return {
                        "http": proxy_url,
                        "https": proxy_url,
                    }

        return None

    def get_all_proxies(self) -> List[Dict[str, str]]:
        """Get all available proxies."""
        proxies = []
        for instance in self.instances:
            if instance.is_ready:
                proxy_url = f"socks5h://127.0.0.1:{instance.socks_port}"
                proxies.append({
                    "http": proxy_url,
                    "https": proxy_url,
                })
        return proxies

    def check_ip(self, proxy: Dict[str, str] = None) -> Optional[str]:
        """Check the current IP address."""
        try:
            resp = requests.get(
                CHECK_IP_URL,
                proxies=proxy,
                timeout=CHECK_IP_TIMEOUT,
            )
            data = resp.json()
            # check.torproject.org returns {"IsTor":bool,"IP":"..."}; httpbin returns {"origin":"..."}
            return data.get("IP") or data.get("origin", "unknown")
        except Exception as e:
            return None

    def test_all(self):
        """Test all proxies."""
        for i, instance in enumerate(self.instances):
            if not instance.is_ready:
                print(f"  Instance {i}: not ready")
                continue

            proxy_url = f"socks5h://127.0.0.1:{instance.socks_port}"
            proxy = {"http": proxy_url, "https": proxy_url}

            ip = self.check_ip(proxy)
            if ip:
                instance.ip_address = ip
                print(f"  Instance {i}: IP = {ip}")
            else:
                print(f"  Instance {i}: connection failed")

    def get_status(self) -> List[Dict]:
        """Get the status of all instances."""
        status = []
        for instance in self.instances:
            status.append({
                "id": instance.instance_id,
                "socks_port": instance.socks_port,
                "is_ready": instance.is_ready,
                "ip": instance.ip_address,
            })
        return status


# ============================================================
# Global proxy pool instance
# ============================================================

_global_pool: Optional[TorPool] = None


def get_tor_pool(num_instances: int = DEFAULT_INSTANCES) -> TorPool:
    """Get the global proxy pool instance (singleton pattern)."""
    global _global_pool
    if _global_pool is None:
        _global_pool = TorPool(num_instances=num_instances)
        _global_pool.start()
    return _global_pool


def get_tor_proxy() -> Optional[Dict[str, str]]:
    """Get a single Tor proxy."""
    pool = get_tor_pool()
    return pool.get_proxy()


# ============================================================
# CLI entry point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tor multi-instance proxy pool manager")
    subparsers = parser.add_subparsers(dest="command", help="subcommand")

    # start
    start_parser = subparsers.add_parser("start", help="start the proxy pool")
    start_parser.add_argument("--instances", type=int, default=DEFAULT_INSTANCES, help="number of instances")
    start_parser.add_argument("--base-port", type=int, default=BASE_SOCKS_PORT, help="starting port")
    start_parser.add_argument("--tor", type=str, default=None, help="Tor executable path")

    # stop
    subparsers.add_parser("stop", help="stop the proxy pool")

    # test
    test_parser = subparsers.add_parser("test", help="test proxies")
    test_parser.add_argument("--instances", type=int, default=DEFAULT_INSTANCES, help="number of instances")

    # status
    subparsers.add_parser("status", help="view status")

    args = parser.parse_args()

    if args.command == "start":
        pool = TorPool(
            num_instances=args.instances,
            base_socks_port=args.base_port,
            tor_exe=args.tor,
        )
        pool.start()

        print("\nProxy pool started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pool.stop()

    elif args.command == "stop":
        # Send stop signal (simplified implementation)
        print("[TOR] Please use Ctrl+C to stop the proxy pool")

    elif args.command == "test":
        pool = TorPool(num_instances=args.instances)
        pool.start()
        pool.test_all()
        pool.stop()

    elif args.command == "status":
        print("[TOR] View status (the proxy pool must be started first)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
