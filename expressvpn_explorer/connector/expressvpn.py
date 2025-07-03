import json
import logging
import platform
import subprocess
import sys
import time


class VPNConnector:
    """Class for handling VPN connections across macOS and Ubuntu."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.is_mac = platform.system() == 'Darwin'
        self.is_linux = platform.system() == 'Linux'

        if self.is_mac:
            self.logger.info("Initializing for macOS (AppleScript)")
        elif self.is_linux:
            self.logger.info("Initializing for Linux (expressvpn CLI)")
            self._verify_linux_cli()
        else:
            self.logger.error("Unsupported operating system")
            sys.exit(1)

    def _verify_linux_cli(self):
        """Verify expressvpn CLI is installed on Linux."""
        try:
            subprocess.run(["expressvpn", "--version"],
                           check=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("expressvpn CLI not found. Please install it first.")
            sys.exit(1)

    def _run_applescript(self, script):
        """Run AppleScript commands on macOS."""
        try:
            process = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                check=True
            )
            return process.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"AppleScript error: {e.stderr}")
            return None

    def _run_linux_command(self, command):
        """Run expressvpn commands on Linux."""
        try:
            process = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                check=True
            )
            return process.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e.stderr}")
            return None

    def disconnect(self):
        """Disconnect from the current VPN server."""
        if self.is_mac:
            script = '''
            tell application "ExpressVPN"
                activate
                disconnect
            end tell
            '''
            result = self._run_applescript(script)
        else:
            result = self._run_linux_command("expressvpn disconnect")

        if result is not None:
            self.logger.info("Disconnected from VPN")
        else:
            self.logger.error("Failed to disconnect")

    def connect(self, server_label):
        """Connect to the specified VPN server."""
        self.disconnect()
        time.sleep(2)  # Give time for disconnection

        if self.is_mac:
            script = f'''
            tell application "ExpressVPN"
                activate
                delay 1
                connect "{server_label}"
            end tell
            '''
            result = self._run_applescript(script)
        else:
            result = self._run_linux_command(f"expressvpn connect {server_label}")

        if result is not None:
            self.logger.info(f"Connection attempt to {server_label} started")
            time.sleep(10)
            if self.is_connected():
                self.logger.info(f"Successfully connected to {server_label}")
            else:
                self.logger.warning(f"Connection to {server_label} may have failed")
        else:
            self.logger.error(f"Failed to connect to {server_label}")

    def is_connected(self):
        """Check if the VPN connection is successful."""
        if self.is_mac:
            script = '''
            tell application "ExpressVPN"
                if state is "connected" then
                    return "connected"
                else
                    return "not connected"
                end if
            end tell
            '''
            result = self._run_applescript(script)
            return result == "connected"
        else:
            result = self._run_linux_command("expressvpn status")
            return result and 'Connected to' in result

    def fetch_server_list(self):
        """Fetch the list of available VPN servers."""
        if self.is_mac:
            # For macOS, we need to maintain a manual list
            with open('server_list.json', 'r') as f:
                servers = json.load(f)
            return servers
        else:
            # For Linux, we can get the list from CLI
            try:
                output = subprocess.run(
                    ["expressvpn", "list", "all"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                servers = []
                for line in output.stdout.split('\n')[3:]:  # Skip header lines
                    if line.strip() and not line.startswith('Smart location:'):
                        server = line.split()[0]
                        if server not in ["", "smart"]:
                            servers.append(server)
                return servers
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error fetching server list: {e.stderr}")
                return []
