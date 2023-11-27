# expressvpn_explorer/connector/expressvpn.py

import logging
import subprocess

from ..utils import check_internet_connection


class VPNConnector:
    """Class for handling VPN connections."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def disconnect(self):
        """Disconnect from the current VPN server."""
        try:
            subprocess.run("expressvpn disconnect", shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                # Non-zero exit status 1 is expected, treat it as a normal behavior
                self.logger.info("Disconnected from the current VPN server.")
            else:
                self.logger.error(f"Error while disconnecting: {e}")

    def connect(self, server_label):
        """Connect to the specified VPN server."""
        self.disconnect()
        connect_command = f"expressvpn connect {server_label}"
        try:
            subprocess.run(connect_command, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error while connecting to {server_label}: {e}")

    def is_connected(self):
        """Check if the VPN connection is successful."""
        try:
            output = subprocess.run("expressvpn status", shell=True, check=True, capture_output=True)
            return 'connected to' in str(output.stdout.lower()) and check_internet_connection()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error while checking connection status: {e}")
            return False

    def fetch_server_list(self):
        """Fetch the list of available VPN servers."""
        try:
            output = subprocess.run(["expressvpn list all | awk '{print $1}' | sed -n '3,$p'"], shell=True,
                                    capture_output=True, text=True)
            server_list = output.stdout.strip().split('\n')
            server_list = [item for item in server_list if item not in ["", 'smart']]
            return server_list
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error while fetching server list: {e}")
            return []
