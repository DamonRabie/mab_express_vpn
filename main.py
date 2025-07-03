# main.py
import logging
import os
import pickle
from logging.handlers import RotatingFileHandler

from expressvpn_explorer.connector.expressvpn import VPNConnector
from expressvpn_explorer.models.bandit import MultiArmedBandit
from expressvpn_explorer.utils import check_internet_connection

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "explorer_mab_model.pkl")
LOG_FILE = 'vpn_explorer.log'


def initialize_logging():
    """Initialize logging configuration to overwrite log file each run."""
    # Ensure the log directory exists (if using a subdirectory)
    os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)

    # Clear any existing log handlers
    logging.root.handlers = []

    # Set up handlers
    handlers = [
        RotatingFileHandler(
            LOG_FILE,
            mode='a',  # Changed from 'w' to 'a' (append)
            maxBytes=5 * 1024 * 1024,
            backupCount=0
        ),
        logging.StreamHandler()
    ]

    # Basic config with handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logging.info("Initialized logging")


def save_model(model: MultiArmedBandit):
    """Save the MultiArmedBandit model to a file."""
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        with open(MODEL_FILE, "wb") as file:
            pickle.dump(model, file)
            logging.info("MultiArmedBandit model saved to file.")
    except Exception as e:
        logging.error(f"Error while saving model: {e}")


def load_model(server_list: list) -> MultiArmedBandit:
    """Load the MultiArmedBandit model from a file or create a new one."""
    try:
        if os.path.exists(MODEL_FILE):
            logging.info("Loading existing MultiArmedBandit model from file.")
            with open(MODEL_FILE, "rb") as file:
                model = pickle.load(file)

            model.activate(server_list)
            return model

        else:
            logging.info("Creating a new MultiArmedBandit model.")
            return MultiArmedBandit(server_list)

    except Exception as e:
        logging.error(f"Error while loading model: {e}")
        raise f"Error while loading model: {e}"


def run_bandit_until_success(
        bandit: MultiArmedBandit,
        vpn_connector: VPNConnector = VPNConnector(),
        max_checks: int = 200
):
    """Run the bandit algorithm until a successful connection is made."""
    for _ in range(max_checks):
        action = bandit.select_action()
        logging.info(f"Selected arm: {action}")

        vpn_connector.connect(action)
        if vpn_connector.is_connected():
            if check_internet_connection():
                bandit.update(action, 3)
                logging.info(f"Successfully connected to VPN using server: {action}")
                break
            else:
                bandit.update(action, 1)
                logging.info(f"Successfully connected but no internet connection using server: {action}")
        else:
            bandit.update(action, 0)

        save_model(bandit)
    else:
        logging.warning("No successful arm found after 200 checks.")


if __name__ == "__main__":
    initialize_logging()

    express_vpn = VPNConnector()
    all_servers = express_vpn.fetch_server_list()

    bandit_model = load_model(all_servers)

    run_bandit_until_success(bandit_model, vpn_connector=express_vpn, max_checks=1000)
