# main.py
import logging
import os
import pickle

import numpy as np

from expressvpn_explorer.connector.expressvpn import VPNConnector
from expressvpn_explorer.models.bandit import MultiArmedBandit

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "explorer_mab_model.pkl")


def initialize_logging():
    """Initialize logging configuration."""
    logging.basicConfig(level=logging.INFO)


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


def load_model(vpn_connector: VPNConnector) -> MultiArmedBandit:
    """Load the MultiArmedBandit model from a file or create a new one."""
    try:
        if os.path.exists(MODEL_FILE):
            logging.info("Loading existing MultiArmedBandit model from file.")
            with open(MODEL_FILE, "rb") as file:
                return pickle.load(file)
        else:
            logging.info("Creating a new MultiArmedBandit model.")
            arm_labels = vpn_connector.fetch_server_list()
            true_rewards = dict(zip(arm_labels, np.zeros(len(arm_labels))))
            return MultiArmedBandit(arm_labels, true_rewards)
    except Exception as e:
        logging.error(f"Error while loading model: {e}")
        return MultiArmedBandit([], {})


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
            bandit.update(action, 1)
            logging.info(f"Successfully connected to VPN using server: {action}")
            break
        else:
            bandit.update(action, 0)

        save_model(bandit)
    else:
        logging.warning("No successful arm found after 200 checks.")


if __name__ == "__main__":
    initialize_logging()

    expressvpn = VPNConnector()
    bandit_model = load_model(expressvpn)

    run_bandit_until_success(bandit_model, vpn_connector=expressvpn)
