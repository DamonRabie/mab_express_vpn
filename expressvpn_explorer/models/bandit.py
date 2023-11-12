# expressvpn_explorer/models/bandit.py

import logging

import numpy as np


class MultiArmedBandit:
    """Class for implementing a Multi-Armed Bandit algorithm."""

    def __init__(self, arm_labels, true_rewards):
        """
        Initialize the MultiArmedBandit instance.

        Parameters:
        - arm_labels (list): List of labels for each arm.
        - true_rewards (dict): Dictionary mapping arm labels to their true rewards.
        """
        self.arm_labels = arm_labels
        self.num_arms = len(arm_labels)
        self.true_rewards = true_rewards
        self.action_counts = {label: 0 for label in arm_labels}
        self.action_values = {label: 0.0 for label in arm_labels}
        self.timestep = 0

    def add_arm(self, arm_label, true_reward):
        """
        Add a new arm to the bandit.

        Parameters:
        - arm_label (str): Label for the new arm.
        - true_reward (float): True reward for the new arm.
        """
        if arm_label not in self.arm_labels:
            self.arm_labels.append(arm_label)
            self.true_rewards[arm_label] = true_reward
            self.action_counts[arm_label] = 0
            self.action_values[arm_label] = 0.0
        else:
            logging.warning(f"Arm '{arm_label}' already exists in the bandit.")

    def select_action(self):
        """
        Select the action with the highest UCB value.

        Returns:
        - str: Selected action label.
        """
        ucb_values = {
            label: value + np.sqrt(2 * np.log(self.timestep + 1) / (self.action_counts[label] + 1e-6))
            for label, value in self.action_values.items()
        }

        action = max(ucb_values, key=ucb_values.get)
        return action

    def update(self, action, reward):
        """
        Update the bandit based on the result of the selected action.

        Parameters:
        - action (str): Selected action label.
        """

        self.action_counts[action] += 1
        self.timestep += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[action]
