# expressvpn_explorer/models/bandit.py

import numpy as np


class MultiArmedBandit:
    """Class for implementing a Multi-Armed Bandit algorithm."""

    def __init__(self, arm_labels):
        """
        Initialize the MultiArmedBandit instance.

        Parameters:
        - arm_labels (list): List of labels for each arm.
        - action_values (dict): Dictionary mapping arm labels to their rewards.
        """
        self.arm_labels = arm_labels
        self.action_counts = {label: 0 for label in arm_labels}
        self.action_values = {label: 0.0 for label in arm_labels}
        self.active_arm = {label: 1 for label in arm_labels}
        self.timestep = 0

    def activate(self, arm_labels):
        """
        Activate a specific arm label in the MultiArmedBandit instance.
        First deselect all and then select the ones in arms_labels.
        """

        self.active_arm = {label: 0 for label in self.arm_labels}

        for arm in arm_labels:
            if arm not in self.arm_labels:
                self.add_arm(arm)

            self.active_arm[arm] = 1

    def add_arm(self, arm_label):
        """
        Add a new arm to the bandit.

        Parameters:
        - arm_label (str): Label for the new arm.
        - true_reward (float): True reward for the new arm.
        """
        self.arm_labels.append(arm_label)
        self.action_counts[arm_label] = 0
        self.action_values[arm_label] = 0
        self.active_arm[arm_label] = 0

    def select_action(self):
        """
        Select the action with the highest UCB value.

        Returns:
        - str: Selected action label.
        """

        max_ucb_value = -10000
        action = None

        for label, value in self.action_values.items():
            if self.active_arm[label] == 1:
                this_value = value + np.sqrt(5 * np.log(self.timestep + 1) / (self.action_counts[label] + 1e-6))
                if this_value > max_ucb_value:
                    max_ucb_value = this_value
                    action = label

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
