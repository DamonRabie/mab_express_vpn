# ExpressVPN Bandit Explorer

ExpressVPN Bandit Explorer is a Python project that uses the Multi-Armed Bandit algorithm to intelligently select a VPN server for a stable and successful connection.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project implements a Multi-Armed Bandit algorithm to optimize the selection of VPN servers based on historical performance. The bandit explores different servers, learns from past experiences, and adapts its strategy to maximize the chances of a successful and stable VPN connection.

## Features

- Multi-Armed Bandit algorithm for intelligent VPN server selection.
- Integration with ExpressVPN for connecting to and disconnecting from servers.
- Logging for monitoring the bandit's decisions and VPN connection status.
- Persistence of bandit model to save and load the learning state.

## Getting Started

### Prerequisites

- Python 3.x
- ExpressVPN command-line tool installed and configured.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/DamonRabie/mab_express_vpn.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the `main.py` script:

    ```bash
    python main.py
    ```

2. The bandit algorithm will iteratively select VPN servers, attempt connections, and learn from the results until a successful connection is established.

## Configuration

- The project can be configured using command-line arguments or by modifying constants in the `main.py` script.
- Customize logging levels, file paths, and other parameters as needed.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── models             <- Store model pickle file
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         and a short `-` delimited description, e.g.
    │                         `1.0-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── expressvpn_explorer      <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── connector      
        │   └── expressvpn.py   <- VPNConnector class for handling VPN connections using ExpressVPN
        │
        ├── models
        │   └── bandit.py   <- Implementation of the Multi-Armed Bandit algorithm
        │
        ├── visualization  <- Scripts to create exploratory and results oriented visualizations
        │
        └── utils.py       <- General utility functions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
