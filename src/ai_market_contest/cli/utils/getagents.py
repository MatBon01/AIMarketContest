import ast
import configparser
import pathlib
import sys

from ai_market_contest.cli.cli_config import (  # type: ignore
    CONFIG_FILENAME,
    TRAINING_CONFIGS_DIR_NAME,
)
from ai_market_contest.cli.utils.filesystemutils import check_config_file_exists


def get_agent_names(proj_dir: pathlib.Path) -> list[str]:
    config: configparser.ConfigParser = configparser.ConfigParser()
    config_file: pathlib.Path = proj_dir / CONFIG_FILENAME
    check_config_file_exists(config_file)
    config.read(config_file)
    try:
        agents: list[str] = ast.literal_eval(config["agent"]["agents"])
    except KeyError:
        print("Error: config file needs an agents attribute")
        sys.exit(1)
    return agents


def get_trained_agents(agent_dir: pathlib.Path) -> list[str]:
    agent_config_file: pathlib.Path = agent_dir / CONFIG_FILENAME
    config: configparser.ConfigParser = configparser.ConfigParser()
    check_config_file_exists(agent_config_file)
    config.read(agent_config_file)
    try:
        trained_agents: list[str] = ast.literal_eval(
            config["training"]["trained-agents"]
        )
    except KeyError:
        print("Error: Config file needs a trained-agents attribute")
        sys.exit(1)
    return trained_agents


def add_trained_agent_to_config_file(agent_dir: pathlib.Path, trained_agent_name: str):
    config_file: pathlib.Path = agent_dir / CONFIG_FILENAME
    check_config_file_exists(config_file)
    config: configparser.ConfigParser = configparser.ConfigParser()
    config.read(config_file)
    try:
        trained_agents: list[str] = ast.literal_eval(
            config["training"]["trained-agents"]
        )
    except KeyError:
        print("Error: Config file needs a trained-agents attribute")
        sys.exit(1)

    trained_agents.append(trained_agent_name)
    config["training"]["trained-agents"] = str(trained_agents)
    with config_file.open("w") as c_file:
        config.write(c_file)


def get_training_configs(proj_dir: pathlib.Path):
    training_configs_dir = proj_dir / TRAINING_CONFIGS_DIR_NAME
    training_configs = []
    for training_config_file in training_configs_dir.rglob("*.py"):
        training_configs.append(training_config_file.stem)
    return training_configs