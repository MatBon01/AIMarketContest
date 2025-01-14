import ast
import configparser
import pathlib
import sys

from ai_market_contest.cli.cli_config import CONFIG_FILENAME
from ai_market_contest.cli.utils.error_codes import ErrorCodes  # type: ignore
from ai_market_contest.cli.utils.filesystemutils import assert_config_file_exists


def get_agent_names(proj_dir: pathlib.Path) -> list[str]:
    config: configparser.ConfigParser = configparser.ConfigParser()
    config_file: pathlib.Path = proj_dir / CONFIG_FILENAME
    assert_config_file_exists(config_file)
    config.read(config_file)
    try:
        agents: list[str] = ast.literal_eval(config["agent"]["agents"])
    except KeyError:
        print("Error: config file needs an agents attribute")
        sys.exit(ErrorCodes.NO_AGENTS_ATTRIBUTE)
    return agents


def get_trained_agents(agent_dir: pathlib.Path) -> list[str]:
    agent_config_file: pathlib.Path = agent_dir / CONFIG_FILENAME
    config: configparser.ConfigParser = configparser.ConfigParser()
    assert_config_file_exists(agent_config_file)
    config.read(agent_config_file)
    try:
        trained_agents: list[str] = ast.literal_eval(
            config["training"]["trained-agents"]
        )
    except KeyError:
        print("Error: Config file needs a trained-agents attribute")
        sys.exit(ErrorCodes.NO_TRAINED_AGENTS_ATTRIBUTE)
    return trained_agents


def add_trained_agent_to_config_file(agent_dir: pathlib.Path, trained_agent_name: str):
    config_file: pathlib.Path = agent_dir / CONFIG_FILENAME
    assert_config_file_exists(config_file)
    config: configparser.ConfigParser = configparser.ConfigParser()
    config.read(config_file)
    try:
        trained_agents: list[str] = ast.literal_eval(
            config["training"]["trained-agents"]
        )
    except KeyError:
        print("Error: Config file needs a trained-agents attribute")
        sys.exit(ErrorCodes.NO_TRAINED_AGENTS_ATTRIBUTE)

    trained_agents.append(trained_agent_name)
    config["training"]["trained-agents"] = str(trained_agents)
    with config_file.open("w") as c_file:
        config.write(c_file)
