import pathlib
from ast import literal_eval
from configparser import ConfigParser
from typing import Any

from ai_market_contest.cli.cli_config import CONFIG_FILENAME
from ai_market_contest.cli.configs.simulation_config_reader import (
    SimulationConfigReader,
)


class TrainingConfigReader(SimulationConfigReader):
    parsed_config: ConfigParser

    def get_self_play_num(self) -> int:
        return int(self.parsed_config["General"]["number_of_self_play_agents"])

    def get_num_agents(self) -> int:
        return self.get_self_play_num() + sum(self.get_naive_agent_counts().values())

    def get_other_config(self) -> dict[str, Any]:
        return {
            setting_name: literal_eval(setting_value)
            for setting_name, setting_value in self.parsed_config["Other"].items()
        }

    def print_training(self) -> bool:
        # TODO add default value
        return bool(self.parsed_config["General"]["print_training"])

    def get_num_epochs(self) -> int:
        # TODO add default value
        return int(self.parsed_config["General"]["epochs"])

    def write_config_to_file(self, new_agent_dir: pathlib.Path):
        config_file: pathlib.Path = new_agent_dir / CONFIG_FILENAME
        with config_file.open("w") as cfg_file:
            self.parsed_config.write(cfg_file)