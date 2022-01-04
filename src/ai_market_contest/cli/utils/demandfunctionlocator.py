from ast import literal_eval
import pathlib
from configparser import ConfigParser
from typing import List
from ai_market_contest.cli.cli_config import CONFIG_FILENAME, CUR_DEMAND_FUNCTIONS
from ai_market_contest.demand_function import DemandFunction


class DemandFunctionLocator:
    def __init__(self, env_dir: pathlib.Path):
        self.env_dir = env_dir

    def _get_demand_functions(self) -> List[str]:
        config_parser: ConfigParser = ConfigParser()
        config_parser.read(self.env_dir / CONFIG_FILENAME)
        demand_functions: List[str] = literal_eval(
            config_parser["environment"]["demand functions"]
        )
        return demand_functions

    def get_demand_function(self, demand_function_name: str) -> DemandFunction:
        if demand_function_name in CUR_DEMAND_FUNCTIONS:
            return CUR_DEMAND_FUNCTIONS[demand_function_name]
