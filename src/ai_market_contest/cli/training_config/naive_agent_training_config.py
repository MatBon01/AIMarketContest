import configparser
import pathlib

from ai_market_contest.agent import Agent  # type: ignore
from ai_market_contest.cli.cli_config import TRAIN_CONFIG_FILENAME  # type: ignore
from ai_market_contest.cli.training_config.train_config import (
    TrainConfig,  # type: ignore
)
from ai_market_contest.demand_function import DemandFunction  # type: ignore
from ai_market_contest.demandfunctions.gaussian_demand_function import (
    GaussianDemandFunction,  # type: ignore
)
from ai_market_contest.environment import Environment  # type: ignore


class NaiveAgentTrainingConfig(TrainConfig):
    """
    Configuration class for agents to train via with naive implementations of agents.

    Attributes
    ----------
    agents: list[Agent]
        A list of naive agents to train the agent with.
    training_duration: int
        The duration for the simulation.
    demand_function: DemandFunction
        The demand function to be used by the environment during the simulation.
    """

    def __init__(
        self,
        agents: list[Agent] = [],
        training_duration: int = 100,
        demand_function: DemandFunction = GaussianDemandFunction(),
    ):
        """
        Parameters
        ----------
        agents: list[Agent], default=[]
            A list of naive agents to train the agent with.
        training_duration: int, default=100
            The duration for the simulation.
        demand_function: DemandFunction, default=GaussianDemandFunction()
            The demand function to be used by the environment during the simulation.
        """
        self.agents: list[Agent] = agents
        self.training_duration: int = training_duration
        self.demand_function: DemandFunction = demand_function

    def create_environment(self, agent_to_train: Agent) -> Environment:
        """
        Returns an environment set up for the training regimine.

        Parameters
        ----------
        agent_to_train : Agent
            The agent the training configuration is designed to train.

        Returns
        ----------
        Environment
            An Environment set up with the rules outlined by the configuration.
        """
        env: Environment = Environment(
            self.training_duration, self.demand_function, len(self.agents) + 1
        )  # max_agents has +1 to make room for the agent to be trained
        env.add_agent(agent_to_train)
        for agent in self.agents:
            env.add_agent(agent)

        return env

    def write_config_to_file(self, path: pathlib.Path):
        """
        Saves the configuration class as a configuration file.

        Parameters
        ----------
        path : pathlib.Path
            Where the configuration file should be written to.
        """
        config: configparser.ConfigParser = configparser.ConfigParser()
        config["training"] = {
            "type": "naive bots",
            "bots": ",".join(list(map(str, self.agents))),
        }
        config["environment"] = {
            "demand function": str(self.demand_function),
            "training duration": str(self.training_duration),
        }

        with open(path / TRAIN_CONFIG_FILENAME, "w") as f:
            config.write(f)

    def add_naive_agent(self, agent: Agent):
        """
        Allows the user to add a naive agent implementation to those to train against.

        Parameters
        ----------
        agent : Agent
            The agent to add to the training session.
        """
        self.agents.append(agent)

    def set_training_duration(self, training_duration: int):
        """
        Allows the user to change the number of time steps before the
        training simulation ends.

        Parameters
        ----------
        training_duration : int
            The new duration for the simulation.
        """
        self.training_duration = training_duration

    def set_demand_function(self, demand_function: DemandFunction):
        """
        Allows the user to change the demand function used by the
        environment in the training.

        Parameters
        ----------
        demand_function : DemandFunction
            The new demand function to be used.
        """
        self.demand_function = demand_function