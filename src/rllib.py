import os
import tempfile
from datetime import datetime
from ai_market_contest.training.agent_name_maker import AgentNameMaker
from ai_market_contest.training.sequential_agent_name_maker import (
    SequentialAgentNameMaker,
)

from ray.rllib import agents  # type: ignore
from ray.tune.logger import UnifiedLogger, pretty_print  # type: ignore
from ray.tune.registry import register_env  # type: ignore

from ai_market_contest.demandfunctions.fixed_lowest_takes_all_demand_function import (
    LowestTakesAllDemandFunction,
)
from ai_market_contest.environment import Market

num_agents: int = 10
agent_name_maker: AgentNameMaker = SequentialAgentNameMaker(num_agents)
register_env(
    "marketplace",
    lambda x: Market(
        num_agents, LowestTakesAllDemandFunction(99), 100, agent_name_maker
    ),
)

config = agents.dqn.DEFAULT_CONFIG.copy()
config.update(
    {
        "num_workers": 1,
    }
)


def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


trainer = agents.dqn.DQNTrainer(env="marketplace")


for i in range(10):
    result = trainer.train()
    print(pretty_print(result))
