# import os
# import pathlib
# import tempfile
# from datetime import datetime

# from ray.rllib import agents  # type: ignore
# from ray.rllib.policy.policy import PolicySpec
# from ray.tune.logger import UnifiedLogger, pretty_print  # type: ignore
# from ray.tune.registry import register_env  # type: ignore

# from ai_market_contest.agents.random_agent import RandomAgent
# from ai_market_contest.demandfunctions.fixed_lowest_takes_all_demand_function import (
#     LowestTakesAllDemandFunction,
# )
# from ai_market_contest.environment import Market
# from ai_market_contest.test_agent import TestAgent
# from ai_market_contest.training.agent_name_maker import AgentNameMaker
# from ai_market_contest.training.agent_trainer import AgentTrainer
# from ai_market_contest.training.policy_selector import PolicySelector
# from ai_market_contest.training.sequential_agent_name_maker import (
#     SequentialAgentNameMaker,
# )

# num_agents: int = 5
# agent_name_maker: AgentNameMaker = SequentialAgentNameMaker(num_agents)


# num_self_play_agents = 2
# demand_function = LowestTakesAllDemandFunction
# training_duration: int = 10
# env = Market(num_agents, demand_function(), training_duration, agent_name_maker)

# random_agent_spec = PolicySpec(policy_class=RandomAgent)
# test_agent_spec = PolicySpec(policy_class=TestAgent)

# naive_agents_counts = {"random-agent": num_agents - num_self_play_agents - 1}
# ps = PolicySelector(
#     "test-agent",
#     self_play_number=num_self_play_agents,
#     naive_agents_counts=naive_agents_counts,
# )
# config = {
#     "num_workers": 0,
#     "prioritized_replay": False,
#     "multiagent": {
#         "policies_to_train": ["test-agent"],
#         "policies": {
#             "random-agent": random_agent_spec,
#             "test-agent": test_agent_spec,
#             "test-agent-opponent": test_agent_spec,
#         },
#         "policy_mapping_fn": ps.get_select_policy_function(),
#     },
# }

# agent_trainer = AgentTrainer(env, config, restored=False, checkpoint_path=None)
# agent_trainer.train(1, True)
# path = pathlib.Path.cwd()
# agent_trainer.save(path)
# agent_trainer = AgentTrainer(
#     env,
#     config,
#     restored=True,
#     checkpoint_path=str((path / "checkpoint_000001/checkpoint-1").resolve()),
# )
# agent_trainer.train(1, True)
# agent_trainer.save(path)
