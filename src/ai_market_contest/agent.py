from abc import ABCMeta, abstractmethod

from ray.rllib.policy.policy import Policy


class Agent(Policy, metaclass=ABCMeta):
    """
    Agent interface - an agent represents a firm selling a product in the market.

    An agent encapsulates the users private pricing strategy.

    The user is free to implement this interface in order to test strategies.
    As is standard for ML models, it uses the policy-update format.
    For those not familiar with policy-update, see the comments on each function.
    """

    @abstractmethod
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
    ):
        """
        Query the agent for the next price to set.

        Parameters
        ----------
        last_round_all_agents_prices : list of float
            List of all the prices set by all agents in the previous timestep.
        identity_index: int
            A positive integer that tells the agent which index in the list
            corresponds to themself.

        Returns
        -------
        float
            Price of the product set by the agent at the current timestep,
            discretised within [0,1].

        Raises
        ------
        NotImplementedError
            If concrete class does not override method.
        """

        raise NotImplementedError

    @abstractmethod
    def learn_on_batch(self, samples):
        """
        Check if the agent's learning has converged.

        Returns
        -------
        bool : True if the agent learning has converged, False otherwise.

        Raises
        ------
        NotImplementedError
            If concrete class does not override method.
        """
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            all(
                [
                    hasattr(subclass, "policy"),
                    callable(subclass.policy),
                    hasattr(subclass, "update"),
                    callable(subclass.update),
                    hasattr(subclass, "learning_has_converged"),
                    callable(subclass.learning_has_converged),
                ]
            )
            or NotImplemented
        )
