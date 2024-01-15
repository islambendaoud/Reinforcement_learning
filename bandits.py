import numpy as np


########################################
#                 Arms                 #
########################################
class Normal:
    def __init__(
        self, mean, std=1.0, rng: np.random.Generator = np.random.default_rng()
    ):
        self.mean = mean
        self.std = std
        self.name = "Normal"
        self.rng = rng

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f}, std={self.std:.3f})"

    def sample(self):
        return self.rng.normal(loc=self.mean, scale=self.std)


class Bernoulli:
    def __init__(self, mean: float, rng: np.random.Generator = np.random.default_rng()):
        assert (
            0 <= mean <= 1
        ), f"The mean of a Bernoulli should between 0 and 1: mean={mean}"
        self.mean = mean
        self.std = mean * (1 - mean)
        self.name = "Bernoulli"
        self.rng = rng

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f})"

    def sample(self):
        return self.rng.binomial(1, self.mean)


########################################
#               Bandit                 #
########################################
class Bandit:
    def __init__(self, arms, structure="unknown"):
        self.arms = arms
        self.nbr_arms = len(arms)
        self.structure = structure

        # Preprocessing of useful statistics

        # Expected value of arms
        # Bandits community vocabulary
        self.rewards = np.array([arms[i].mean for i in range(self.nbr_arms)])
        # Probability community vocabulary
        self.means = self.rewards

        # Best arm index (one of) and expected value (unique)
        self.best_arm = np.argmax(self.rewards)
        self.best_reward = np.max(self.rewards)
        self.best_mean = self.best_reward

        # Regret/suboptimality gap of arms
        self.regrets = self.best_reward - self.rewards

    def __str__(self):
        return f"Bandit({self.arms})"

    def __repr__(self):
        return f"Bandit({self.arms})"

    # Bandits community vocabulary
    def pull(self, arm):
        return self.arms[arm].sample()

    # Probability community vocabulary
    def sample(self, idx):
        return self.pull(idx)


class NormalBandit(Bandit):
    def __init__(self, means, stds=None, structure="unknown", seed: int = 42):
        assert len(means) > 0, "means should not be empty"
        seeds = get_n_seeds_from_one_seed(seed, len(means))
        if stds is not None:
            assert len(means) == len(
                stds
            ), f"Lengths should match: len(means)={len(means)} - len(stds)={len(stds)}"

            arms = [
                Normal(m, s, rng=np.random.default_rng(seeds[i]))
                for i, (m, s) in enumerate(zip(means, stds))
            ]
        else:
            arms = [
                Normal(m, rng=np.random.default_rng(seeds[i]))
                for i, m in enumerate(means)
            ]
        Bandit.__init__(self, arms, structure=structure)


def get_n_seeds_from_one_seed(seed: int, n: int):
    return np.random.SeedSequence(seed).spawn(n)


class BernoulliBandit(Bandit):
    def __init__(self, means, structure="unknown", seed: int = 42):
        assert len(means) > 0, "means should not be empty"
        assert np.all(means >= 0) and np.all(
            means <= 1
        ), "Bernoulli mean should be between 0 and 1:\n(means={means})"
        seeds = get_n_seeds_from_one_seed(seed, len(means))
        arms = [
            Bernoulli(m, np.random.default_rng(seeds[i])) for i, m in enumerate(means)
        ]
        Bandit.__init__(
            self,
            arms,
            structure=structure,
        )
