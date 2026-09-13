"""Episode allocation independent of completion time and simulator imports."""


class EpisodeQuota:
    """Preassign trials to slots so fast successes cannot crowd out timeouts."""

    def __init__(self, episodes: int, num_envs: int):
        if episodes <= 0 or num_envs <= 0:
            raise ValueError("episodes and num_envs must be positive")
        self.quotas = [episodes // num_envs + int(i < episodes % num_envs) for i in range(num_envs)]
        self.counts = [0] * num_envs

    def accept(self, env_id: int) -> bool:
        if not 0 <= env_id < len(self.quotas):
            raise IndexError("environment id outside allocation")
        if self.counts[env_id] >= self.quotas[env_id]:
            return False
        self.counts[env_id] += 1
        return True

    @property
    def complete(self) -> bool:
        return self.counts == self.quotas
