from __future__ import annotations

from dataclasses import dataclass, field

from zwm.core.hexagram import Hexagram


@dataclass
class OnlineLearner:
    learning_rate: float = 0.01
    preference_weights: dict[str, float] = field(default_factory=lambda: {
        "time": 0.17, "space": 0.17, "social": 0.17,
        "element": 0.17, "risk": 0.16, "narrative": 0.16,
    })
    visit_counts: dict[int, int] = field(default_factory=dict)
    total_visits: int = 0

    def update_from_outcome(
        self,
        h: Hexagram,
        reward: float,
        moe_weights: dict[str, float] | None = None,
    ) -> None:
        self.visit_counts[h.normal_order] = (
            self.visit_counts.get(h.normal_order, 0) + 1
        )
        self.total_visits += 1

        if moe_weights:
            for expert, weight in moe_weights.items():
                if expert in self.preference_weights:
                    delta = self.learning_rate * (reward - 0.5) * weight
                    self.preference_weights[expert] += delta

            total_w = sum(self.preference_weights.values())
            if total_w > 0:
                for k in self.preference_weights:
                    self.preference_weights[k] /= total_w

    def get_visit_count(self, h: Hexagram) -> int:
        return self.visit_counts.get(h.normal_order, 0)

    def novelty_bonus(self, h: Hexagram) -> float:
        visits = self.get_visit_count(h)
        return 1.0 / (1.0 + visits)


@dataclass
class CuriosityScheduler:
    beta_initial: float = 0.5
    beta_final: float = 0.05
    decay_rate: float = 0.001
    step_count: int = 0

    @property
    def beta(self) -> float:
        return self.beta_final + (self.beta_initial - self.beta_final) * (
            1.0 / (1.0 + self.decay_rate * self.step_count)
        )

    def step(self) -> float:
        self.step_count += 1
        return self.beta


@dataclass
class GrowthManager:
    total_episodes: int = 0

    @property
    def phase(self) -> str:
        if self.total_episodes < 100:
            return "explore"
        elif self.total_episodes < 500:
            return "exploit"
        return "expert"

    @property
    def curiosity_weight(self) -> float:
        weights = {"explore": 0.5, "exploit": 0.2, "expert": 0.05}
        return weights[self.phase]

    def advance(self) -> None:
        self.total_episodes += 1
