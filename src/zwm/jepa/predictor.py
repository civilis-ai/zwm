from __future__ import annotations

import numpy as np


class JEPAPredictor:
    def __init__(self, input_dim: int = 77, hidden_dim: int = 32, latent_dim: int = 32) -> None:
        rng = np.random.default_rng(42)
        self._w1 = rng.normal(0, 1.0 / np.sqrt(input_dim), (input_dim, hidden_dim)).astype(np.float32)
        self._b1 = np.zeros(hidden_dim, dtype=np.float32)
        self._w2 = rng.normal(0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, latent_dim)).astype(np.float32)
        self._b2 = np.zeros(latent_dim, dtype=np.float32)
        self._w_pred = rng.normal(0, 1.0 / np.sqrt(latent_dim), (latent_dim, input_dim)).astype(np.float32)
        self._b_pred = np.zeros(input_dim, dtype=np.float32)

    def predict(self, z_world: np.ndarray) -> np.ndarray:
        h = np.tanh(z_world @ self._w1 + self._b1)
        latent = np.tanh(h @ self._w2 + self._b2)
        return latent @ self._w_pred + self._b_pred

    def sigreg_loss(self, latent: np.ndarray, lambda_reg: float = 0.1) -> float:
        n = latent.shape[-1]
        z = (latent - np.mean(latent)) / (np.std(latent) + 1e-8)

        s1 = np.mean(np.cos(z))
        s2 = np.mean(z * z)
        loss = float(s1 * s1 + (s2 - 1.0) * (s2 - 1.0))
        return float(lambda_reg * loss)

    def train_step(
        self,
        z_current: np.ndarray,
        z_next: np.ndarray,
        learning_rate: float = 0.001,
    ) -> float:
        pred = self.predict(z_current)
        error = pred - z_next
        mse_loss = float(np.mean(error * error))

        h = np.tanh(z_current @ self._w1 + self._b1)
        latent = np.tanh(h @ self._w2 + self._b2)
        sigreg = self.sigreg_loss(latent)

        total_loss = mse_loss + sigreg

        grad_pred = (2.0 / len(z_next)) * error
        self._w_pred -= learning_rate * np.outer(latent, grad_pred)
        self._b_pred -= learning_rate * grad_pred

        d_latent = grad_pred @ self._w_pred.T * (1.0 - latent * latent)
        self._w2 -= learning_rate * np.outer(h, d_latent)
        self._b2 -= learning_rate * d_latent

        d_h = d_latent @ self._w2.T * (1.0 - h * h)
        self._w1 -= learning_rate * np.outer(z_current, d_h)
        self._b1 -= learning_rate * d_h

        return float(total_loss)
