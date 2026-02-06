"""Parameter definitions and G generators for permeation simulations."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def constant_G(value: float = 0.0) -> Callable[[int, float], np.ndarray]:
    """
    G generator: constant incident flux (dimensionless before scaling by ks).

    Returns a callable (Nt, T) -> array of shape (Nt+1,) filled with value.
    """
    def gen(Nt: int, T: float) -> np.ndarray:
        return np.full(Nt + 1, value, dtype=float)
    return gen


def zeros_G() -> Callable[[int, float], np.ndarray]:
    """G generator: zero incident flux. Same as constant_G(0)."""
    return constant_G(0.0)


def step_G(
    value: float = 1.0,
    t_start_frac: float = 0.0,
    t_end_frac: float = 1.0,
) -> Callable[[int, float], np.ndarray]:
    """
    G generator: constant value between t_start_frac*T and t_end_frac*T, else zero.

    (Nt, T) -> array of shape (Nt+1,) with correct time mapping.
    """
    def gen(Nt: int, T: float) -> np.ndarray:
        t = np.linspace(0, T, Nt + 1)
        t_start = t_start_frac * T
        t_end = t_end_frac * T
        out = np.zeros(Nt + 1, dtype=float)
        out[(t >= t_start) & (t <= t_end)] = value
        return out
    return gen


class Parameters:
    """
    Explicit parameters for the permeation solver.

    All solver inputs are set here. G is produced by G_generator(Nt, T)
    so it always has the correct shape (Nt+1,). Use constant_G(value) or
    a custom callable (Nt, T) -> np.ndarray of shape (Nt+1,).
    """

    def __init__(
        self,
        *,
        Nx: int = 30,
        Nt: int = 100,
        T: float = 1000.0,
        D: float = 1.1e-8,
        L: float = 2e-5,
        ku: float = 1e-33,
        kd: float = 2e-33,
        ks: float = 1e19,
        ncorrection: int = 3,
        Tend: float = 705.0,
        PLOT: bool = False,
        I: Any = None,
        Uinit: np.ndarray | None = None,
        G_generator: Callable[[int, float], np.ndarray] | None = None,
    ) -> None:
        self.Nx = Nx
        self.Nt = Nt
        self.T = T
        self.D = D
        self.L = L
        self.ku = ku
        self.kd = kd
        self.ks = ks
        self.ncorrection = ncorrection
        self.Tend = Tend
        self.PLOT = PLOT
        self.I = I
        self.Uinit = np.zeros(Nx + 1, dtype=float) if Uinit is None else np.asarray(Uinit, dtype=float)
        self.G_generator = G_generator if G_generator is not None else zeros_G()

    def get_G(self) -> np.ndarray:
        """Incident flux array with shape (Nt+1,)."""
        return self.G_generator(self.Nt, self.T)

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for compatibility; 'G' returns get_G()."""
        if key == "G":
            return self.get_G()
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        """Dict of all parameters for the solver (G and Uinit with correct shapes)."""
        return {
            "Nx": self.Nx,
            "Nt": self.Nt,
            "T": self.T,
            "D": self.D,
            "L": self.L,
            "ku": self.ku,
            "kd": self.kd,
            "ks": self.ks,
            "ncorrection": self.ncorrection,
            "Tend": self.Tend,
            "PLOT": self.PLOT,
            "I": self.I,
            "Uinit": self.Uinit,
            "G": self.get_G(),
        }

    def show(self, *, array_preview: int = 4) -> None:
        """
        Pretty-print parameters for inspection.

        - Floats shown in scientific notation when appropriate
        - Arrays shown with a short preview
        """

        def fmt_scalar(v):
            if isinstance(v, float):
                av = abs(v)
                if (av != 0) and (av >= 1e6 or av < 1e-3):
                    return f"{v:.3e}"
                return v
            return v

        def fmt_array(a: np.ndarray):
            if a.size == 0:
                return "[]"
            n = min(array_preview, a.size)
            head = " ".join(f"{x:.3g}" for x in a[:n])
            if a.size > n:
                return f"[{head} …] (len={a.size})"
            return f"[{head}]"

        rows = [
            ("Nx", self.Nx),
            ("Nt", self.Nt),
            ("T", fmt_scalar(self.T)),
            ("D", fmt_scalar(self.D)),
            ("L", fmt_scalar(self.L)),
            ("ku", fmt_scalar(self.ku)),
            ("kd", fmt_scalar(self.kd)),
            ("ks", fmt_scalar(self.ks)),
            ("ncorrection", self.ncorrection),
            ("Tend", fmt_scalar(self.Tend)),
            ("PLOT", self.PLOT),
            ("I", self.I),
            ("Uinit", fmt_array(self.Uinit)),
            ("G", fmt_array(self.get_G())),
        ]

        try:
            import pandas as pd
            from IPython.display import display

            df = pd.DataFrame(rows, columns=["parameter", "value"])
            display(df)
        except Exception:
            w = max(len(k) for k, _ in rows)
            for k, v in rows:
                print(f"{k:<{w}} : {v}")
    