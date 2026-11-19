"""ctypes binding for the C ABI (libchaosys).

Build the shared library first:
    cd cpp && make shared

Then:
    from chaosys import Params, State, rk4, lyapunov
    rk4((1, 1, 1), steps=1000, dt=0.01)
"""
from __future__ import annotations

import csv
import ctypes
import ctypes.util
import operator
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
_LIB_NAMES = ("libchaosys.dylib", "libchaosys.so")
_SIZE_MAX = ctypes.c_size_t(-1).value
_INT_MIN = -(2 ** 31)
_INT_MAX = 2 ** 31 - 1

__all__ = [
    "Params",
    "State",
    "ChaosysError",
    "rk4",
    "adaptive",
    "ensemble",
    "lyapunov",
    "lyapunov_report",
    "LyapunovReport",
    "lyapunov_spectrum",
    "poincare",
    "rk4_system",
    "adaptive_system",
    "lyapunov_system",
    "lyapunov_report_system",
    "lyapunov_spectrum_system",
    "poincare_system",
    "ensemble_system",
    "embed",
    "correlation_dimension",
    "permutation_entropy",
    "rk4_time_system",
    "adaptive_time_system",
    "lyapunov_time_system",
    "to_csv",
    "version",
]


class ChaosysError(RuntimeError):
    """Raised when the C ABI reports a non-OK status."""


@dataclass(frozen=True)
class Params:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


@dataclass(frozen=True)
class State:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class LyapunovReport:
    """Largest Lyapunov exponent with per-segment convergence statistics."""
    exponent: float
    stddev: float
    stderr: float
    converged: bool
    segments: int
    segment_lambdas: tuple[float, ...]


def _index(name: str, value: object) -> int:
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _usize(name: str, value: object) -> int:
    result = _index(name, value)
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    if result > _SIZE_MAX:
        raise OverflowError(f"{name} is too large for size_t")
    return result


def _cint(name: str, value: object) -> int:
    result = _index(name, value)
    if result < _INT_MIN or result > _INT_MAX:
        raise OverflowError(f"{name} is outside the C int range")
    return result


def _positive(name: str, value: object) -> int:
    result = _cint(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0")
    return result


class _CallbackState:
    __slots__ = ("error",)

    def __init__(self) -> None:
        self.error: BaseException | None = None


def _raise_callback_error(state: _CallbackState) -> None:
    if state.error is not None:
        error = state.error
        state.error = None
        raise error


def _run_deriv(dim: int, state: _CallbackState, compute, dxp) -> None:
    """Run a Python derivative, recording exceptions for the C caller."""
    if state.error is not None:
        return
    try:
        out = compute()
        if len(out) != dim:
            raise ValueError("deriv returned the wrong length")
        for i, value in enumerate(out):
            dxp[i] = value
    except BaseException as exc:
        # ctypes swallows callbacks; stash and re-raise after the C call.
        state.error = exc
        for i in range(dim):
            dxp[i] = 0.0


class _CParams(ctypes.Structure):
    _fields_ = [("sigma", ctypes.c_double), ("rho", ctypes.c_double),
                ("beta", ctypes.c_double)]


class _CState(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double),
                ("z", ctypes.c_double)]


class _CSystem(ctypes.Structure):
    pass


_DERIV = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_double),
                          ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
_CSystem._fields_ = [("dim", ctypes.c_int), ("deriv", _DERIV),
                     ("user", ctypes.c_void_p)]

_TIME_DERIV = ctypes.CFUNCTYPE(None, ctypes.c_double,
                               ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_double),
                               ctypes.c_void_p)


class _CTimeSystem(ctypes.Structure):
    _fields_ = [("dim", ctypes.c_int), ("deriv", _TIME_DERIV),
                ("user", ctypes.c_void_p)]


class _CStats(ctypes.Structure):
    _fields_ = [("lambda_value", ctypes.c_double),
                ("stddev", ctypes.c_double),
                ("stderr_value", ctypes.c_double),
                ("converged", ctypes.c_int),
                ("segments", ctypes.c_size_t)]


def _find_library() -> str:
    override = os.environ.get("CHAOSYS_LIB")
    if override:
        path = Path(override)
        if not path.is_file():
            raise ChaosysError(
                f"CHAOSYS_LIB must name a shared library file: {path}"
            )
        return str(path)
    for build in (ROOT / "cpp" / "build", ROOT / "build"):
        for name in _LIB_NAMES:
            path = build / name
            if path.is_file():
                return str(path)
    found = ctypes.util.find_library("chaosys")
    if found:
        return found
    for prefix in ("/usr/local", "/opt/homebrew", "/usr"):
        for libdir in ("lib", "lib64"):
            for name in _LIB_NAMES:
                path = Path(prefix) / libdir / name
                if path.is_file():
                    return str(path)
    raise ChaosysError(
        "libchaosys not found; install it, run `cd cpp && make shared`, "
        "or set CHAOSYS_LIB"
    )


try:
    _lib = ctypes.CDLL(_find_library())
except OSError as exc:
    raise ChaosysError(f"cannot load libchaosys: {exc}") from exc
_lib.chaosys_default_params.restype = _CParams
_lib.chaosys_version.restype = ctypes.c_char_p
_lib.chaosys_status_string.restype = ctypes.c_char_p
_lib.chaosys_status_string.argtypes = [ctypes.c_int]
_lib.chaosys_rk4.argtypes = [ctypes.POINTER(_CState), ctypes.c_size_t,
                            ctypes.c_double, _CParams]
_lib.chaosys_adaptive.argtypes = [ctypes.POINTER(_CState), ctypes.c_double,
                                 ctypes.c_double, ctypes.c_double, _CParams,
                                 ctypes.POINTER(ctypes.c_size_t)]
_lib.chaosys_ensemble.argtypes = [ctypes.POINTER(_CState), ctypes.c_size_t,
                                 ctypes.c_size_t, ctypes.c_double, ctypes.c_int,
                                 _CParams]
_lib.chaosys_lyapunov.argtypes = [ctypes.POINTER(_CState), ctypes.c_double,
                                 ctypes.c_size_t, ctypes.c_size_t,
                                 ctypes.c_size_t, _CParams,
                                 ctypes.POINTER(ctypes.c_double)]
_lib.chaosys_lyapunov_report.argtypes = [
    ctypes.POINTER(_CState), ctypes.c_double, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, _CParams, ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.POINTER(_CStats)]
_lib.chaosys_lyapunov_spectrum.argtypes = [
    ctypes.POINTER(_CState), ctypes.c_double, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, _CParams, ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_lib.chaosys_poincare.argtypes = [ctypes.POINTER(_CState), ctypes.c_double,
                                 ctypes.c_size_t, ctypes.c_size_t, _CParams,
                                 ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                 ctypes.POINTER(_CState), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_size_t)]
_lib.chaosys_rk4_system.argtypes = [ctypes.POINTER(_CSystem),
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.c_size_t, ctypes.c_double]
_lib.chaosys_adaptive_system.argtypes = [ctypes.POINTER(_CSystem),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.c_double, ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.POINTER(ctypes.c_size_t)]
_lib.chaosys_lyapunov_system.argtypes = [
    ctypes.POINTER(_CSystem), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double)]
_lib.chaosys_lyapunov_report_system.argtypes = [
    ctypes.POINTER(_CSystem), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
    ctypes.POINTER(_CStats)]
_lib.chaosys_lyapunov_spectrum_system.argtypes = [
    ctypes.POINTER(_CSystem), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t)]
_lib.chaosys_poincare_system.argtypes = [
    ctypes.POINTER(_CSystem), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int,
    ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_lib.chaosys_ensemble_system.argtypes = [
    ctypes.POINTER(_CSystem), ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_int]
_lib.chaosys_embed.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_int,
    ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t)]
_lib.chaosys_correlation_dimension.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_int,
    ctypes.c_int, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
_lib.chaosys_permutation_entropy.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_int,
    ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_lib.chaosys_rk4_time_system.argtypes = [
    ctypes.POINTER(_CTimeSystem), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_size_t, ctypes.c_double]
_lib.chaosys_adaptive_time_system.argtypes = [
    ctypes.POINTER(_CTimeSystem), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_size_t)]
_lib.chaosys_lyapunov_time_system.argtypes = [
    ctypes.POINTER(_CTimeSystem), ctypes.POINTER(ctypes.c_double),
    ctypes.c_double, ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
for _name in ("chaosys_rk4", "chaosys_adaptive", "chaosys_ensemble",
              "chaosys_lyapunov", "chaosys_lyapunov_report",
              "chaosys_lyapunov_spectrum", "chaosys_poincare",
              "chaosys_rk4_system",
              "chaosys_adaptive_system", "chaosys_lyapunov_system",
              "chaosys_lyapunov_report_system",
              "chaosys_lyapunov_spectrum_system",
              "chaosys_poincare_system", "chaosys_ensemble_system",
              "chaosys_embed", "chaosys_correlation_dimension",
              "chaosys_permutation_entropy", "chaosys_rk4_time_system",
              "chaosys_adaptive_time_system",
              "chaosys_lyapunov_time_system"):
    getattr(_lib, _name).restype = ctypes.c_int


def version() -> str:
    return _lib.chaosys_version().decode()


def _params(p: Params | Sequence[float] | None) -> _CParams:
    if p is None:
        return _lib.chaosys_default_params()
    if isinstance(p, Params):
        return _CParams(p.sigma, p.rho, p.beta)
    sigma, rho, beta = p
    return _CParams(float(sigma), float(rho), float(beta))


def _state(s: State | Sequence[float]) -> _CState:
    if isinstance(s, State):
        return _CState(s.x, s.y, s.z)
    x, y, z = s
    return _CState(float(x), float(y), float(z))


def _out_state(s: _CState) -> State:
    return State(s.x, s.y, s.z)


def _check(status: int, what: str) -> None:
    if status != 0:
        text = _lib.chaosys_status_string(status).decode()
        raise ChaosysError(f"{what}: {text} (status {status})")


def _finish(status: int, what: str, state: _CallbackState | None = None) -> None:
    if state is not None:
        _raise_callback_error(state)
    _check(status, what)


def _lyapunov_report(stats: _CStats, values) -> LyapunovReport:
    return LyapunovReport(
        exponent=float(stats.lambda_value),
        stddev=float(stats.stddev),
        stderr=float(stats.stderr_value),
        converged=bool(stats.converged),
        segments=int(stats.segments),
        segment_lambdas=tuple(float(values[i]) for i in range(stats.segments)),
    )


def _floats(values, count: int) -> list[float]:
    return [float(values[i]) for i in range(count)]


def _window(transient, segment, segments) -> tuple[int, int, int]:
    return (
        _usize("transient", transient),
        _usize("segment", segment),
        _usize("segments", segments),
    )


def rk4(state: State | Sequence[float], steps: int, dt: float,
        params: Params | Sequence[float] | None = None) -> State:
    s = _state(state)
    steps = _usize("steps", steps)
    _check(_lib.chaosys_rk4(ctypes.byref(s), steps, dt, _params(params)), "rk4")
    return _out_state(s)


def adaptive(state: State | Sequence[float], t_end: float, rtol: float = 1e-9,
             atol: float = 1e-11,
             params: Params | Sequence[float] | None = None) -> tuple[State, int]:
    s = _state(state)
    accepted = ctypes.c_size_t(0)
    _check(_lib.chaosys_adaptive(ctypes.byref(s), t_end, rtol, atol,
                                _params(params), ctypes.byref(accepted)),
           "adaptive")
    return _out_state(s), int(accepted.value)


def ensemble(states: Iterable[State | Sequence[float]], steps: int, dt: float,
             threads: int = 8,
             params: Params | Sequence[float] | None = None) -> list[State]:
    steps = _usize("steps", steps)
    threads = _cint("threads", threads)
    items = [_state(s) for s in states]
    n = len(items)
    if n == 0:
        return []
    buf = (_CState * n)(*items)
    _check(_lib.chaosys_ensemble(buf, n, steps, dt, threads, _params(params)),
           "ensemble")
    return [_out_state(buf[i]) for i in range(n)]


def lyapunov(state: State | Sequence[float], dt: float,
             transient: int = 2000, segment: int = 50, segments: int = 200,
             params: Params | Sequence[float] | None = None) -> float:
    s = _state(state)
    transient, segment, segments = _window(transient, segment, segments)
    value = ctypes.c_double(0.0)
    _check(_lib.chaosys_lyapunov(ctypes.byref(s), dt, transient, segment,
                                segments, _params(params),
                                ctypes.byref(value)), "lyapunov")
    return float(value.value)


def lyapunov_report(
        state: State | Sequence[float], dt: float, transient: int = 2000,
        segment: int = 50, segments: int = 200,
        params: Params | Sequence[float] | None = None) -> LyapunovReport:
    """Largest Lyapunov exponent plus per-segment convergence statistics."""
    s = _state(state)
    transient, segment, segments = _window(transient, segment, segments)
    if segment == 0 or segments == 0:
        raise ValueError("segment and segments must be > 0")
    values = (ctypes.c_double * segments)()
    stats = _CStats()
    _check(_lib.chaosys_lyapunov_report(
        ctypes.byref(s), dt, transient, segment, segments, _params(params),
        values, segments, ctypes.byref(stats)), "lyapunov_report")
    return _lyapunov_report(stats, values)


def lyapunov_spectrum(
        state: State | Sequence[float], dt: float, transient: int = 2000,
        segment: int = 50, segments: int = 200,
        params: Params | Sequence[float] | None = None) -> list[float]:
    """Full Lyapunov spectrum (descending) of the Lorenz system."""
    s = _state(state)
    transient, segment, segments = _window(transient, segment, segments)
    values = (ctypes.c_double * 3)()
    count = ctypes.c_size_t(0)
    _check(_lib.chaosys_lyapunov_spectrum(
        ctypes.byref(s), dt, transient, segment, segments, _params(params),
        values, 3, ctypes.byref(count)), "lyapunov_spectrum")
    return _floats(values, count.value)


def poincare(state: State | Sequence[float], dt: float,
             transient: int = 2000, max_steps: int = 20000,
             capacity: int = 200, component: int = 2, level: float = 20.0,
             direction: int = 1,
             params: Params | Sequence[float] | None = None) -> list[State]:
    transient = _usize("transient", transient)
    max_steps = _usize("max_steps", max_steps)
    capacity = _usize("capacity", capacity)
    component = _cint("component", component)
    direction = _cint("direction", direction)
    if capacity == 0:
        raise ValueError("capacity must be > 0")
    s = _state(state)
    buf = (_CState * capacity)()
    count = ctypes.c_size_t(0)
    _check(_lib.chaosys_poincare(ctypes.byref(s), dt, transient, max_steps,
                                _params(params), component, level, direction,
                                buf, capacity, ctypes.byref(count)),
           "poincare")
    return [_out_state(buf[i]) for i in range(count.value)]


def _make_system(dim: int, deriv: Callable[[Sequence[float]], Sequence[float]]):
    dim = _positive("dim", dim)
    arr_type = ctypes.c_double * dim
    state = _CallbackState()

    def _deriv(xp, dxp, _user):
        _run_deriv(dim, state, lambda: deriv([xp[i] for i in range(dim)]), dxp)

    callback = _DERIV(_deriv)
    return callback, _CSystem(dim, callback, None), arr_type, state


def rk4_system(dim: int, deriv: Callable[[Sequence[float]], Sequence[float]],
               x: Sequence[float], steps: int, dt: float) -> list[float]:
    """Integrate a custom chaotic ODE; deriv(x) returns dx as a sequence."""
    dim = _positive("dim", dim)
    steps = _usize("steps", steps)
    if len(x) != dim:
        raise ValueError("len(x) != dim")
    callback, system, arr_type, state = _make_system(dim, deriv)
    arr = arr_type(*map(float, x))
    status = _lib.chaosys_rk4_system(ctypes.byref(system), arr, steps, dt)
    _finish(status, "rk4_system", state)
    return [arr[i] for i in range(dim)]


def adaptive_system(dim: int,
                    deriv: Callable[[Sequence[float]], Sequence[float]],
                    x: Sequence[float], t_end: float, rtol: float = 1e-9,
                    atol: float = 1e-11) -> tuple[list[float], int]:
    """Adaptive DP5 for a custom chaotic ODE."""
    dim = _positive("dim", dim)
    if len(x) != dim:
        raise ValueError("len(x) != dim")
    callback, system, arr_type, state = _make_system(dim, deriv)
    arr = arr_type(*map(float, x))
    accepted = ctypes.c_size_t(0)
    status = _lib.chaosys_adaptive_system(ctypes.byref(system), arr, t_end,
                                         rtol, atol, ctypes.byref(accepted))
    _finish(status, "adaptive_system", state)
    return [arr[i] for i in range(dim)], int(accepted.value)


def lyapunov_system(dim: int,
                    deriv: Callable[[Sequence[float]], Sequence[float]],
                    x0: Sequence[float], dt: float, transient: int = 2000,
                    segment: int = 50, segments: int = 200) -> float:
    """Largest Lyapunov exponent of a custom chaotic ODE."""
    dim = _positive("dim", dim)
    transient, segment, segments = _window(transient, segment, segments)
    if len(x0) != dim:
        raise ValueError("len(x0) != dim")
    callback, system, arr_type, state = _make_system(dim, deriv)
    xarr = arr_type(*map(float, x0))
    value = ctypes.c_double(0.0)
    status = _lib.chaosys_lyapunov_system(
        ctypes.byref(system), xarr, dt, transient, segment, segments,
        ctypes.byref(value))
    _finish(status, "lyapunov_system", state)
    return float(value.value)


def lyapunov_report_system(
        dim: int, deriv: Callable[[Sequence[float]], Sequence[float]],
        x0: Sequence[float], dt: float, transient: int = 2000,
        segment: int = 50, segments: int = 200) -> LyapunovReport:
    """LyapunovReport for a custom chaotic ODE."""
    dim = _positive("dim", dim)
    transient, segment, segments = _window(transient, segment, segments)
    if len(x0) != dim:
        raise ValueError("len(x0) != dim")
    if segment == 0 or segments == 0:
        raise ValueError("segment and segments must be > 0")
    callback, system, arr_type, state = _make_system(dim, deriv)
    xarr = arr_type(*map(float, x0))
    values = (ctypes.c_double * segments)()
    stats = _CStats()
    status = _lib.chaosys_lyapunov_report_system(
        ctypes.byref(system), xarr, dt, transient, segment, segments, values,
        segments, ctypes.byref(stats))
    _finish(status, "lyapunov_report_system", state)
    return _lyapunov_report(stats, values)


def lyapunov_spectrum_system(
        dim: int, deriv: Callable[[Sequence[float]], Sequence[float]],
        x0: Sequence[float], dt: float, transient: int = 2000,
        segment: int = 50, segments: int = 200) -> list[float]:
    """Full Lyapunov spectrum (descending) of a custom chaotic ODE."""
    dim = _positive("dim", dim)
    transient, segment, segments = _window(transient, segment, segments)
    if len(x0) != dim:
        raise ValueError("len(x0) != dim")
    callback, system, arr_type, state = _make_system(dim, deriv)
    xarr = arr_type(*map(float, x0))
    values = (ctypes.c_double * dim)()
    count = ctypes.c_size_t(0)
    status = _lib.chaosys_lyapunov_spectrum_system(
        ctypes.byref(system), xarr, dt, transient, segment, segments, values,
        dim, ctypes.byref(count))
    _finish(status, "lyapunov_spectrum_system", state)
    return _floats(values, count.value)


def poincare_system(dim: int,
                    deriv: Callable[[Sequence[float]], Sequence[float]],
                    x0: Sequence[float], dt: float, transient: int = 2000,
                    max_steps: int = 20000, capacity: int = 200,
                    component: int = 2, level: float = 20.0,
                    direction: int = 1) -> list[tuple[float, ...]]:
    """Poincare section of a custom chaotic ODE, as tuples of length dim."""
    dim = _positive("dim", dim)
    transient = _usize("transient", transient)
    max_steps = _usize("max_steps", max_steps)
    capacity = _usize("capacity", capacity)
    component = _cint("component", component)
    direction = _cint("direction", direction)
    if len(x0) != dim or capacity == 0:
        raise ValueError("len(x0) == dim and capacity > 0 required")
    callback, system, arr_type, state = _make_system(dim, deriv)
    xarr = arr_type(*map(float, x0))
    out = (ctypes.c_double * (capacity * dim))()
    count = ctypes.c_size_t(0)
    status = _lib.chaosys_poincare_system(
        ctypes.byref(system), xarr, dt, transient, max_steps, component, level,
        direction, out, capacity, ctypes.byref(count))
    _finish(status, "poincare_system", state)
    return [tuple(out[i * dim:(i + 1) * dim]) for i in range(count.value)]


def ensemble_system(dim: int,
                    deriv: Callable[[Sequence[float]], Sequence[float]],
                    states: Iterable[Sequence[float]], steps: int, dt: float,
                    threads: int = 1) -> list[list[float]]:
    """Batch integration of a custom chaotic ODE (flat AoS in C)."""
    dim = _positive("dim", dim)
    steps = _usize("steps", steps)
    threads = _cint("threads", threads)
    rows = [list(map(float, s)) for s in states]
    if any(len(s) != dim for s in rows):
        raise ValueError("every state must have length dim")
    n = len(rows)
    if n == 0:
        return []
    callback, system, arr_type, state = _make_system(dim, deriv)
    flat = (ctypes.c_double * (n * dim))(*[v for row in rows for v in row])
    status = _lib.chaosys_ensemble_system(ctypes.byref(system), flat, n, steps,
                                         dt, threads)
    _finish(status, "ensemble_system", state)
    return [list(flat[i * dim:(i + 1) * dim]) for i in range(n)]


def embed(series: Iterable[float], dimension: int = 3,
          delay: int = 1) -> list[tuple[float, ...]]:
    """Takens embedding of a scalar series."""
    values = [float(v) for v in series]
    n = len(values)
    dimension = _positive("dimension", dimension)
    delay = _positive("delay", delay)
    if n < (dimension - 1) * delay + 1:
        raise ValueError("series too short for dimension/delay")
    rows = n - (dimension - 1) * delay
    arr = (ctypes.c_double * n)(*values)
    out = (ctypes.c_double * (rows * dimension))()
    count = ctypes.c_size_t(0)
    _check(_lib.chaosys_embed(arr, n, dimension, delay, out,
                             rows * dimension, ctypes.byref(count)), "embed")
    return [tuple(out[i * dimension:(i + 1) * dimension])
            for i in range(count.value)]


def correlation_dimension(series: Iterable[float], dimension: int = 4,
                          delay: int = 20,
                          max_points: int = 2000) -> float:
    """Grassberger-Procaccia correlation dimension of a scalar series."""
    values = [float(v) for v in series]
    n = len(values)
    if n == 0:
        raise ValueError("series must not be empty")
    dimension = _positive("dimension", dimension)
    delay = _positive("delay", delay)
    max_points = _usize("max_points", max_points)
    arr = (ctypes.c_double * n)(*values)
    value = ctypes.c_double(0.0)
    _check(_lib.chaosys_correlation_dimension(arr, n, dimension, delay,
                                             max_points, ctypes.byref(value)),
           "correlation_dimension")
    return float(value.value)


def permutation_entropy(series: Iterable[float], order: int = 3,
                        delay: int = 1) -> float:
    """Normalized permutation entropy in [0, 1]."""
    values = [float(v) for v in series]
    n = len(values)
    if n == 0:
        raise ValueError("series must not be empty")
    order = _cint("order", order)
    delay = _cint("delay", delay)
    arr = (ctypes.c_double * n)(*values)
    value = ctypes.c_double(0.0)
    _check(_lib.chaosys_permutation_entropy(arr, n, order, delay,
                                           ctypes.byref(value)),
           "permutation_entropy")
    return float(value.value)


def _make_time_system(dim: int,
                      deriv: Callable[[float, Sequence[float]],
                                      Sequence[float]]) -> tuple:
    dim = _positive("dim", dim)
    state = _CallbackState()

    def _deriv(t, xp, dxp, _user):
        _run_deriv(
            dim,
            state,
            lambda: deriv(t, [xp[i] for i in range(dim)]),
            dxp,
        )

    callback = _TIME_DERIV(_deriv)
    return callback, _CTimeSystem(dim, callback, None), state


def rk4_time_system(dim: int,
                    deriv: Callable[[float, Sequence[float]],
                                    Sequence[float]],
                    x: Sequence[float], t0: float, steps: int,
                    dt: float) -> list[float]:
    """Fixed RK4 for a time-dependent ODE dx/dt = f(t, x)."""
    dim = _positive("dim", dim)
    steps = _usize("steps", steps)
    if len(x) != dim:
        raise ValueError("len(x) != dim")
    callback, system, state = _make_time_system(dim, deriv)
    arr = (ctypes.c_double * dim)(*map(float, x))
    status = _lib.chaosys_rk4_time_system(ctypes.byref(system), arr, t0, steps,
                                         dt)
    _finish(status, "rk4_time_system", state)
    return [arr[i] for i in range(dim)]


def adaptive_time_system(dim: int,
                         deriv: Callable[[float, Sequence[float]],
                                         Sequence[float]],
                         x: Sequence[float], t0: float, t_end: float,
                         rtol: float = 1e-9,
                         atol: float = 1e-11) -> tuple[list[float], int]:
    """Adaptive DP5 for a time-dependent ODE."""
    dim = _positive("dim", dim)
    if len(x) != dim:
        raise ValueError("len(x) != dim")
    callback, system, state = _make_time_system(dim, deriv)
    arr = (ctypes.c_double * dim)(*map(float, x))
    accepted = ctypes.c_size_t(0)
    status = _lib.chaosys_adaptive_time_system(
        ctypes.byref(system), arr, t0, t_end, rtol, atol,
        ctypes.byref(accepted))
    _finish(status, "adaptive_time_system", state)
    return [arr[i] for i in range(dim)], int(accepted.value)


def lyapunov_time_system(dim: int,
                         deriv: Callable[[float, Sequence[float]],
                                         Sequence[float]],
                         x0: Sequence[float], t0: float, dt: float,
                         transient: int = 2000, segment: int = 50,
                         segments: int = 200) -> float:
    """Largest Lyapunov exponent of a time-dependent ODE."""
    dim = _positive("dim", dim)
    transient, segment, segments = _window(transient, segment, segments)
    if len(x0) != dim:
        raise ValueError("len(x0) != dim")
    callback, system, state = _make_time_system(dim, deriv)
    arr = (ctypes.c_double * dim)(*map(float, x0))
    value = ctypes.c_double(0.0)
    status = _lib.chaosys_lyapunov_time_system(
        ctypes.byref(system), arr, t0, dt, transient, segment, segments,
        ctypes.byref(value))
    _finish(status, "lyapunov_time_system", state)
    return float(value.value)


def to_csv(path, rows: Iterable, header: Sequence[str] | None = None,
           *, delimiter: str = ",") -> int:
    """Write rows to a CSV file or text stream.

    Rows may be :class:`State` instances, tuples/lists, or any iterable of
    values.  `header`, when given, must match every row width.  Returns the
    number of data rows written.
    """
    if len(delimiter) != 1:
        raise ValueError("delimiter must be a single character")
    header_values = None if not header else tuple(str(h) for h in header)
    written = 0

    def values(row):
        if isinstance(row, State):
            return (row.x, row.y, row.z)
        return tuple(row)

    def emit(file_obj):
        nonlocal written
        writer = csv.writer(file_obj, delimiter=delimiter, lineterminator="\n")
        if header_values is not None:
            writer.writerow(header_values)
        for row in rows:
            row_values = values(row)
            if header_values is not None and len(row_values) != len(header_values):
                raise ValueError("row width does not match header")
            writer.writerow(row_values)
            written += 1

    if hasattr(path, "write"):
        emit(path)
    else:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            emit(handle)
    return written


if __name__ == "__main__":
    print(f"libchaosys {version()} @ {_find_library()}")
    print("lyapunov:", lyapunov((1, 1, 1), 0.005))
