"""Fit candidate equations written as text, e.g. "A*exp(-k*t)*cos(w*t + p) + c".

Expressions are parsed with a strict whitelist (numbers, + - * / **, t, parameter names and a
few numpy functions), so text from a language model cannot run arbitrary code.
"""

from __future__ import annotations

import ast
import math
import re

import numpy as np
from scipy.optimize import curve_fit

FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh, "arctan": np.arctan,
    "arcsin": np.arcsin, "arccos": np.arccos, "sign": np.sign,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)
MAX_PARAMS = 8


def parse_expression(expression: str, variable: str = "t"):
    """Return (compiled code, parameter names in order of appearance)."""
    text = expression.strip().replace("^", "**")
    if "=" in text:  # accept "y = ..." or "theta(t) = ..."
        text = text.split("=", 1)[1].strip()
    if not text or len(text) > 400:
        raise ValueError("the expression is empty or too long")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError, MemoryError) as error:
        raise ValueError(f"cannot read the expression: {getattr(error, 'msg', error)}") from error
    params: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"'{type(node).__name__}' is not allowed in an expression")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
            raise ValueError("only numbers are allowed as constants")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise ValueError(f"unknown function; use one of: {', '.join(sorted(FUNCTIONS))}")
            if node.keywords or len(node.args) != 1:
                raise ValueError(f"{node.func.id}() takes exactly one argument")
        if isinstance(node, ast.Name):
            name = node.id
            if name == variable or name in FUNCTIONS or name in CONSTANTS:
                continue
            if name.startswith("_") or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"bad parameter name {name!r}")
            if name not in params:
                params.append(name)
    if variable not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}:
        raise ValueError(f"the expression must use the variable '{variable}'")
    if len(params) > MAX_PARAMS:
        raise ValueError(f"too many parameters ({len(params)}); use at most {MAX_PARAMS}")
    # Integers become floats so that something like 9**9**9 overflows at once instead of
    # building a gigantic integer.
    tree = ast.fix_missing_locations(_FloatConstants().visit(tree))
    return compile(tree, "<expression>", "eval"), params


class _FloatConstants(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return ast.copy_location(ast.Constant(float(node.value)), node)
        return node


def make_function(code, params: list[str], variable: str = "t"):
    def f(t, *values):
        namespace = {**FUNCTIONS, **CONSTANTS, variable: t, **dict(zip(params, values))}
        with np.errstate(all="ignore"):
            out = eval(code, {"__builtins__": {}}, namespace)  # noqa: S307 - whitelisted AST only
        return np.broadcast_to(np.asarray(out, dtype=float), np.shape(t)).astype(float)

    return f


def fit_expression(t: np.ndarray, data: np.ndarray, expression: str, initial: dict | None = None, seed: int = 0) -> dict:
    """Least-squares fit of an expression to data; returns parameters with errors and model scores."""
    code, params = parse_expression(expression)
    f = make_function(code, params)
    initial = initial or {}
    p0 = np.array([float(initial.get(p, 1.0)) for p in params], dtype=float)
    n = len(t)
    if n <= len(params) + 2:
        raise ValueError("not enough points for this many parameters")
    rng = np.random.default_rng(seed)
    best = None
    starts = [p0] + [p0 * rng.uniform(0.5, 1.5, p0.size) * rng.choice([1, 1, 1, -1], p0.size) for _ in range(7)]
    for start in starts:
        try:
            if params:
                popt, pcov = curve_fit(f, t, data, p0=start, maxfev=20000)
            else:
                popt, pcov = np.array([]), np.zeros((0, 0))
            residual = data - f(t, *popt)
        except (RuntimeError, ValueError, TypeError, ArithmeticError, NameError):
            if not params:
                break
            continue
        sse = float(np.sum(residual**2))
        if np.isfinite(sse) and (best is None or sse < best[0]):
            best = (sse, popt, pcov, residual)
        if not params:
            break
    if best is None:
        raise ValueError("the fit did not converge; give better starting values in 'initial'")
    sse, popt, pcov, residual = best
    k = len(params)
    errors = np.sqrt(np.abs(np.diag(pcov))) if k else np.array([])
    sst = float(np.sum((data - data.mean()) ** 2)) or 1e-300
    sse_n = max(sse / n, 1e-300)
    dw = float(np.sum(np.diff(residual) ** 2) / max(sse, 1e-300))
    return {
        "expression": expression,
        "parameters": {p: {"value": float(v), "error": float(e) if np.isfinite(e) else None} for p, v, e in zip(params, popt, errors)},
        "points": n,
        "rms": math.sqrt(sse / n),
        "r_squared": 1 - sse / sst,
        "aic": n * math.log(sse_n) + 2 * k,
        "bic": n * math.log(sse_n) + k * math.log(n),
        "durbin_watson": dw,
        "residual_note": "about 2 means the leftovers look like random noise; near 0 means a smooth pattern is left over"
        " (also normal when tracking noise is smooth, so compare models by BIC)",
        "_f": f,
        "_popt": popt,
        "_residual": residual,
    }
