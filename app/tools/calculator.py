from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any

from app.tools.base import ToolContext, ToolExecutionError


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass(slots=True)
class CalculatorTool:
    name: str = "calculator"

    def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        expression = str(arguments.get("expression", "")).strip()
        if not expression:
            raise ToolExecutionError(self.name, "missing expression", retryable=False)
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._evaluate(tree.body)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(self.name, f"invalid expression: {exc}", retryable=False) from exc
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__,
        }

    def _evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            return _BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            operand = self._evaluate(node.operand)
            return _UNARY_OPERATORS[type(node.op)](operand)
        raise ToolExecutionError(self.name, "unsupported expression", retryable=False)
