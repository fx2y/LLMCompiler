import logging
import math
from functools import lru_cache
from typing import List, Optional, Union

import numexpr
import numpy as np
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_MATH_DESCRIPTION = (
    "math(problem: str, context: Optional[list[str]]) -> float:\n"
    " - Solves the provided math problem.\n"
    ' - `problem` can be either a simple math problem (e.g. "1 + 3") or a word problem (e.g. "how many apples are there if there are 3 apples and 2 apples").\n'
    " - You cannot calculate multiple expressions in one call. For instance, `math('1 + 3, 2 + 4')` does not work. "
    "If you need to calculate multiple expressions, you need to call them separately like `math('1 + 3')` and then `math('2 + 4')`\n"
    " - Minimize the number of `math` actions as much as possible. For instance, instead of calling "
    '2. math("what is the 10% of $1") and then call 3. math("$1 + $2"), '
    'you MUST call 2. math("what is the 110% of $1") instead, which will reduce the number of math actions.\n'
    # Context specific rules below
    " - You can optionally provide a list of strings as `context` to help the agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `math` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do math on it.\n"
    " - You MUST NEVER provide `search` type action's outputs as a variable in the `problem` argument. "
    "This is because `search` returns a text blob that contains the information about the entity, not a number or value. "
    "Therefore, when you need to provide an output of `search` action, you MUST provide it as a `context` argument to `math` action. "
    'For example, 1. search("Barack Obama") and then 2. math("age of $1") is NEVER allowed. '
    'Use 2. math("age of Barack Obama", context=["$1"]) instead.\n'
    " - When you ask a question about `context`, specify the units. "
    'For instance, "what is xx in height?" or "what is xx in millions?" instead of "what is xx?"\n'
)

_SYSTEM_PROMPT = """Translate a math problem into an expression that can be executed using Python's numexpr library. Provide detailed reasoning and any intermediate steps.

Question: ${{Question with math problem.}}

Reasoning:
1. Problem interpretation: ${{Explain how you understand the problem}}
2. Solution approach: ${{Describe your step-by-step approach to solving the problem}}
3. Mathematical concepts: ${{Explain any relevant mathematical concepts}}
4. Context incorporation: ${{If applicable, explain how you're using the provided context}}
5. Final expression justification: ${{Justify your final mathematical expression}}

Intermediate steps:
${{List any intermediate calculations or transformations, if necessary}}

Code:
```
${{single line mathematical expression that solves the problem}}
```
...numexpr.evaluate(text)...
```output
${{Output of running the code}}
```
Answer: ${{Answer}}

Begin.

Question: What is 37593 * 67?
ExecuteCode({{code: "37593 * 67"}})
...numexpr.evaluate("37593 * 67")...
```output
2518731
```
Answer: 2518731

Question: 37593^(1/5)
ExecuteCode({{code: "37593**(1/5)"}})
...numexpr.evaluate("37593**(1/5)")...
```output
8.222831614237718
```
Answer: 8.222831614237718
"""

_ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions.\
    Use it to substitute into any ${{#}} variables or other words in the problem.\
    \n\n${context}\n\nNote that context variables are not defined in code yet.\
You must extract the relevant numbers and directly put them in code."""


class ExecuteCode(BaseModel):
    """The input to the numexpr.evaluate() function."""
    reasoning: str = Field(
        ...,
        description="Detailed reasoning behind the code expression, including:\n"
                    "1. Problem interpretation\n"
                    "2. Step-by-step solution approach\n"
                    "3. Explanation of any mathematical concepts used\n"
                    "4. How context is incorporated (if applicable)\n"
                    "5. Justification for the final expression",
    )
    code: str = Field(
        ...,
        description="The simple code expression to execute by numexpr.evaluate().",
    )
    intermediate_steps: List[str] = Field(
        default_factory=list,
        description="List of intermediate calculations or transformations, if any.",
    )


class MathEvaluationError(Exception):
    """Custom exception for math evaluation errors."""
    pass


@lru_cache(maxsize=100)
def _cached_evaluate_expression(expression: str) -> Union[float, int]:
    """
    Cached version of expression evaluation using numexpr.

    Args:
        expression (str): The mathematical expression to evaluate.

    Returns:
        Union[float, int]: The result of the evaluation.

    Raises:
        MathEvaluationError: If the expression cannot be evaluated.
    """
    return _evaluate_expression(expression)


def _evaluate_expression(expression: str) -> Union[float, int]:
    """
    Safely evaluate a mathematical expression using numexpr.

    Args:
        expression (str): The mathematical expression to evaluate.

    Returns:
        Union[float, int]: The result of the evaluation.

    Raises:
        MathEvaluationError: If the expression cannot be evaluated.
    """
    logger.info(f"Evaluating expression: {expression}")
    try:
        local_dict = {
            # Constants
            "pi": math.pi,
            "e": math.e,
            "inf": math.inf,
            "nan": math.nan,
            # Trigonometric functions
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            # Hyperbolic functions
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "asinh": math.asinh,
            "acosh": math.acosh,
            "atanh": math.atanh,
            # Exponential and logarithmic functions
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            # Power and roots
            "sqrt": math.sqrt,
            "cbrt": np.cbrt,
            # Rounding and absolute value
            "ceil": math.ceil,
            "floor": math.floor,
            "abs": abs,
            # Other functions
            "factorial": math.factorial,
            "degrees": math.degrees,
            "radians": math.radians,
        }
        result = numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical constants and functions
        )

        # Convert numpy types to Python native types
        if isinstance(result, (np.ndarray, np.number)):
            if getattr(result, 'size', 1) > 1:
                raise MathEvaluationError("Expression resulted in an array. Please provide a scalar expression.")
            result = result.item()

        # Check for invalid results
        if not np.isfinite(result):
            if np.isnan(result):
                raise MathEvaluationError("Result is Not a Number (NaN). Check your expression for invalid operations.")
            raise MathEvaluationError(
                "Result is Infinity. Check your expression for division by zero or very large numbers.")

        logger.info(f"Expression result: {result}")
        return result
    except Exception as e:
        logger.error(f'Failed to evaluate "{expression}". Error: {str(e)}')
        raise MathEvaluationError(
            f'Failed to evaluate "{expression}". Error: {str(e)}. '
            "Please provide a valid numerical expression."
        )


def get_math_tool(llm: ChatOpenAI) -> StructuredTool:
    """
    Create and return a StructuredTool for mathematical calculations.

    Args:
        llm (ChatOpenAI): The language model to use for code generation.

    Returns:
        StructuredTool: A tool that can be used for mathematical calculations.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )
    extractor = prompt | llm.with_structured_output(ExecuteCode)

    # extractor = create_structured_output_runnable(ExecuteCode, llm, prompt)

    def calculate_expression(
            problem: str,
            context: Optional[List[str]] = None,
            config: Optional[RunnableConfig] = None,
    ) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            problem (str): The mathematical problem to solve.
            context (Optional[List[str]]): Additional context for the problem.
            config (Optional[RunnableConfig]): Configuration for the runnable.

        Returns:
            str: The result of the calculation or an error message.
        """
        logger.info(f"Received problem: {problem}")
        if context:
            logger.info(f"Received context: {context}")

        # Input validation
        if not isinstance(problem, str) or not problem.strip():
            logger.warning("Invalid 'problem' input")
            return "Error: 'problem' must be a non-empty string."

        if context is not None:
            if not isinstance(context, list):
                logger.warning("Invalid 'context' input: not a list")
                return "Error: 'context' must be a list of strings or None."
            if not all(isinstance(item, str) for item in context):
                logger.warning("Invalid 'context' input: not all items are strings")
                return "Error: All items in 'context' must be strings."

        chain_input = {"problem": problem}
        if context:
            context_str = "\n".join(context)
            if context_str.strip():
                context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
                    context=context_str.strip()
                )
            chain_input["context"] = [SystemMessage(content=context_str)]

        try:
            code_model = extractor.invoke(chain_input, config)
            logger.info(f"Generated code: {code_model.code}")
            logger.info(f"Reasoning: {code_model.reasoning}")
            result = _cached_evaluate_expression(code_model.code)  # Use cached version
            logger.info(f"Calculation result: {result}")
            return str(result)
        except MathEvaluationError as e:
            logger.error(f"MathEvaluationError: {str(e)}")
            return str(e)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"

    return StructuredTool.from_function(
        name="math",
        func=calculate_expression,
        description=_MATH_DESCRIPTION,
    )
