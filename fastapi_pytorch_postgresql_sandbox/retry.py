"""retry"""

# This module provides function with commmon parameters for the tenacity.retry
# decorator

from typing import Any, Dict

import tenacity

TenacityParameters = Dict[str, Any]


def _base_parameters() -> TenacityParameters:
    return {
        "stop": tenacity.stop_after_attempt(10),
    }


def linear_backoff(**kwargs: Any) -> TenacityParameters:
    """
    Returns parameters for tenacity.retry that configure a linear backoff.

    This can be used as follows:
        @tenacity.retry(**linear_backoff())
        def function_to_retry()

    Keyword arguments for tenacity.retry can be passed to this function to
    override arguments as desired.
    """
    return {**_base_parameters(), **{"wait": tenacity.wait_fixed(3)}, **kwargs}


def exponential_backoff(**kwargs: Any) -> TenacityParameters:
    """
    Returns parameters for tenacity.retry that configure an exponential backoff.

    This can be used as follows:
        @tenacity.retry(**exponential_backoff())
        def function_to_retry()

    Keyword arguments for tenacity.retry can be passed to this function to
    override arguments as desired.
    """
    return {
        **_base_parameters(),
        **{"wait": tenacity.wait_exponential(min=1, max=15, multiplier=2)},
        **kwargs,
    }


def is_result_none(value: Any) -> bool:
    """
    Function that can be assigned to the retry_if_result paramter to
    tenacity.retry to conditionally retry if None is returned.
    """
    # https://github.com/jd/tenacity#whether-to-retry
    return value is None


def return_outcome_result(retry_state: tenacity.RetryCallState) -> Any:
    """
    Callback that can be assigned to the retry_error_callback parameter to
    tenacity.retry to return the last return value instead of raising RetryError
    when the retry stop condition is reached.
    """
    # https://github.com/jd/tenacity#custom-callbacks
    return None if retry_state.outcome is None else retry_state.outcome.result()
