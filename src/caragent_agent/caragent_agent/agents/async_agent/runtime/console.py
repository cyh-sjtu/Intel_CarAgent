"""Console-display helpers for the async agent."""


class Colors:
    """Centralize ANSI color codes used by async-agent console output."""

    USER = "\033[94m"
    ORCHESTRATE = "\033[94m"
    PLAN = "\033[95m"
    REACT = "\033[92m"
    TOOL = "\033[93m"
    RESULT = "\033[92m"
    ANSWER = "\033[1;96m"
    ERROR = "\033[91m"
    RESET = "\033[0m"


__all__ = ["Colors"]
