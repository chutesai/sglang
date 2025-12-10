import json
import logging
import re
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral model function call formats.

    Supports two format variants:

    1. Legacy Mistral format (Mistral-7B-Instruct-v0.3):
    ```
    [TOOL_CALLS] [{"name": "function_name", "arguments": {json_args}}, ...]
    ```

    2. Devstral format (Devstral-Small-2505, Devstral-2-123B-Instruct-2512):
    ```
    [TOOL_CALLS]function_name[ARGS]{json_args}
    ```

    References:
    - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3?chat_template=default
    - https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512/raw/main/chat_template.jinja
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        # Common token for both formats
        self.bot_token = "[TOOL_CALLS]"
        self.eot_token = "]"
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        self.tool_call_separator = ", "

        # Devstral format pattern: [TOOL_CALLS]function_name[ARGS]{json_args}
        # Allow optional newline after [TOOL_CALLS] and make closing brace optional for incomplete output
        self.devstral_pattern = re.compile(
            r"\[TOOL_CALLS\]\s*\n?([a-zA-Z_][a-zA-Z0-9_]*)\[ARGS\](\{.*?)(?:\}|$)",
            re.DOTALL,
        )
        # More lenient pattern that captures everything after [ARGS]
        self.devstral_pattern_lenient = re.compile(
            r"\[TOOL_CALLS\]\s*\n?([a-zA-Z_][a-zA-Z0-9_]*)\[ARGS\](.*)",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Mistral format tool call."""
        return self.bot_token in text

    def _is_devstral_format(self, text: str) -> bool:
        """
        Check if the text uses Devstral format ([TOOL_CALLS]name[ARGS]{...})
        vs legacy format ([TOOL_CALLS] [{...}]).
        """
        idx = text.find(self.bot_token)
        if idx == -1:
            return False

        # Check what comes after [TOOL_CALLS]
        after_token = text[idx + len(self.bot_token) :].lstrip()
        # Legacy format starts with '[', Devstral format starts with function name or newline then name
        if after_token.startswith("["):
            return False
        # Devstral format: function name follows (possibly after newline)
        return "[ARGS]" in text

    def _parse_devstral_format(
        self, text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Parse the Devstral format: [TOOL_CALLS]function_name[ARGS]{json_args}

        Handles:
        - Multiple tool calls (each starting with [TOOL_CALLS])
        - Missing closing braces (incomplete JSON)
        - Newlines between [TOOL_CALLS] and function name
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else ""

        calls = []
        tool_indices = self._get_tool_indices(tools)

        # Find all tool calls in the text
        # Pattern: [TOOL_CALLS]\n?function_name[ARGS]{json}
        remaining = text[idx:] if idx != -1 else text

        tool_call_idx = 0
        while self.bot_token in remaining:
            match = self.devstral_pattern_lenient.search(remaining)
            if not match:
                break

            function_name = match.group(1)
            args_text = match.group(2).strip()

            # Try to extract valid JSON from args_text
            arguments = self._extract_json_object(args_text)

            if function_name:
                if function_name not in tool_indices:
                    logger.warning(
                        f"Model attempted to call undefined function: {function_name}"
                    )
                else:
                    calls.append(
                        ToolCallItem(
                            tool_index=tool_call_idx,
                            name=function_name,
                            parameters=json.dumps(arguments, ensure_ascii=False),
                        )
                    )
                    tool_call_idx += 1

            # Move past this match to find more tool calls
            match_end = match.end()
            # Check if there's another [TOOL_CALLS] after this one
            next_tool_call = remaining.find(self.bot_token, len(self.bot_token))
            if next_tool_call != -1:
                remaining = remaining[next_tool_call:]
            else:
                break

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _extract_json_object(self, text: str) -> dict:
        """
        Extract a JSON object from text, handling incomplete JSON gracefully.

        :param text: Text that should start with '{' and contain JSON
        :return: Parsed dict or empty dict if parsing fails
        """
        if not text:
            return {}

        text = text.strip()
        if not text.startswith("{"):
            return {}

        # First try to parse as-is
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find a complete JSON object using brace counting
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[: i + 1])
                        except json.JSONDecodeError:
                            pass
                        break

        # If we couldn't find complete JSON, try adding closing brace
        if brace_count > 0:
            try:
                return json.loads(text + "}" * brace_count)
            except json.JSONDecodeError:
                pass

        return {}

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        Supports both legacy Mistral format and Devstral format.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Check which format we're dealing with
        if self._is_devstral_format(text):
            return self._parse_devstral_format(text, tools)

        # Legacy format: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
        # Extract the JSON array part from [TOOL_CALLS] [...]
        # Use bracket counting to properly handle nested brackets in JSON content
        json_array_str = self._extract_json_array(text)
        if not json_array_str:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            function_call_arr = json.loads(json_array_str)
            # Handle both single object and array of objects
            if not isinstance(function_call_arr, list):
                function_call_arr = [function_call_arr]
            calls = self.parse_base_json(function_call_arr, tools)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON part: {json_array_str}, JSON parse error: {str(e)}"
            )

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _extract_json_array(self, text: str) -> Optional[str]:
        """
        Extract the JSON array part using bracket counting to handle nested brackets.

        For legacy format: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]

        :param text: The complete text containing [TOOL_CALLS] [...]
        :return: The JSON array string or None if not found
        """
        start_idx = text.find(self.bot_token)
        if start_idx == -1:
            return None

        # Find the opening bracket after [TOOL_CALLS]
        # Legacy format has " [" after [TOOL_CALLS]
        after_token = text[start_idx + len(self.bot_token) :]
        bracket_pos = after_token.find("[")
        if bracket_pos == -1:
            return None

        json_start = start_idx + len(self.bot_token) + bracket_pos
        bracket_count = 0
        in_string = False
        escape_next = False

        for i in range(json_start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[json_start : i + 1]

        # Handle incomplete JSON - try to close brackets
        if bracket_count > 0:
            incomplete_json = text[json_start:]
            try:
                # Try adding closing brackets
                return incomplete_json + "]" * bracket_count
            except Exception:
                pass

        return None

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='[TOOL_CALLS] [{"name":"' + name + '", "arguments":',
            end="}]",
            trigger="[TOOL_CALLS]",
        )
