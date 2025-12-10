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

        # Streaming state for Devstral format
        self._devstral_mode: Optional[bool] = (
            None  # None = unknown, True = devstral, False = legacy
        )
        self._devstral_tool_calls_emitted = False
        self._devstral_current_tool_name: Optional[str] = None
        self._devstral_args_buffer = ""
        self._devstral_streamed_args = ""

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Mistral format tool call."""
        return self.bot_token in text

    def _is_devstral_format(self, text: str) -> Optional[bool]:
        """
        Check if the text uses Devstral format ([TOOL_CALLS]name[ARGS]{...})
        vs legacy format ([TOOL_CALLS] [{...}]).

        Returns:
            True if Devstral format
            False if legacy format
            None if we can't determine yet (need more text)
        """
        idx = text.find(self.bot_token)
        if idx == -1:
            return None

        # Check what comes after [TOOL_CALLS]
        after_token = text[idx + len(self.bot_token) :]
        after_stripped = after_token.lstrip()

        if not after_stripped:
            # Not enough text to determine
            return None

        # Legacy format starts with '[', Devstral format starts with function name
        if after_stripped.startswith("["):
            return False

        # If first char is a letter or underscore, it's Devstral format
        if after_stripped[0].isalpha() or after_stripped[0] == "_":
            return True

        # Unknown - shouldn't happen
        return None

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

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for both Mistral formats.

        For Devstral format ([TOOL_CALLS]name[ARGS]{...}), we do incremental
        streaming of the arguments.

        For legacy format ([TOOL_CALLS] [{...}]), we delegate to the base class.
        """
        self._buffer += new_text

        # Check if we have [TOOL_CALLS] yet
        if self.bot_token not in self._buffer:
            # Check if buffer might be a partial bot_token
            if self._ends_with_partial_token(self._buffer, self.bot_token):
                return StreamingParseResult()
            # No tool call starting, return as normal text
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        # We have [TOOL_CALLS] - determine which format (only once per stream)
        if self._devstral_mode is None:
            format_result = self._is_devstral_format(self._buffer)
            if format_result is None:
                # Can't determine yet, need more text - keep buffering
                return StreamingParseResult()
            self._devstral_mode = format_result

        # Use legacy parsing for legacy format
        if self._devstral_mode is False:
            # For legacy format, use the base class streaming
            text_to_parse = self._buffer
            self._buffer = ""
            return super().parse_streaming_increment(text_to_parse, tools)

        # Devstral format streaming parsing
        return self._parse_devstral_streaming(tools)

    def _parse_devstral_streaming(self, tools: List[Tool]) -> StreamingParseResult:
        """
        Handle streaming for Devstral format: [TOOL_CALLS]function_name[ARGS]{json}
        """
        # If we've already emitted tool calls, just consume remaining content
        if self._devstral_tool_calls_emitted:
            self._buffer = ""
            return StreamingParseResult()

        # Check if we have [TOOL_CALLS] token
        if self.bot_token not in self._buffer:
            # Could be partial, keep buffering
            if self._ends_with_partial_token(self._buffer, self.bot_token):
                return StreamingParseResult()
            # Return normal text
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        # Extract normal text before [TOOL_CALLS]
        idx = self._buffer.find(self.bot_token)
        normal_text = ""
        if idx > 0:
            normal_text = self._buffer[:idx].strip()
            self._buffer = self._buffer[idx:]

        # Check if we have [ARGS] yet
        if "[ARGS]" not in self._buffer:
            # Still waiting for function name and [ARGS]
            # Check if we might have a partial [ARGS]
            for i in range(1, len("[ARGS]")):
                if self._buffer.endswith("[ARGS]"[:i]):
                    return (
                        StreamingParseResult(normal_text=normal_text)
                        if normal_text
                        else StreamingParseResult()
                    )
            # Keep buffering
            return (
                StreamingParseResult(normal_text=normal_text)
                if normal_text
                else StreamingParseResult()
            )

        # We have [TOOL_CALLS]...[ARGS], extract function name
        match = self.devstral_pattern_lenient.search(self._buffer)
        if not match:
            return (
                StreamingParseResult(normal_text=normal_text)
                if normal_text
                else StreamingParseResult()
            )

        function_name = match.group(1)
        args_text = match.group(2).strip()

        tool_indices = self._get_tool_indices(tools)

        # Send tool name if we haven't yet
        if not self.current_tool_name_sent:
            if function_name and function_name in tool_indices:
                self.current_tool_id = 0
                self.current_tool_name_sent = True
                self._devstral_current_tool_name = function_name
                self._devstral_args_buffer = args_text
                self._devstral_streamed_args = ""
                self.streamed_args_for_tool = [""]

                result = StreamingParseResult(
                    normal_text=normal_text,
                    calls=[
                        ToolCallItem(
                            tool_index=0,
                            name=function_name,
                            parameters="",
                        )
                    ],
                )
                return result
            elif function_name and function_name not in tool_indices:
                # Unknown function, consume and ignore
                logger.warning(
                    f"Model attempted to call undefined function: {function_name}"
                )
                self._buffer = ""
                self._devstral_tool_calls_emitted = True
                return (
                    StreamingParseResult(normal_text=normal_text)
                    if normal_text
                    else StreamingParseResult()
                )
            else:
                return (
                    StreamingParseResult(normal_text=normal_text)
                    if normal_text
                    else StreamingParseResult()
                )

        # We've sent the tool name, now stream arguments
        self._devstral_args_buffer = args_text

        # Check if JSON is complete (balanced braces)
        if args_text.startswith("{"):
            brace_count = 0
            in_string = False
            escape_next = False
            json_complete = False

            for i, char in enumerate(args_text):
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
                            json_complete = True
                            break

            if json_complete:
                # Parse the complete JSON and send final arguments
                arguments = self._extract_json_object(args_text)
                args_json = json.dumps(arguments, ensure_ascii=False)

                # Calculate what we haven't sent yet
                remaining_args = args_json[len(self._devstral_streamed_args) :]

                self._devstral_tool_calls_emitted = True
                self._buffer = ""
                self.current_tool_name_sent = False
                self.current_tool_id = -1

                if remaining_args:
                    self.streamed_args_for_tool[0] = args_json
                    return StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=0,
                                parameters=remaining_args,
                            )
                        ]
                    )
                return StreamingParseResult()

            # JSON not complete yet, try to stream partial arguments
            # Only stream if we have meaningful content
            if len(args_text) > 1:  # More than just "{"
                try:
                    # Try to parse partial JSON to get streamable content
                    partial_args = self._extract_json_object(args_text)
                    if partial_args:
                        args_json = json.dumps(partial_args, ensure_ascii=False)
                        if len(args_json) > len(self._devstral_streamed_args):
                            # Find safe prefix to stream
                            new_content = args_json[len(self._devstral_streamed_args) :]
                            # Don't stream trailing incomplete parts
                            if new_content and not args_json.endswith("}"):
                                # Remove potentially incomplete trailing content
                                new_content = ""
                            if new_content:
                                self._devstral_streamed_args = args_json
                                self.streamed_args_for_tool[0] = args_json
                                return StreamingParseResult(
                                    calls=[
                                        ToolCallItem(
                                            tool_index=0,
                                            parameters=new_content,
                                        )
                                    ]
                                )
                except Exception:
                    pass

        return StreamingParseResult()

    def flush_buffered_content(self, tools: List[Tool]) -> StreamingParseResult:
        """
        Force-parse any buffered content when generation finishes.
        Handles incomplete tool calls at end of stream.
        """
        if not self._buffer:
            return StreamingParseResult()

        logger.debug(f"Flushing buffer: {repr(self._buffer[:200])}")

        # Try to parse what we have
        result = self.detect_and_parse(self._buffer, tools)
        self._buffer = ""

        # Reset streaming state
        self._devstral_mode = None
        self._devstral_tool_calls_emitted = False
        self._devstral_current_tool_name = None
        self._devstral_args_buffer = ""
        self._devstral_streamed_args = ""

        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='[TOOL_CALLS] [{"name":"' + name + '", "arguments":',
            end="}]",
            trigger="[TOOL_CALLS]",
        )
