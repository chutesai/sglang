import json
import logging
import re
from typing import Dict, List

from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _find_common_prefix, _partial_json_loads

logger = logging.getLogger(__name__)


def _remove_suffix(text: str, suffix: str) -> str:
    """Remove a literal suffix from text (unlike rstrip which removes character sets)."""
    if suffix and text.endswith(suffix):
        return text[: -len(suffix)]
    return text


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek-V3.2 DSML function call format.

    Supports two parameter encodings inside each invoke block:
    1) XML parameter tags (original DeepSeek V3.2 format):
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="tool">
    <｜DSML｜parameter name="key" string="true|false">value</｜DSML｜parameter>
    ...
    </｜DSML｜invoke>
    ...
    </｜DSML｜function_calls>

    2) Direct JSON object
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="tool">
    {"key": "value"}
    </｜DSML｜invoke>
    ...
    </｜DSML｜function_calls>

    Features:
    - Robust regex matching: tolerates whitespace, missing quotes, case variations
    - Supports plain format (without DSML tokens) for compatibility
    - Incremental streaming: streams tool names and arguments as they arrive
    - Handles missing closing tags gracefully via flush_buffered_content
    """

    def __init__(self):
        super().__init__()
        # Flexible prefix: supports both plain format and DSML format
        prefix = r"(?:｜\s*DSML\s*｜)?"
        tail = r"(?:｜)?\s*>"
        end_tail = r"(?:｜)?\s*>"

        flags = re.IGNORECASE

        # Compiled patterns for robust matching
        self.bot_pattern = re.compile(rf"<\s*{prefix}function_calls{tail}", flags)
        self.eot_pattern = re.compile(rf"</\s*{prefix}function_calls{end_tail}", flags)

        # Start tokens for partial-prefix detection during streaming
        self._start_tokens = [
            "<function_calls",
            "<｜dsml｜function_calls",
        ]

        # Match complete invoke blocks (for one-shot parsing)
        # Make closing quote optional to handle malformed output
        self.invoke_pattern = re.compile(
            rf"<\s*{prefix}invoke\s+name\s*=\s*[\"'](?P<name>[^\"'>]+)[\"']?{tail}\s*(?P<body>.*?)\s*</\s*{prefix}invoke{end_tail}",
            re.DOTALL | flags,
        )

        # Match invoke blocks that may be partial (for streaming)
        # end group is closing tag or end of string
        self.invoke_pattern_streaming = re.compile(
            rf"<\s*{prefix}invoke\s+name\s*=\s*[\"'](?P<name>[^\"'>]+)[\"']?{tail}\s*(?P<body>.*?)(?P<end></\s*{prefix}invoke{end_tail}|$)",
            re.DOTALL | flags,
        )

        # Match complete parameter blocks
        self.param_pattern = re.compile(
            rf"<\s*{prefix}parameter\s+name\s*=\s*[\"'](?P<key>[^\"'>]+)[\"']?(?:\s+string\s*=\s*[\"'](?P<string>true|false)[\"']?)?\s*{tail}\s*(?P<val>.*?)\s*</\s*{prefix}parameter{end_tail}",
            re.DOTALL | flags,
        )

        # Match partial parameter blocks (for streaming, no closing tag required)
        self.param_pattern_partial = re.compile(
            rf"<\s*{prefix}parameter\s+name\s*=\s*[\"'](?P<key>[^\"'>]+)[\"']?(?:\s+string\s*=\s*[\"'](?P<string>true|false)[\"']?)?\s*{tail}\s*(?P<val>.*)$",
            re.DOTALL | flags,
        )

        # Tokens for stripping partial closing tags during streaming
        self.prefix_parameter_end_call = ["</", "｜DSML｜", "parameter"]
        self.prefix_invoke_end_call = ["</", "｜DSML｜", "inv", "oke"]

    def has_tool_call(self, text: str) -> bool:
        return bool(self.bot_pattern.search(text))

    def _parse_arguments(self, body: str) -> Dict:
        """Parse arguments from an invoke body (one-shot, returns Dict)."""
        # First, try the direct JSON format for the entire invoke body
        stripped_body = body.strip()
        if stripped_body.startswith("{") and stripped_body.endswith("}"):
            try:
                parsed = json.loads(stripped_body)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                # Fall back to DSML parameter parsing on JSON errors
                logger.debug(
                    "DeepSeekV32Detector: JSON parameter parse failed", exc_info=True
                )

        args: Dict[str, object] = {}
        for match in self.param_pattern.finditer(body):
            key = match.group("key").strip()
            string_flag = (match.group("string") or "true").lower().strip()
            is_str = string_flag == "true"
            raw_val = match.group("val").strip()
            if is_str:
                args[key] = raw_val
            else:
                try:
                    args[key] = json.loads(raw_val)
                except Exception:
                    args[key] = raw_val
        return args

    def _parse_arguments_streaming(self, body: str, allow_partial: bool = False) -> str:
        """Parse arguments from an invoke body for streaming (returns JSON str).

        When allow_partial=True, handles incomplete parameter tags and JSON."""
        stripped_body = body.strip()
        if stripped_body.startswith("{"):
            if allow_partial:
                # Remove incomplete invoke end call prefix
                for token in reversed(self.prefix_invoke_end_call):
                    stripped_body = _remove_suffix(stripped_body, token)
                return stripped_body
            elif stripped_body.endswith("}"):
                return stripped_body

        # Fall back to XML parameter tag parsing
        parameters: Dict[str, object] = {}
        param_matches = list(self.param_pattern.finditer(body))
        last_match_end = 0

        for match in param_matches:
            key = match.group("key").strip()
            string_flag = (match.group("string") or "true").lower().strip()
            is_str = string_flag == "true"
            raw_val = match.group("val").strip()
            last_match_end = match.end()

            if is_str:
                parameters[key] = raw_val
            else:
                try:
                    parameters[key] = json.loads(raw_val)
                except (json.JSONDecodeError, ValueError):
                    parameters[key] = raw_val

        # If allowed, try to parse a partial parameter at the end
        if allow_partial:
            remaining = body[last_match_end:]
            for token in reversed(self.prefix_parameter_end_call):
                remaining = _remove_suffix(remaining, token)

            partial_match = self.param_pattern_partial.search(remaining)
            if partial_match and (param_value := partial_match.group("val")):
                param_name = partial_match.group("key").strip()
                string_flag = (partial_match.group("string") or "true").lower().strip()
                if string_flag == "true":
                    parameters[param_name] = param_value.strip()
                else:
                    try:
                        parameters[param_name] = _partial_json_loads(
                            param_value, Allow.ALL
                        )[0]
                    except json.JSONDecodeError:
                        parameters[param_name] = param_value.strip()

        return json.dumps(parameters, ensure_ascii=False)

    def _decode_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        """Decode all invoke blocks within a function_calls block (one-shot)."""
        calls = []
        for match in self.invoke_pattern.finditer(block):
            name = match.group("name").strip()
            args = self._parse_arguments(match.group("body"))
            parsed_calls = self.parse_base_json(
                {"name": name, "parameters": args}, tools
            )
            for call in parsed_calls:
                call.tool_index = len(calls)
                calls.append(call)
        return calls

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parsing with robust handling of missing closing tags."""
        start_match = self.bot_pattern.search(text)
        if not start_match:
            return StreamingParseResult(normal_text=text)

        end_match = self.eot_pattern.search(text, start_match.end())
        block_end = end_match.start() if end_match else len(text)

        normal_text = text[: start_match.start()].strip()
        block = text[start_match.end() : block_end]
        calls = self._decode_block(block, tools)

        # If we couldn't find an end tag and also didn't decode anything,
        # check if we at least found complete invoke blocks
        if not end_match and not calls:
            has_complete_invoke = bool(self.invoke_pattern.search(block))
            if has_complete_invoke:
                # We found invoke blocks but they were undefined/invalid
                # Consume the markup instead of leaking it as normal text
                return StreamingParseResult(normal_text=normal_text, calls=[])
            else:
                # No complete invoke blocks, treat as normal text
                return StreamingParseResult(normal_text=text)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with argument streaming support.

        Streams tool names immediately when seen and incrementally delivers
        argument diffs as they arrive. Uses robust compiled regex from HEAD
        with incremental argument streaming from upstream.
        """
        self._buffer += new_text

        # Phase 1: Haven't entered function_calls block yet
        if self.current_tool_id < 0:
            if not self.bot_pattern.search(self._buffer):
                # Check for partial prefix of start token
                buffer_low = self._buffer.lower()
                for token in self._start_tokens:
                    token_low = token.lower()
                    if buffer_low.endswith(token_low) or self._ends_with_partial_token(
                        buffer_low, token_low
                    ):
                        return StreamingParseResult()
                # No tool call starting, emit as normal text
                normal_text = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text)

        # Phase 2: Inside function_calls, process invoke blocks
        current_text = self._buffer
        all_calls: List[ToolCallItem] = []

        try:
            while True:
                invoke_match = self.invoke_pattern_streaming.search(current_text)
                if not invoke_match:
                    break

                func_name = invoke_match.group("name").strip()
                invoke_content = invoke_match.group("body")
                end_group = invoke_match.group("end")
                is_tool_end = bool(end_group) and bool(end_group.strip())

                # Initialize state on first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure arrays are large enough
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                # Send tool name if not sent yet
                if not self.current_tool_name_sent:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True

                # Parse parameters (partial or complete)
                current_params = self._parse_arguments_streaming(
                    invoke_content, allow_partial=not is_tool_end
                )

                # Calculate incremental argument diff
                sent_len = len(self.streamed_args_for_tool[self.current_tool_id])
                prev_params = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )

                argument_diff = None

                if is_tool_end:
                    # Complete: send everything remaining
                    argument_diff = current_params[sent_len:]
                elif prev_params is not None:
                    # Partial: send stable prefix diff
                    if current_params != prev_params:
                        prefix = _find_common_prefix(current_params, prev_params)
                        if len(prefix) > sent_len:
                            argument_diff = prefix[sent_len:]

                if argument_diff:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                # Update stored arguments
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": current_params,
                }

                if is_tool_end:
                    # Remove completed invoke block from buffer
                    current_text = current_text[invoke_match.end() :]
                    self._buffer = current_text
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    continue
                else:
                    break

            # Check for closing function_calls tag
            eot_match = self.eot_pattern.search(self._buffer)
            if eot_match:
                self._buffer = self._buffer[eot_match.end() :]
                # Reset streaming state
                self.current_tool_id = -1
                self.current_tool_name_sent = False

            return StreamingParseResult(normal_text="", calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text="")

    def flush_buffered_content(self, tools: List[Tool]) -> StreamingParseResult:
        """
        Force-parse any buffered content when generation finishes.
        This handles the case where the model generates a complete tool call
        but doesn't emit the closing </｜DSML｜function_calls> tag before stopping.
        """
        if not self._buffer:
            return StreamingParseResult()

        logger.debug(f"Flushing buffer: {len(self._buffer)} chars")

        # Try to parse what we have, even without the closing tag
        result = self.detect_and_parse(self._buffer, tools)
        self._buffer = ""

        # Reset streaming state
        self.current_tool_id = -1
        self.current_tool_name_sent = False

        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger="<｜DSML｜invoke",
        )
