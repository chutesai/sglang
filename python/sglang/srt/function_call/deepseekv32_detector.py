import json
import logging
import re
from typing import Dict, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek-V3.2 DSML function call format.

    Format:
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="tool">
    <｜DSML｜parameter name="key" string="true|false">value</｜DSML｜parameter>
    ...
    </｜DSML｜invoke>
    ...
    </｜DSML｜function_calls>
    """

    def __init__(self):
        super().__init__()
        prefix = r"(?:｜\s*DSML\s*｜)?"
        tail = r"(?:｜)?\s*>"
        end_tail = r"(?:｜)?\s*>"

        flags = re.IGNORECASE

        self.bot_pattern = re.compile(rf"<\s*{prefix}function_calls{tail}", flags)
        self.eot_pattern = re.compile(rf"</\s*{prefix}function_calls{end_tail}", flags)
        self._start_tokens = [
            "<function_calls",
            "<｜dsml｜function_calls",
        ]
        # Track whether we've emitted tool calls and should consume remaining tags
        self._tool_calls_emitted = False

        self.invoke_pattern = re.compile(
            rf"<\s*{prefix}invoke\s+name\s*=\s*[\"'](?P<name>[^\"'>]+)[\"']{tail}\s*(?P<body>.*?)\s*</\s*{prefix}invoke{end_tail}",
            re.DOTALL | flags,
        )
        self.param_pattern = re.compile(
            rf"<\s*{prefix}parameter\s+name\s*=\s*[\"'](?P<key>[^\"'>]+)[\"'](?:\s+string\s*=\s*[\"'](?P<string>true|false)[\"'])?\s*{tail}\s*(?P<val>.*?)\s*</\s*{prefix}parameter{end_tail}",
            re.DOTALL | flags,
        )

    def has_tool_call(self, text: str) -> bool:
        return bool(self.bot_pattern.search(text))

    def _parse_arguments(self, body: str) -> Dict:
        args: Dict[str, object] = {}
        for match in self.param_pattern.finditer(body):
            key = match.group("key")
            string_flag = (match.group("string") or "true").lower()
            is_str = string_flag == "true"
            raw_val = match.group("val")
            if is_str:
                args[key] = raw_val
            else:
                try:
                    args[key] = json.loads(raw_val)
                except Exception:
                    args[key] = raw_val
        return args

    def _decode_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        calls = []
        for match in self.invoke_pattern.finditer(block):
            name = match.group("name")
            args = self._parse_arguments(match.group("body"))
            parsed_calls = self.parse_base_json(
                {"name": name, "parameters": args}, tools
            )
            # Update tool_index to be the sequential position in the response
            for call in parsed_calls:
                call.tool_index = len(calls)
                calls.append(call)
        return calls

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
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
            # Check if there are complete invoke blocks (even if undefined/invalid)
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
        Simplified streaming: buffer until the closing </｜DSML｜function_calls> is seen,
        then parse the complete block. After emitting tool calls, consume any remaining
        tool-related tags.
        """
        self._buffer += new_text

        # If we've already emitted tool calls, consume any remaining tool-related tags
        if self._tool_calls_emitted:
            # Check if we've hit the closing tag
            end_match = self.eot_pattern.search(self._buffer)
            if end_match:
                # Found closing tag, consume it and reset
                self._buffer = self._buffer[end_match.end() :]
                self._tool_calls_emitted = False
                # Return any remaining text after the closing tag
                if self._buffer:
                    remaining = self._buffer
                    self._buffer = ""
                    return StreamingParseResult(normal_text=remaining)
                return StreamingParseResult()

            # Keep consuming until we see the complete closing tag
            # Just buffer everything and return empty - the closing tag will eventually arrive
            return StreamingParseResult()

        # No start token yet; keep buffering if current buffer could be a partial prefix
        # of the start token (e.g. "<function" across chunks).
        if not self.bot_pattern.search(self._buffer):
            buffer_low = self._buffer.lower()
            for token in self._start_tokens:
                token_low = token.lower()
                # Treat both strict prefixes and the full token (without the closing '>') as partial
                if buffer_low.endswith(token_low) or self._ends_with_partial_token(
                    buffer_low, token_low
                ):
                    return StreamingParseResult()

            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        if self.eot_pattern.search(self._buffer):
            result = self.detect_and_parse(self._buffer, tools)
            self._buffer = ""
            # No need to set _tool_calls_emitted here since we already have the closing tag
            return result

        # Fallback: if we have a complete invoke block that reaches the current end
        # of the buffer but the model never emitted </function_calls>, parse what we
        # have so far.
        # Search in the trimmed buffer to avoid issues with trailing whitespace
        trimmed_buffer = self._buffer.rstrip()
        if trimmed_buffer:
            last_invoke_end = None
            for match in self.invoke_pattern.finditer(trimmed_buffer):
                last_invoke_end = match.end()

            if last_invoke_end is not None and last_invoke_end == len(trimmed_buffer):
                result = self.detect_and_parse(self._buffer, tools)
                self._buffer = ""
                # Mark that we've processed tool call markup - consume any remaining tags
                # Set flag even if calls are empty (undefined/invalid tools) since we found invoke blocks
                self._tool_calls_emitted = True
                return result

        return StreamingParseResult()

    def flush_buffered_content(self, tools: List[Tool]) -> StreamingParseResult:
        """
        Force-parse any buffered content when generation finishes.
        This handles the case where the model generates a complete tool call
        but doesn't emit the closing </｜DSML｜function_calls> tag before stopping.
        """
        if not self._buffer:
            return StreamingParseResult()

        # Try to parse what we have, even without the closing tag
        result = self.detect_and_parse(self._buffer, tools)
        self._buffer = ""

        # Reset state
        self._tool_calls_emitted = False

        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}"',
            end="</｜DSML｜invoke",
            trigger=f'<｜DSML｜invoke name="{name}"',
        )
