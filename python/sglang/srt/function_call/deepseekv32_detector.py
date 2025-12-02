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
        tail = r"(?:｜)?>"
        end_tail = r"(?:｜)?>"

        self.bot_pattern = re.compile(rf"<{prefix}function_calls{tail}")
        self.eot_pattern = re.compile(rf"</{prefix}function_calls{end_tail}")

        self.invoke_pattern = re.compile(
            rf"<{prefix}invoke name=\"(?P<name>[^\"]+)\"{tail}\s*(?P<body>.*?)\s*</{prefix}invoke{end_tail}",
            re.DOTALL,
        )
        self.param_pattern = re.compile(
            rf"<{prefix}parameter name=\"(?P<key>[^\"]+)\" string=\"(?P<string>true|false)\"{tail}\s*(?P<val>.*?)\s*</{prefix}parameter{end_tail}",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return bool(self.bot_pattern.search(text))

    def _parse_arguments(self, body: str) -> Dict:
        args: Dict[str, object] = {}
        for match in self.param_pattern.finditer(body):
            key = match.group("key")
            is_str = match.group("string") == "true"
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
            calls.extend(
                self.parse_base_json({"name": name, "parameters": args}, tools)
            )
        return calls

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        start_match = self.bot_pattern.search(text)
        if not start_match:
            return StreamingParseResult(normal_text=text)

        end_match = self.eot_pattern.search(text, start_match.end())
        if not end_match:
            return StreamingParseResult(normal_text=text)

        normal_text = text[: start_match.start()].strip()
        block = text[start_match.end() : end_match.start()]
        calls = self._decode_block(block, tools)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Simplified streaming: buffer until the closing </｜DSML｜function_calls> is seen,
        then parse the complete block.
        """
        self._buffer += new_text
        has_start = bool(self.bot_pattern.search(self._buffer))
        if not has_start:
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        if not self.eot_pattern.search(self._buffer):
            return StreamingParseResult()

        result = self.detect_and_parse(self._buffer, tools)
        self._buffer = ""
        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}"',
            end="</｜DSML｜invoke",
            trigger=f'<｜DSML｜invoke name="{name}"',
        )
