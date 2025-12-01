import json
import logging
import re
from typing import Dict, List, Tuple

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
        self.bot_token = "<｜DSML｜function_calls>"
        self.eot_token = "</｜DSML｜function_calls>"
        self.invoke_pattern = re.compile(
            r'<｜DSML｜invoke name="(?P<name>[^"]+)">(?P<body>.*?)</｜DSML｜invoke>',
            re.DOTALL,
        )
        self.param_pattern = re.compile(
            r'<｜DSML｜parameter name="(?P<key>[^"]+)" string="(?P<string>true|false)">(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def _parse_arguments(self, body: str) -> Dict:
        args: Dict[str, object] = {}
        for match in self.param_pattern.finditer(body):
            key = match.group("key")
            is_str = match.group("string") == "true"
            raw_val = match.group(3)
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
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        end_idx = text.find(self.eot_token, idx)
        if end_idx == -1:
            # Incomplete block; keep everything as normal text
            return StreamingParseResult(normal_text=text)

        block = text[idx + len(self.bot_token) : end_idx]
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
        if self.bot_token not in self._buffer:
            # No DSML block yet; flush plain text
            normal_text = self._buffer
            self._buffer = ""
            if self.eot_token in normal_text:
                normal_text = normal_text.replace(self.eot_token, "")
            return StreamingParseResult(normal_text=normal_text)

        if self.eot_token not in self._buffer:
            return StreamingParseResult()

        # We have a full block; parse once
        result = self.detect_and_parse(self._buffer, tools)
        self._buffer = ""
        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger=f'<｜DSML｜invoke name="{name}">',
        )
