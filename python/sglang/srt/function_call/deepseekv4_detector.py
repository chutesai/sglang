import re

from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek-V4 DSML tool call format.

    Identical to V3.2 except the outer block tag is "tool_calls" instead of
    "function_calls":

        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="tool">
        <｜DSML｜parameter name="key" string="true|false">value</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>

    All invoke/parameter parsing is inherited unchanged from DeepSeekV32Detector.
    """

    def __init__(self):
        super().__init__()

        prefix = r"(?:｜\s*DSML\s*｜)?"
        tail = r"(?:｜)?\s*>"
        end_tail = r"(?:｜)?\s*>"
        flags = re.IGNORECASE

        # Override only the outer block patterns: function_calls -> tool_calls
        self.bot_pattern = re.compile(rf"<\s*{prefix}tool_calls{tail}", flags)
        self.eot_pattern = re.compile(rf"</\s*{prefix}tool_calls{end_tail}", flags)

        self._start_tokens = [
            "<tool_calls",
            "<｜dsml｜tool_calls",
        ]
