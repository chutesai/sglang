import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="foo",
                description="test",
                parameters={"type": "object", "properties": {"x": {"type": "number"}}},
            ),
        )
    ]


def test_dsml_parser_one_shot():
    detector = DeepSeekV32Detector()
    tools = _make_tools()
    text = (
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="foo">\n'
        '<｜DSML｜parameter name="x" string="false">1</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )

    result = detector.detect_and_parse(text, tools)
    assert result.calls
    assert len(result.calls) == 1
    parsed_args = json.loads(result.calls[0].parameters)
    assert parsed_args["x"] == 1


def test_dsml_parser_streaming_collects_until_end():
    detector = DeepSeekV32Detector()
    tools = _make_tools()
    part1 = "<｜DSML｜function_calls><｜DSML｜invoke name=\"foo\">"
    part2 = (
        '<｜DSML｜parameter name="x" string="true">hi</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜function_calls>"
    )

    first = detector.parse_streaming_increment(part1, tools)
    assert not first.calls  # not complete yet

    second = detector.parse_streaming_increment(part2, tools)
    assert len(second.calls) == 1
    parsed_args = json.loads(second.calls[0].parameters)
    assert parsed_args["x"] == "hi"
