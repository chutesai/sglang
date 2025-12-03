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
    part1 = '<｜DSML｜function_calls><｜DSML｜invoke name="foo">'
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


def test_dsml_streaming_partial_start_token():
    detector = DeepSeekV32Detector()
    tools = _make_tools()
    part1 = "Lead <｜DS"
    part2 = (
        'ML｜function_calls><｜DSML｜invoke name="foo">'
        '<｜DSML｜parameter name="x" string="false">2</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜function_calls>"
    )

    first = detector.parse_streaming_increment(part1, tools)
    assert not first.calls
    assert first.normal_text == ""

    result = detector.parse_streaming_increment(part2, tools)
    assert result.calls
    parsed_args = json.loads(result.calls[0].parameters)
    assert parsed_args["x"] == 2
    assert result.normal_text == "Lead"


def test_dsml_streaming_missing_end_tag():
    detector = DeepSeekV32Detector()
    tools = _make_tools()

    chunks = [
        "Preface ",
        "<｜DSML｜function_calls",
        ">",
        '<｜DSML｜invoke name="foo">',
        '<｜DSML｜parameter name="x" string="false">3</｜DSML｜parameter>',
        "</｜DSML｜invoke>\n",
    ]

    first = detector.parse_streaming_increment(chunks[0], tools)
    assert first.normal_text == "Preface "
    assert not first.calls

    for chunk in chunks[1:-1]:
        res = detector.parse_streaming_increment(chunk, tools)
        assert not res.calls

    result = detector.parse_streaming_increment(chunks[-1], tools)
    assert result.calls
    parsed_args = json.loads(result.calls[0].parameters)
    assert parsed_args["x"] == 3
    assert result.normal_text == ""
