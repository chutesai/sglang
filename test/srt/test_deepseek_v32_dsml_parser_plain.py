import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="test",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        )
    ]


def test_plain_function_calls_parses():
    detector = DeepSeekV32Detector()
    tools = _make_tools()
    text = (
        "<function_calls>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="city" string="true">Berlin</parameter>\n'
        "</invoke>\n"
        "</function_calls>"
    )

    result = detector.detect_and_parse(text, tools)
    assert result.calls
    parsed_args = json.loads(result.calls[0].parameters)
    assert parsed_args["city"] == "Berlin"


def test_plain_streaming():
    detector = DeepSeekV32Detector()
    tools = _make_tools()
    part1 = '<function_calls><invoke name="get_weather">'
    part2 = '<parameter name="city" string="true">Paris</parameter></invoke></function_calls>'

    assert not detector.parse_streaming_increment(part1, tools).calls
    result = detector.parse_streaming_increment(part2, tools)
    assert result.calls
    parsed_args = json.loads(result.calls[0].parameters)
    assert parsed_args["city"] == "Paris"


def test_plain_streaming_partial_start_token():
    detector = DeepSeekV32Detector()
    tools = _make_tools()
    part1 = "Here is the tool <funct"
    part2 = (
        'ion_calls><invoke name="get_weather">'
        '<parameter name="city" string="true">Rome</parameter>'
        "</invoke></function_calls>"
    )

    first = detector.parse_streaming_increment(part1, tools)
    assert not first.calls
    assert first.normal_text == ""

    result = detector.parse_streaming_increment(part2, tools)
    assert result.calls
    parsed_args = json.loads(result.calls[0].parameters)
    assert parsed_args["city"] == "Rome"
    assert result.normal_text == "Here is the tool"
