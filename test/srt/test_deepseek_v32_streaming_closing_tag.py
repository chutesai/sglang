"""
Test for DeepSeek V3.2 streaming with closing tag arriving after tool call parsing.

This tests the scenario where:
1. Regular content is streamed
2. The tool call with complete invoke block arrives
3. The fallback logic parses it without the closing tag
4. The closing </｜DSML｜function_calls> tag arrives in a subsequent chunk

The parser should consume the closing tag and not emit it as content.
"""

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


@pytest.fixture
def tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_capital_info",
                description="Get information about a capital city",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "population": {"type": "integer"},
                    },
                    "required": ["name", "population"],
                },
            ),
        )
    ]


def test_streaming_closing_tag_after_invoke(tools):
    """Test that closing tag is consumed when it arrives after tool call parsing."""
    detector = DeepSeekV32Detector()

    # Chunk 1: Regular content
    result1 = detector.parse_streaming_increment(
        "I'll use the tool to get information.\n\n", tools
    )
    assert result1.normal_text == "I'll use the tool to get information.\n\n"
    assert len(result1.calls) == 0

    # Chunk 2: Opening tag and complete invoke block (no closing tag yet)
    tool_call_chunk = (
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="get_capital_info">\n'
        '<｜DSML｜parameter name="name" string="true">Paris</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="population" string="false">2100000</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>'
    )
    result2 = detector.parse_streaming_increment(tool_call_chunk, tools)

    # Should parse the tool call using fallback logic
    assert len(result2.calls) == 1
    assert result2.calls[0].name == "get_capital_info"
    assert result2.calls[0].tool_index == 0  # Should be 0, not -1
    assert result2.normal_text == ""

    # Chunk 3: Closing tag arrives
    result3 = detector.parse_streaming_increment("</｜DSML｜function_calls>", tools)

    # The closing tag should be consumed, not returned as normal text
    assert result3.normal_text == ""
    assert len(result3.calls) == 0


def test_streaming_closing_tag_split_across_chunks(tools):
    """Test that partial closing tags are handled correctly."""
    detector = DeepSeekV32Detector()

    # Chunk 1: Complete invoke block
    tool_call_chunk = (
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="get_capital_info">\n'
        '<｜DSML｜parameter name="name" string="true">Paris</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="population" string="false">2100000</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>'
    )
    result1 = detector.parse_streaming_increment(tool_call_chunk, tools)
    assert len(result1.calls) == 1
    assert result1.calls[0].tool_index == 0

    # Chunk 2: Partial closing tag (split across chunks)
    result2 = detector.parse_streaming_increment("</｜DSML｜", tools)
    assert result2.normal_text == ""
    assert len(result2.calls) == 0

    # Chunk 3: Rest of closing tag
    result3 = detector.parse_streaming_increment("function_calls>", tools)
    assert result3.normal_text == ""
    assert len(result3.calls) == 0


def test_streaming_normal_flow_with_closing_tag(tools):
    """Test normal streaming flow when closing tag arrives before parsing."""
    detector = DeepSeekV32Detector()

    # Chunk 1: Complete tool call with closing tag
    complete_call = (
        "I'll use the tool.\n\n"
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="get_capital_info">\n'
        '<｜DSML｜parameter name="name" string="true">Paris</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="population" string="false">2100000</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜function_calls>'
    )
    result = detector.parse_streaming_increment(complete_call, tools)

    # Should parse normally with closing tag present
    assert result.normal_text == "I'll use the tool."
    assert len(result.calls) == 1
    assert result.calls[0].name == "get_capital_info"
    assert result.calls[0].tool_index == 0


def test_multiple_tool_calls_sequential_indices(tools):
    """Test that multiple tool calls get sequential indices."""
    detector = DeepSeekV32Detector()

    # Add a second tool
    tools.append(
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        )
    )

    complete_call = (
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="get_capital_info">\n'
        '<｜DSML｜parameter name="name" string="true">Paris</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="population" string="false">2100000</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="city" string="true">London</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜function_calls>'
    )
    result = detector.parse_streaming_increment(complete_call, tools)

    # Should have two tool calls with sequential indices
    assert len(result.calls) == 2
    assert result.calls[0].name == "get_capital_info"
    assert result.calls[0].tool_index == 0
    assert result.calls[1].name == "get_weather"
    assert result.calls[1].tool_index == 1


def test_plain_format_closing_tag(tools):
    """Test plain format (without DSML tokens) also handles closing tag correctly."""
    detector = DeepSeekV32Detector()

    # Chunk 1: Plain format invoke block
    tool_call_chunk = (
        '<function_calls>\n'
        '<invoke name="get_capital_info">\n'
        '<parameter name="name" string="true">Paris</parameter>\n'
        '<parameter name="population" string="false">2100000</parameter>\n'
        '</invoke>'
    )
    result1 = detector.parse_streaming_increment(tool_call_chunk, tools)
    assert len(result1.calls) == 1
    assert result1.calls[0].tool_index == 0

    # Chunk 2: Closing tag
    result2 = detector.parse_streaming_increment("</function_calls>", tools)
    assert result2.normal_text == ""
    assert len(result2.calls) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
