"""
Test for DeepSeek V3.2 flushing buffered tool calls when generation finishes.

This tests the scenario where:
1. Model generates a tool call
2. Model stops before emitting the closing </｜DSML｜function_calls> tag
3. The tool call should still be emitted via flush_buffered_content()
"""

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


def test_flush_incomplete_tool_call():
    """Test that incomplete tool calls are flushed when generation finishes."""
    detector = DeepSeekV32Detector()
    tools = [
        Tool(
            type="function",
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": ["location", "unit"],
                },
            ),
        )
    ]

    # Chunk 1: Normal content
    result1 = detector.parse_streaming_increment(
        "I'll check the weather for you.\n\n", tools
    )
    assert result1.normal_text == "I'll check the weather for you.\n\n"
    assert len(result1.calls) == 0

    # Chunk 2: Tool call WITHOUT closing tag (model stopped mid-generation)
    incomplete_tool_call = (
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="get_current_weather">\n'
        '<｜DSML｜parameter name="location" string="true">Boston, MA</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="unit" string="true">fahrenheit</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>'
        # NOTE: Missing </｜DSML｜function_calls>
    )
    result2 = detector.parse_streaming_increment(incomplete_tool_call, tools)

    # Tool call should NOT be emitted yet (waiting for closing tag)
    assert len(result2.calls) == 0
    assert result2.normal_text == ""

    # Verify buffer has content
    assert len(detector._buffer) > 0

    # Simulate generation finishing - call flush_buffered_content
    flush_result = detector.flush_buffered_content(tools)

    # Tool call should NOW be emitted from the flush
    assert len(flush_result.calls) == 1
    assert flush_result.calls[0].name == "get_current_weather"
    assert flush_result.calls[0].tool_index == 0
    assert '"location": "Boston, MA"' in flush_result.calls[0].parameters
    assert '"unit": "fahrenheit"' in flush_result.calls[0].parameters

    # Buffer should be cleared
    assert detector._buffer == ""
    assert detector._tool_calls_emitted == False


def test_flush_with_no_buffer():
    """Test that flushing with empty buffer returns empty result."""
    detector = DeepSeekV32Detector()
    tools = [
        Tool(
            type="function",
            function=Function(
                name="test_func",
                description="test",
                parameters={"type": "object", "properties": {}},
            ),
        )
    ]

    # Flush with empty buffer
    flush_result = detector.flush_buffered_content(tools)

    assert len(flush_result.calls) == 0
    assert flush_result.normal_text == ""


def test_flush_after_complete_tool_call():
    """Test flushing after a complete tool call (with closing tag) - should be no-op."""
    detector = DeepSeekV32Detector()
    tools = [
        Tool(
            type="function",
            function=Function(
                name="test_func",
                description="test",
                parameters={"type": "object", "properties": {}},
            ),
        )
    ]

    # Complete tool call with closing tag
    complete_call = (
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="test_func">\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜function_calls>'
    )
    result = detector.parse_streaming_increment(complete_call, tools)

    # Tool call should be emitted immediately
    assert len(result.calls) == 1

    # Buffer should be empty
    assert detector._buffer == ""

    # Flushing should be a no-op
    flush_result = detector.flush_buffered_content(tools)
    assert len(flush_result.calls) == 0


def test_flush_handles_multiple_invokes():
    """Test flushing with multiple invoke blocks."""
    detector = DeepSeekV32Detector()
    tools = [
        Tool(
            type="function",
            function=Function(
                name="func1",
                description="test",
                parameters={"type": "object", "properties": {}},
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="func2",
                description="test",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    ]

    # Multiple tool calls WITHOUT closing tag
    incomplete_calls = (
        '<｜DSML｜function_calls>\n'
        '<｜DSML｜invoke name="func1">\n'
        '</｜DSML｜invoke>\n'
        '<｜DSML｜invoke name="func2">\n'
        '</｜DSML｜invoke>'
        # Missing </｜DSML｜function_calls>
    )
    result = detector.parse_streaming_increment(incomplete_calls, tools)

    # Nothing emitted yet
    assert len(result.calls) == 0

    # Flush should emit both tool calls
    flush_result = detector.flush_buffered_content(tools)
    assert len(flush_result.calls) == 2
    assert flush_result.calls[0].name == "func1"
    assert flush_result.calls[0].tool_index == 0
    assert flush_result.calls[1].name == "func2"
    assert flush_result.calls[1].tool_index == 1


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
