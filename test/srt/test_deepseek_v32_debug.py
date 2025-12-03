"""Debug test to simulate exact streaming scenario from user's output."""

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


def test_exact_user_scenario():
    """Simulate the exact scenario from user's output."""
    tools = [
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

    detector = DeepSeekV32Detector()

    # Simulate streaming chunks similar to user's output
    chunks = [
        "I'll use the tool to get information about Paris, the capital of France, with the population you mentioned.\n\n",
        '<｜DSML｜function_calls>\n<｜DSML｜invoke name="get_capital_info">\n<｜DSML｜parameter name="name" string="true">Paris</｜DSML｜parameter>\n<｜DSML｜parameter name="population" string="false">2100000</｜DSML｜parameter>\n</｜DSML｜invoke>',
        "</｜DSML｜",
        "function",
        "_c",
        "alls",
        ">",
    ]

    results = []
    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing chunk {i+1}: {repr(chunk)}")
        print(f"Buffer before: {repr(detector._buffer)}")
        print(f"_tool_calls_emitted before: {detector._tool_calls_emitted}")

        result = detector.parse_streaming_increment(chunk, tools)

        print(f"Buffer after: {repr(detector._buffer)}")
        print(f"_tool_calls_emitted after: {detector._tool_calls_emitted}")
        print(f"Result normal_text: {repr(result.normal_text)}")
        print(f"Result calls: {len(result.calls)} calls")
        if result.calls:
            for call in result.calls:
                print(f"  - {call.name} (index={call.tool_index})")

        results.append(result)

    # Verify results
    # Chunk 1: Normal text
    assert results[0].normal_text == chunks[0]
    assert len(results[0].calls) == 0

    # Chunk 2: Tool call
    assert len(results[1].calls) == 1
    assert results[1].calls[0].tool_index == 0
    assert results[1].calls[0].name == "get_capital_info"

    # Chunks 3-7: Closing tag parts should be consumed, not returned as normal text
    for i in range(2, 7):
        print(f"\nAsserting chunk {i+1}: normal_text should be empty, got {repr(results[i].normal_text)}")
        assert results[i].normal_text == "", f"Chunk {i+1} returned normal_text={repr(results[i].normal_text)}, should be empty!"
        assert len(results[i].calls) == 0


if __name__ == "__main__":
    test_exact_user_scenario()
    print("\n✓ All assertions passed!")
