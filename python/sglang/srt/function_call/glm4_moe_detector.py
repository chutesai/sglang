import ast
import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult, _GetInfoFunc

logger = logging.getLogger(__name__)


def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    properties = (tool.function.parameters or {}).get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    if arg_key not in properties:
        return None
    return properties[arg_key].get("type", None)


def parse_arguments(json_value):
    try:
        parsed_value = json.loads(json_value)
        return parsed_value, True
    except:
        # If that fails, try wrapping it to unescape JSON characters
        try:
            # Wrap the value as a JSON string field
            wrapped = json.loads('{"tmp": "' + json_value + '"}')
            # parse the unescaped value
            parsed_value = json.loads(wrapped["tmp"])
            return parsed_value, True
        except:
            # Final fallback to ast.literal_eval
            try:
                parsed_value = ast.literal_eval(json_value)
                return parsed_value, True
            except:
                return json_value, False


class Glm4MoeDetector(BaseFormatDetector):
    """
    Detector for GLM-4.5 and GLM-4.6 models.
    Assumes function call format:
      <tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>北京</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n</tool_call>\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>上海</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n</tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.func_call_regex = r"<tool_call>.*?</tool_call>"
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(?:\\n|\n)(.*)</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a glm-4.5 / glm-4.6 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = self.func_detail_regex.search(match_result)
                if func_detail is None:
                    continue
                func_name = func_detail.group(1) if func_detail.group(1) else ""
                func_args = func_detail.group(2) if func_detail.group(2) else ""
                pairs = self.func_arg_regex.findall(func_args)
                arguments = {}
                for arg_key, arg_value in pairs:
                    arg_key = arg_key.strip()
                    arg_value = arg_value.strip()
                    arg_type = get_argument_type(func_name, arg_key, tools)
                    if arg_type != "string":
                        arg_value, is_good_json = parse_arguments(arg_value)
                    arguments[arg_key] = arg_value
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": arguments}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for GLM-4.5 and GLM-4.6 format.
        Uses a state machine to convert XML to JSON incrementally for true character-by-character streaming.
        Outputs JSON increments immediately as XML data arrives.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call
        has_tool_call = self.bot_token in current_text

        if not has_tool_call:
            # Check if buffer could be the start of a tool call
            # Keep buffer if it could be a partial match of bot_token
            is_potential_start = any(
                self.bot_token.startswith(current_text[-i:])
                for i in range(1, min(len(current_text), len(self.bot_token)) + 1)
            )

            if not is_potential_start:
                # Not a potential tool call, return as normal text
                # Must return the entire buffer (current_text), not just new_text,
                # because buffer may contain previously accumulated characters like '<'
                # that turned out not to be part of a tool call
                output_text = current_text
                self._buffer = ""
                if self.eot_token in output_text:
                    output_text = output_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=output_text)
            else:
                # Could be start of tool call, keep buffering
                return StreamingParseResult(normal_text="", calls=[])

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            # Try to match a partial or complete tool call
            partial_match = re.search(
                pattern=r"<tool_call>(.*?)(?:\\n|\n)(.*?)(</tool_call>|$)",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name_raw = partial_match.group(1)
                func_args_raw = partial_match.group(2)
                is_tool_end = partial_match.group(3)

                # Only proceed if we have a non-empty function name
                if func_name_raw is None or not func_name_raw.strip():
                    # If we only have the start token without a function name,
                    # continue buffering until we get more content
                    return StreamingParseResult(normal_text="", calls=[])

                func_name = func_name_raw.strip()
                func_args_raw = func_args_raw.strip() if func_args_raw else ""

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]
                    self._streamed_raw_length = 0
                    self.current_tool_name_sent = False
                    self._reset_streaming_state()

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                # Send tool name first if not sent yet
                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self._streamed_raw_length = 0
                    self._reset_streaming_state()
                    # Store the tool call info
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    # Process XML to JSON streaming
                    current_raw_length = len(func_args_raw)

                    if current_raw_length > self._streamed_raw_length:
                        # Get the new raw XML content
                        raw_increment = func_args_raw[self._streamed_raw_length :]

                        # Convert XML increment to JSON increment using state machine
                        json_increment = self._process_xml_to_json_streaming(
                            raw_increment, func_name, tools
                        )

                        # CRITICAL: Update streamed length BEFORE checking json_increment
                        # Even if json_increment is empty, the input has been consumed by the state machine
                        self._streamed_raw_length = current_raw_length

                        if json_increment:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=json_increment,
                                )
                            )
                            self._last_arguments += json_increment
                            self.streamed_args_for_tool[
                                self.current_tool_id
                            ] += json_increment

                    if is_tool_end == self.eot_token:
                        if self._is_first_param:
                            empty_object = "{}"
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=empty_object,
                                )
                            )
                            self._last_arguments += empty_object
                        elif not self._last_arguments.endswith("}"):
                            closing_brace = "}"
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=closing_brace,
                                )
                            )
                            self._last_arguments += closing_brace
                            self.streamed_args_for_tool[
                                self.current_tool_id
                            ] += closing_brace

                        try:
                            pairs = self.func_arg_regex.findall(func_args_raw)
                            if pairs:
                                arguments = self._parse_argument_pairs(
                                    pairs, func_name, tools
                                )
                                self.prev_tool_call_arr[self.current_tool_id][
                                    "arguments"
                                ] = arguments
                        except Exception as e:
                            logger.debug(
                                f"Failed to parse arguments: {e}", exc_info=True
                            )

                        # Remove the completed tool call from buffer
                        self._buffer = current_text[partial_match.end(3) :]

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        self._streamed_raw_length = 0
                        self._reset_streaming_state()
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}", exc_info=True)
            return StreamingParseResult(normal_text=current_text)

    def _parse_argument_pairs(
        self, pairs: List[Tuple[str, str]], func_name: str, tools: List[Tool]
    ) -> Dict[str, Any]:
        """Parse argument key-value pairs with type coercion.

        Args:
            pairs: List of (key, value) tuples from regex matching
            func_name: Name of the function
            tools: List of available tools

        Returns:
            Dictionary of parsed arguments
        """
        arguments = {}
        for arg_key, arg_value in pairs:
            arg_key = arg_key.strip()
            arg_value = arg_value.strip()
            arg_type = get_argument_type(func_name, arg_key, tools)
            parsed_value, is_good_json = parse_arguments(arg_value, arg_type)

            if arg_type == "string":
                # Only convert to string if explicitly defined as string type
                if isinstance(parsed_value, str):
                    arguments[arg_key] = parsed_value
                elif isinstance(parsed_value, (dict, list)):
                    # If parsed as dict/list but schema says string, convert to JSON string
                    arguments[arg_key] = json.dumps(parsed_value, ensure_ascii=False)
                else:
                    arguments[arg_key] = str(parsed_value)
            elif arg_type is None:
                # If type is not defined, keep the parsed value as-is
                arguments[arg_key] = parsed_value if is_good_json else arg_value
            else:
                # For other types (number, object, array, etc.), use parsed value
                arguments[arg_key] = parsed_value if is_good_json else arg_value

        return arguments

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
