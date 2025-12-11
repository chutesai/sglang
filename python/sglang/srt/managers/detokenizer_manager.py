# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DetokenizerManager is a process that detokenizes the token ids."""

import dataclasses
import logging
import os
import re
import signal
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import psutil
import setproctitle
import zmq

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchMultimodalDecodeReq,
    BatchStrOutput,
    BatchTokenIDOutput,
    FreezeGCReq,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiHttpWorkerDetokenizerMixin
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    freeze_gc,
    get_zmq_socket,
    kill_itself_when_parent_died,
)
from sglang.srt.utils.hf_transformers_utils import get_config, get_tokenizer
from sglang.utils import (
    TypeBasedDispatcher,
    find_printable_text,
    get_exception_traceback,
)

logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0


# GLM vision model architectures that may emit stray bounding box tokens
GLM_VISION_ARCHITECTURES = [
    "Glm4vForConditionalGeneration",
    "Glm4vMoeForConditionalGeneration",
]


class GlmBoundingBoxFilter:
    """
    Filter stray bounding box tokens from GLM vision model outputs.

    GLM-4.5V/4.6V models emit <|begin_of_box|> and <|end_of_box|> tokens for
    bounding box outputs. However, they sometimes emit these tokens spuriously
    even when there's no valid bounding box (e.g., in response to "Hi, how are you?").

    This filter buffers tokens after seeing <|begin_of_box|> and only keeps them
    if a valid bounding box pattern is detected (4 numbers enclosed in brackets).
    Invalid bounding box sequences are filtered out.

    Valid bounding box format: <|begin_of_box|>[x1, y1, x2, y2]<|end_of_box|>
    where x1, y1, x2, y2 are numbers.
    """

    # Regex pattern to match valid bounding box content
    BBOX_PATTERN = re.compile(
        r"^[\[\(<\{\s]*"  # Opening brackets
        r"\s*[\d\.]+\s*,\s*[\d+\.]+\s*,\s*[\d+\.]+\s*,\s*[\d+\.]+\s*"  # 4 numbers
        r"[\]\)>\}]*$"  # Closing brackets
    )

    def __init__(self, tokenizer):
        """
        Initialize the filter.

        Args:
            tokenizer: The HuggingFace tokenizer to use for encoding special tokens.
        """
        self.tokenizer = tokenizer
        # Get token IDs for the special bounding box tokens
        self.begin_box_token_id = self._get_token_id("<|begin_of_box|>")
        self.end_box_token_id = self._get_token_id("<|end_of_box|>")

        # Per-request state tracking
        # Maps rid -> (buffered_ids, buffered_logprobs_val, buffered_logprobs_idx)
        self.buffer_state: Dict[str, dict] = {}

    def _get_token_id(self, token_str: str) -> Optional[int]:
        """Get the token ID for a special token string."""
        try:
            # Try to encode the token
            encoded = self.tokenizer.encode(token_str, add_special_tokens=False)
            if len(encoded) == 1:
                return encoded[0]
            # Try to get from vocab directly
            vocab = self.tokenizer.get_vocab()
            if token_str in vocab:
                return vocab[token_str]
        except Exception:
            pass
        return None

    def _is_valid_bbox(self, content: str) -> bool:
        """
        Check if the content between box tokens is a valid bounding box.

        Args:
            content: The decoded text content between <|begin_of_box|> and <|end_of_box|>

        Returns:
            True if it's a valid bounding box pattern, False otherwise.
        """
        content = content.strip()
        if not content:
            return False
        return bool(self.BBOX_PATTERN.match(content))

    def filter_tokens(
        self,
        rid: str,
        token_ids: List[int],
        logprobs_val: Optional[List[float]] = None,
        logprobs_idx: Optional[List[int]] = None,
        top_logprobs_val: Optional[List] = None,
        top_logprobs_idx: Optional[List] = None,
        is_finished: bool = False,
    ) -> Tuple[
        List[int],
        Optional[List[float]],
        Optional[List[int]],
        Optional[List],
        Optional[List],
    ]:
        """
        Filter stray bounding box tokens from the output.

        This method buffers tokens when it sees <|begin_of_box|> and waits to see
        if the content forms a valid bounding box. If valid, it passes through.
        If invalid (or request finishes without <|end_of_box|>), the tokens are filtered.

        Args:
            rid: Request ID for tracking state across calls.
            token_ids: List of token IDs to filter.
            logprobs_val: List of logprob values (same length as token_ids), or None.
            logprobs_idx: List of logprob token indices (same length as token_ids), or None.
            top_logprobs_val: List of top logprobs values, or None.
            top_logprobs_idx: List of top logprobs indices, or None.
            is_finished: Whether this is the final output for the request.

        Returns:
            Tuple of (filtered_token_ids, filtered_logprobs_val, filtered_logprobs_idx,
                     filtered_top_logprobs_val, filtered_top_logprobs_idx).
        """
        if self.begin_box_token_id is None or self.end_box_token_id is None:
            # Can't filter if we don't have the token IDs
            return (
                token_ids,
                logprobs_val,
                logprobs_idx,
                top_logprobs_val,
                top_logprobs_idx,
            )

        # Initialize or get buffer state for this request
        if rid not in self.buffer_state:
            self.buffer_state[rid] = {
                "buffered_ids": [],
                "buffered_logprobs_val": [],
                "buffered_logprobs_idx": [],
                "buffered_top_logprobs_val": [],
                "buffered_top_logprobs_idx": [],
                "in_bbox": False,
            }

        state = self.buffer_state[rid]
        result_ids = []
        result_logprobs_val = [] if logprobs_val is not None else None
        result_logprobs_idx = [] if logprobs_idx is not None else None
        result_top_logprobs_val = [] if top_logprobs_val is not None else None
        result_top_logprobs_idx = [] if top_logprobs_idx is not None else None

        for i, token_id in enumerate(token_ids):
            lp_val = logprobs_val[i] if logprobs_val is not None else None
            lp_idx = logprobs_idx[i] if logprobs_idx is not None else None
            top_lp_val = top_logprobs_val[i] if top_logprobs_val is not None else None
            top_lp_idx = top_logprobs_idx[i] if top_logprobs_idx is not None else None

            if token_id == self.begin_box_token_id:
                # Start buffering
                state["in_bbox"] = True
                state["buffered_ids"] = [token_id]
                state["buffered_logprobs_val"] = [lp_val] if lp_val is not None else []
                state["buffered_logprobs_idx"] = [lp_idx] if lp_idx is not None else []
                state["buffered_top_logprobs_val"] = (
                    [top_lp_val] if top_lp_val is not None else []
                )
                state["buffered_top_logprobs_idx"] = (
                    [top_lp_idx] if top_lp_idx is not None else []
                )

            elif state["in_bbox"]:
                # We're buffering content inside a bbox
                state["buffered_ids"].append(token_id)
                if lp_val is not None:
                    state["buffered_logprobs_val"].append(lp_val)
                if lp_idx is not None:
                    state["buffered_logprobs_idx"].append(lp_idx)
                if top_lp_val is not None:
                    state["buffered_top_logprobs_val"].append(top_lp_val)
                if top_lp_idx is not None:
                    state["buffered_top_logprobs_idx"].append(top_lp_idx)

                if token_id == self.end_box_token_id:
                    # End of bbox - check if valid
                    state["in_bbox"] = False
                    # Decode the content between begin and end (excluding them)
                    content_ids = state["buffered_ids"][
                        1:-1
                    ]  # Exclude begin and end tokens
                    if content_ids:
                        try:
                            content_text = self.tokenizer.decode(
                                content_ids, skip_special_tokens=False
                            )
                            is_valid = self._is_valid_bbox(content_text)
                        except Exception:
                            is_valid = False
                    else:
                        is_valid = False

                    if is_valid:
                        # Valid bbox - emit all buffered tokens
                        result_ids.extend(state["buffered_ids"])
                        if result_logprobs_val is not None:
                            result_logprobs_val.extend(state["buffered_logprobs_val"])
                        if result_logprobs_idx is not None:
                            result_logprobs_idx.extend(state["buffered_logprobs_idx"])
                        if result_top_logprobs_val is not None:
                            result_top_logprobs_val.extend(
                                state["buffered_top_logprobs_val"]
                            )
                        if result_top_logprobs_idx is not None:
                            result_top_logprobs_idx.extend(
                                state["buffered_top_logprobs_idx"]
                            )
                    # else: Invalid bbox - discard all buffered tokens (don't emit)

                    # Clear buffer
                    state["buffered_ids"] = []
                    state["buffered_logprobs_val"] = []
                    state["buffered_logprobs_idx"] = []
                    state["buffered_top_logprobs_val"] = []
                    state["buffered_top_logprobs_idx"] = []

            else:
                # Normal token outside of bbox
                result_ids.append(token_id)
                if result_logprobs_val is not None:
                    result_logprobs_val.append(lp_val)
                if result_logprobs_idx is not None:
                    result_logprobs_idx.append(lp_idx)
                if result_top_logprobs_val is not None:
                    result_top_logprobs_val.append(top_lp_val)
                if result_top_logprobs_idx is not None:
                    result_top_logprobs_idx.append(top_lp_idx)

        # If request is finished and we still have buffered content, discard it
        # (it's an incomplete/invalid bbox)
        if is_finished:
            if rid in self.buffer_state:
                del self.buffer_state[rid]

        return (
            result_ids,
            result_logprobs_val,
            result_logprobs_idx,
            result_top_logprobs_val,
            result_top_logprobs_idx,
        )

    def cleanup_request(self, rid: str):
        """Clean up state for a finished request."""
        if rid in self.buffer_state:
            del self.buffer_state[rid]


def is_glm_vision_model(model_path: str, trust_remote_code: bool = False) -> bool:
    """
    Check if the model is a GLM vision model that may emit stray bounding box tokens.

    Args:
        model_path: Path to the model (local or HuggingFace hub).
        trust_remote_code: Whether to trust remote code when loading config.

    Returns:
        True if it's a GLM vision model, False otherwise.
    """
    try:
        config = get_config(model_path, trust_remote_code=trust_remote_code)
        architectures = getattr(config, "architectures", []) or []
        return any(arch in GLM_VISION_ARCHITECTURES for arch in architectures)
    except Exception:
        model_path_lower = model_path.lower()
        return re.search(r"^zai-org/glm[0-9\.\-]+v", model_path_lower)


class DetokenizerManager(MultiHttpWorkerDetokenizerMixin):
    """DetokenizerManager is a process that detokenizes the token ids."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Init inter-process communication
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )

        # Init tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )

        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)
        self.is_dummy = False
        self.is_tool_call_parser_gpt_oss = server_args.tool_call_parser == "gpt-oss"
        self.disable_tokenizer_batch_decode = server_args.disable_tokenizer_batch_decode

        # Init GLM bounding box filter for GLM vision models
        self.glm_bbox_filter = None
        if self.tokenizer is not None:
            # Check if this is a GLM vision model
            enable_filter = getattr(server_args, "enable_glm_bbox_filter", None)
            if enable_filter is None:
                # Auto-detect: enable for GLM vision models
                enable_filter = is_glm_vision_model(
                    server_args.model_path,
                    trust_remote_code=server_args.trust_remote_code,
                )
            if enable_filter:
                self.glm_bbox_filter = GlmBoundingBoxFilter(self.tokenizer)
                logger.info(
                    "GLM bounding box filter enabled for filtering stray "
                    "<|begin_of_box|>/<|end_of_box|> tokens"
                )

        # Init dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOutput, self.handle_batch_embedding_out),
                (BatchTokenIDOutput, self.handle_batch_token_id_out),
                (BatchMultimodalDecodeReq, self.handle_multimodal_decode_req),
                (FreezeGCReq, self.handle_freeze_gc_req),
            ]
        )

    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            output = self._request_dispatcher(recv_obj)
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)

    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if no_stop_trim or not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            # 200012 <|call|> is the tool call token and one of eos tokens for gpt-oss model
            if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
                return output
            assert len(output) > 0
            # NOTE: We can always assume the last token is the matched stop token
            return output[:-1]
        return output

    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOutput):
        # If it is embedding model, no detokenization is needed.
        return recv_obj

    def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
        bs = len(recv_obj.rids)

        # Initialize decode status
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=recv_obj.decode_ids[i],
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s
            else:
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])

            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

        # Decode token ids to strings
        # TODO(lmzheng): handle skip_special_tokens/spaces_between_special_tokens per request
        if not self.disable_tokenizer_batch_decode:
            if not self.is_dummy:
                # Run normal batch decode
                surr_texts = self.tokenizer.batch_decode(
                    surr_ids,
                    skip_special_tokens=recv_obj.skip_special_tokens[0],
                    spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[
                        0
                    ],
                )
                read_texts = self.tokenizer.batch_decode(
                    read_ids,
                    skip_special_tokens=recv_obj.skip_special_tokens[0],
                    spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[
                        0
                    ],
                )
            else:
                # If it is dummy weights, just return dummy strings to prevent potential detokenization edge cases
                surr_texts = ["dog" for _ in surr_ids]
                read_texts = ["cat" for _ in read_ids]
        else:
            # Do not use batch decode to prevent some detokenization edge cases (e.g., gpt-oss).
            surr_texts = [
                self.tokenizer.decode(
                    surr, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for surr, skip, space in zip(
                    surr_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]
            read_texts = [
                self.tokenizer.decode(
                    read, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for read, skip, space in zip(
                    read_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]

        # Incremental decoding
        output_strs = []
        for i in range(bs):
            try:
                s = self.decode_status[recv_obj.rids[i]]
            except KeyError:
                raise RuntimeError(
                    f"Decode status not found for request {recv_obj.rids[i]}. "
                    "It may be due to the request being evicted from the decode status due to memory pressure. "
                    "Please increase the maximum number of requests by setting "
                    "the SGLANG_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                    f"The current value is {DETOKENIZER_MAX_STATES}. "
                    "For more details, see: https://github.com/sgl-project/sglang/issues/2812"
                )
            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming chunk: update the decode status
                if len(new_text) > 0 and not new_text.endswith("�"):
                    s.decoded_text = s.decoded_text + new_text
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    new_text = ""
                else:
                    new_text = find_printable_text(new_text)
            else:
                del self.decode_status[recv_obj.rids[i]]

            output_str = self.trim_matched_stop(
                s.decoded_text + new_text,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
            # Incrementally send text.
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            output_strs.append(incremental_output)

        return output_strs

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # Apply GLM bounding box filter if enabled - MUST happen before decoding
        if self.glm_bbox_filter is not None and recv_obj.decode_ids is not None:
            filtered_decode_ids = []
            filtered_output_ids = []
            filtered_logprobs_val = [] if recv_obj.output_token_logprobs_val else None
            filtered_logprobs_idx = [] if recv_obj.output_token_logprobs_idx else None
            filtered_top_logprobs_val = [] if recv_obj.output_top_logprobs_val else None
            filtered_top_logprobs_idx = [] if recv_obj.output_top_logprobs_idx else None

            for i, rid in enumerate(recv_obj.rids):
                is_finished = recv_obj.finished_reasons[i] is not None
                token_ids = recv_obj.decode_ids[i]

                # Get logprobs for this request if available
                lp_val = (
                    recv_obj.output_token_logprobs_val[i]
                    if recv_obj.output_token_logprobs_val
                    else None
                )
                lp_idx = (
                    recv_obj.output_token_logprobs_idx[i]
                    if recv_obj.output_token_logprobs_idx
                    else None
                )
                top_lp_val = (
                    recv_obj.output_top_logprobs_val[i]
                    if recv_obj.output_top_logprobs_val
                    else None
                )
                top_lp_idx = (
                    recv_obj.output_top_logprobs_idx[i]
                    if recv_obj.output_top_logprobs_idx
                    else None
                )

                # Apply filter
                (
                    filtered_ids,
                    filtered_lp_val,
                    filtered_lp_idx,
                    filtered_top_lp_val,
                    filtered_top_lp_idx,
                ) = self.glm_bbox_filter.filter_tokens(
                    rid=rid,
                    token_ids=token_ids,
                    logprobs_val=lp_val,
                    logprobs_idx=lp_idx,
                    top_logprobs_val=top_lp_val,
                    top_logprobs_idx=top_lp_idx,
                    is_finished=is_finished,
                )

                filtered_decode_ids.append(filtered_ids)
                # Also filter output_ids if present
                if recv_obj.output_ids is not None:
                    output_ids_for_req = (
                        recv_obj.output_ids[i]
                        if isinstance(recv_obj.output_ids[i], list)
                        else [recv_obj.output_ids[i]]
                    )
                    # Filter output_ids to match filtered decode_ids
                    # For now, just use the filtered_ids since they should be the same
                    filtered_output_ids.append(
                        filtered_ids[-len(output_ids_for_req) :]
                        if len(filtered_ids) >= len(output_ids_for_req)
                        else filtered_ids
                    )

                if filtered_logprobs_val is not None:
                    filtered_logprobs_val.append(
                        filtered_lp_val if filtered_lp_val is not None else []
                    )
                if filtered_logprobs_idx is not None:
                    filtered_logprobs_idx.append(
                        filtered_lp_idx if filtered_lp_idx is not None else []
                    )
                if filtered_top_logprobs_val is not None:
                    filtered_top_logprobs_val.append(
                        filtered_top_lp_val if filtered_top_lp_val is not None else []
                    )
                if filtered_top_logprobs_idx is not None:
                    filtered_top_logprobs_idx.append(
                        filtered_top_lp_idx if filtered_top_lp_idx is not None else []
                    )

            # Replace the decode_ids in recv_obj with filtered ones
            recv_obj.decode_ids = filtered_decode_ids
            if recv_obj.output_ids is not None:
                recv_obj.output_ids = filtered_output_ids
            recv_obj.output_token_logprobs_val = filtered_logprobs_val
            recv_obj.output_token_logprobs_idx = filtered_logprobs_idx
            recv_obj.output_top_logprobs_val = filtered_top_logprobs_val
            recv_obj.output_top_logprobs_idx = filtered_top_logprobs_idx

        # Now decode with the filtered token IDs
        output_strs = self._decode_batch_token_id_output(recv_obj)

        output_ids = recv_obj.output_ids
        output_token_logprobs_val = recv_obj.output_token_logprobs_val
        output_token_logprobs_idx = recv_obj.output_token_logprobs_idx
        output_top_logprobs_val = recv_obj.output_top_logprobs_val
        output_top_logprobs_idx = recv_obj.output_top_logprobs_idx

        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            reasoning_tokens=recv_obj.reasoning_tokens,
            spec_verify_ct=recv_obj.spec_verify_ct,
            spec_accepted_tokens=recv_obj.spec_accepted_tokens,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=output_token_logprobs_val,
            output_token_logprobs_idx=output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=output_top_logprobs_val,
            output_top_logprobs_idx=output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_token_entropy_val=recv_obj.output_token_entropy_val,
            output_hidden_states=recv_obj.output_hidden_states,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=recv_obj.retraction_counts,
            token_steps=recv_obj.token_steps,
            queue_time=recv_obj.queue_time,
            forward_entry_time=recv_obj.forward_entry_time,
            prefill_launch_delay=recv_obj.prefill_launch_delay,
            prefill_launch_latency=recv_obj.prefill_launch_latency,
        )

    def handle_multimodal_decode_req(self, recv_obj: BatchMultimodalDecodeReq):
        raise NotImplementedError()

    def handle_freeze_gc_req(self, recv_req: FreezeGCReq):
        freeze_gc("Detokenizer Manager")
        return None


class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    detokenizer_manager_class=DetokenizerManager,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = detokenizer_manager_class(server_args, port_args)
        if server_args.tokenizer_worker_num > 1:
            manager.multi_http_worker_event_loop()
        else:
            manager.event_loop()
    except Exception:
        manager.maybe_clear_socket_mapping()
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
