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
from typing import Dict, List, Union

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
    Filter stray bounding box tags from GLM vision model text outputs.

    GLM-4.5V/4.6V models emit <|begin_of_box|> and <|end_of_box|> tags for
    bounding box outputs. However, they sometimes emit these spuriously.

    This filter works at the STRING level (not token level) to:
    - Remove invalid bbox tags while keeping the content
    - Preserve valid bboxes with coordinates
    - Leave all token IDs and logprobs completely untouched
    """

    # Regex patterns
    BEGIN_TAG = "<|begin_of_box|>"
    END_TAG = "<|end_of_box|>"
    BBOX_PATTERN = re.compile(
        r"<\|begin_of_box\|>([\[\(<\{\s]*\s*[\d\.]+\s*,\s*[\d\.]+\s*,\s*[\d\.]+\s*,\s*[\d\.]+\s*[\]\)>\}]*)<\|end_of_box\|>"
    )

    def __init__(self):
        """Initialize the filter."""
        # Per-request buffer for streaming
        self.buffer_state: Dict[str, dict] = {}

    def filter_text(self, text: str) -> str:
        """
        Filter bbox tags from text. Works on complete (non-streaming) text.

        - Valid bboxes (with coordinates): Keep everything
        - Invalid bboxes (with text): Remove tags, keep content
        """
        if not text:
            return text

        # Find all complete bbox sequences
        def replace_bbox(match):
            content = match.group(1)
            # Check if content looks like coordinates (numbers)
            if re.match(
                r"^[\[\(<\{\s]*\s*[\d\.]+\s*,\s*[\d\.]+\s*,\s*[\d\.]+\s*,\s*[\d\.]+\s*[\]\)>\}]*$",
                content,
            ):
                # Valid bbox - keep everything
                return match.group(0)
            else:
                # Invalid bbox - return only content (strip tags)
                return content

        result = self.BBOX_PATTERN.sub(replace_bbox, text)

        # Also handle incomplete sequences (begin without end) - just remove the tag
        result = result.replace(self.BEGIN_TAG, "")
        result = result.replace(self.END_TAG, "")

        return result


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
                self.glm_bbox_filter = GlmBoundingBoxFilter()
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
        # Decode tokens to strings first (no filtering yet)
        output_strs = self._decode_batch_token_id_output(recv_obj)

        # Apply GLM bounding box filter to the DECODED STRINGS only
        # All token IDs and logprobs pass through completely untouched
        if self.glm_bbox_filter is not None:
            output_strs = [self.glm_bbox_filter.filter_text(s) for s in output_strs]

        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            reasoning_tokens=recv_obj.reasoning_tokens,
            spec_verify_ct=recv_obj.spec_verify_ct,
            spec_accepted_tokens=recv_obj.spec_accepted_tokens,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
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
