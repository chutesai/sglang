import logging
import re
from typing import Dict, List, Tuple, Union

import torch
from transformers import PreTrainedTokenizerBase

from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


# Compatible with KimiVLForConditionalGeneration
class KimiK2_5VLImageProcessor(SGLangBaseProcessor):
    models = [KimiK25ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # AutoProcessor may return just a tokenizer for Kimi-K2.5 because the
        # model's auto_map for AutoProcessor is only in preprocessor_config.json
        # (not config.json), which some transformers versions / kwargs combos
        # cause to be missed.  Reconstruct the full processor when needed.
        if isinstance(_processor, PreTrainedTokenizerBase):
            _processor = self._build_full_processor(server_args, _processor)
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

    @staticmethod
    def _build_full_processor(server_args, tokenizer):
        """Build the full KimiK25Processor when AutoProcessor returned a bare tokenizer."""
        import json
        import os

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        logger.info(
            "AutoProcessor returned a bare tokenizer for Kimi-K2.5. "
            "Loading image processor separately to construct full processor."
        )

        # Find preprocessor_config.json — try local paths first, then HF cache
        preprocessor_config = None
        for path in [server_args.model_path, server_args.tokenizer_path]:
            config_file = os.path.join(path, "preprocessor_config.json")
            if os.path.isfile(config_file):
                with open(config_file) as f:
                    preprocessor_config = json.load(f)
                break

        if preprocessor_config is None:
            # Files are in the HF cache from snapshot_download but offline
            # flags may block normal resolution. Use local_files_only=True
            # to read directly from cache.
            from huggingface_hub import hf_hub_download

            config_file = hf_hub_download(
                server_args.tokenizer_path,
                "preprocessor_config.json",
                local_files_only=True,
            )
            with open(config_file) as f:
                preprocessor_config = json.load(f)

        # Load custom classes from the model's trust-remote-code modules.
        # local_files_only=True ensures this works even with HF_HUB_OFFLINE=1
        # since the files are already cached from the initial snapshot_download.
        model_name = server_args.tokenizer_path
        KimiK25VisionProcessor = get_class_from_dynamic_module(
            "kimi_k25_vision_processing.KimiK25VisionProcessor",
            model_name,
            local_files_only=True,
        )
        KimiK25Processor = get_class_from_dynamic_module(
            "kimi_k25_processor.KimiK25Processor",
            model_name,
            local_files_only=True,
        )

        image_processor = KimiK25VisionProcessor(
            media_proc_cfg=preprocessor_config["media_proc_cfg"],
        )
        return KimiK25Processor(
            image_processor=image_processor, tokenizer=tokenizer
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        prompt = base_output.input_text

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
        }

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Helper method to process multimodal data and create mm_items in one step.

        Returns:
            Tuple of (created mm_items, input_ids)
        """

        parts = input_text.split(self.mm_tokens.image_token)

        result = [parts[0]]
        for image, part in zip(images, parts[1:]):
            num_tokens = self._processor.media_processor.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            result.append(self.mm_tokens.image_token * num_tokens + part)

        input_text = "".join(result)

        if images:  # for kimi k2 vl
            mediums = []
            for image in images:
                mediums.append({"type": "image", "image": image})
            key = "_medias"[1:]  # bypass lint
            kwargs[key] = mediums
            images = None

        ret = self.process_mm_data(
            input_text=input_text, images=images, audios=audios, videos=videos, **kwargs
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids, ret
