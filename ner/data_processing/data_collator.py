# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: YR269,VMF24
import logging
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any
from fsspec.utils import tokenize

import numpy as np
import torch

from ner.data_processing.constants import NER_ENCODING_MAP, PAD_NER_TAG
from ner.data_processing.tokenizer import Tokenizer


class DataCollator(object):
    def __init__(
        self,
        tokenizer: Tokenizer,
        padding: Union[str, bool] = "longest",
        max_length: Optional[int] = None,
        padding_side: str = "right",
        truncation_side: str = "right",
        pad_tag: str = PAD_NER_TAG,
        text_colname: str = "text",
        label_colname: str = "NER",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.pad_tag = pad_tag
        self.text_colname = text_colname
        self.label_colname = label_colname

    def _get_max_length(self, data_instances: List[Dict[str, Any]]) -> Optional[int]:
        if not ((self.padding == "longest" or self.padding) and self.max_length is None):
            logging.warning(
                f"both max_length={self.max_length} and padding={self.padding} provided; ignoring "
                f"padding={self.padding} and using max_length={self.max_length}"
            )
            self.padding = "max_length"

        if self.padding == "longest" or (isinstance(self.padding, bool) and self.padding):
            return max([len(data_instance[self.text_colname]) for data_instance in data_instances])
        elif self.padding == "max_length":
            return self.max_length
        elif isinstance(self.padding, bool) and not self.padding:
            return None
        raise ValueError(f"padding strategy {self.padding} is invalid")

    @staticmethod
    def _process_labels(labels: List) -> torch.Tensor:
        return torch.LongTensor([NER_ENCODING_MAP[label] for label in labels])

    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.data_processing.data_collator.html.
        """
        # TODO-1.2-1
        batch_max_length = self._get_max_length(data_instances)

        ids_lst = []
        mask_lst = []
        labels_lst = []

        for data in data_instances:
            ids = torch.empty((0, batch_max_length), dtype=torch.long)
            mask = torch.empty((0, batch_max_length), dtype=torch.long)
            labels = torch.empty((0, batch_max_length), dtype=torch.long)

            text = " ".join(data[self.text_colname])

            tokenized = self.tokenizer.tokenize(text, batch_max_length, self.padding_side, self.truncation_side)

            ids = tokenized['input_ids'].squeeze()
            mask = tokenized['padding_mask'].squeeze()

            # Check if we are not test
            if data.get(self.label_colname) != None:
                l = data[self.label_colname]
                length_label = len(data[self.label_colname])

                if length_label < batch_max_length:
                    # Pad
                    for i in range(batch_max_length - length_label):
                        if self.padding_side == "right":
                            l.append(PAD_NER_TAG)
                        else:
                            l.insert(0, PAD_NER_TAG)
                elif length_label > batch_max_length:
                    # Truncate
                    for i in range(length_label - batch_max_length):
                        if self.truncation_side == "right":
                            l.pop()
                        else:
                            l.pop(0)

                labels = self._process_labels(l)

            ids_lst.append(ids)
            mask_lst.append(mask)
            labels_lst.append(labels)

        ids = torch.stack(ids_lst)
        mask = torch.stack(mask_lst)
        labels = torch.stack(labels_lst)

        if data_instances[0].get(self.label_colname) != None:
          return {"input_ids":ids,"padding_mask":mask, "labels":labels}
        else:
          return {"input_ids":ids,"padding_mask":mask}
