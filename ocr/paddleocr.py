from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
from paddleocr import PaddleOCR, draw_ocr

COORDINATES = List[List]
TEXTWITHCONFIDENCE = Tuple[str, float]


class OCR(ABC):
    @abstractmethod
    def get_text_and_bounding_boxes_per_page(self, image_path_or_np_array: Union[str, np.array]) -> List[Union[
        COORDINATES, TEXTWITHCONFIDENCE]]:
        pass


class PaddleOCRWrapper(OCR):
    def __init__(self, use_gpu: bool = False):
        self.ocr = PaddleOCR(ocr_version="PP-OCRv4", show_log=True, use_angle_cls=True, use_gpu=use_gpu)
        self.text_only = []

    def get_text_and_bounding_boxes_per_page(self, image_path_or_np_array: Union[str, np.array]) ->List[Union[
        COORDINATES, TEXTWITHCONFIDENCE]]:
        if image_path_or_np_array:
            result = self.ocr.ocr(image_path_or_np_array, cls=True)
        else:
            raise ValueError("Image path or numpy array is required")
        result = result[0]
        # bounding_boxes = [line[0] for line in result]
        # self.text_only.extend [line[1][0] for line in result]
        # scores = [line[1][1] for line in result]
        return result
