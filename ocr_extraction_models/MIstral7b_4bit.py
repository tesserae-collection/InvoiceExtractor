from functools import lru_cache
from typing import List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig

from ocr.paddleocr import COORDINATES, TEXTWITHCONFIDENCE

MODELID = "mychen76/mistral7b_ocr_to_json_v1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FineTunedMistral7B:
    def __init__(self):
        self.bnb_config = ...
        # controls model memory allocation between devices for low GPU resources (0, cpu)

        self.device_map = ...
        self.model = ...
        self.tokenizer = ...
        self.load_model()

    @lru_cache(maxsize=5)
    def load_model(self):
        self.bnb_config = BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=True,
            load_in_4bit=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # controls model memory allocation between devices for low GPU resources (0, cpu)

        self.device_map = {
            "transformer.word_embeddings": DEVICE,
            "transformer.word_embeddings_layernorm": DEVICE,
            "lm_head": DEVICE,
            "transformer.h": DEVICE,
            "transformer.ln_f": DEVICE,
            "model.embed_tokens": DEVICE,
            "model.layers": DEVICE,
            "model.norm": DEVICE
        }
        self.model = AutoModelForCausalLM.from_pretrained(MODELID,
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.float16,
                                                          quantization_config=self.bnb_config,
                                                          device_map=self.device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(MODELID, trust_remote_code=True)

    def generate_output_json(self, ocr_results: List[Union[COORDINATES, TEXTWITHCONFIDENCE]]):
        if not ocr_results:
            return "No OCR results found"
        with torch.inference_mode():
            prompt = f"""### Instruction:
                         You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object. 
                         Don't make up value not in the Input. Output must be a well-formed JSON object.```json
                         
                         ### Input:
                         {ocr_results}
                         
                         ### Output:
                         """
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            outputs = self.model.generate(**input_ids,
                                          max_new_tokens=512)  ##use_cache=True, do_sample=True,temperature=0.1, top_p=0.95
            result_text = self.tokenizer.batch_decode(outputs)[0]

        return result_text
    # clear
