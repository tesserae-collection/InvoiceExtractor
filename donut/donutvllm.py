import re
from typing import Union
from functools import lru_cache

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

# model_id = "mychen76/invoice-and-receipts_donut_v1"
model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_prompt_dict = {"mychen76/invoice-and-receipts_donut_v1": "<s_receipt>",
                    "naver-clova-ix/donut-base-finetuned-cord-v2": "<s_cord-v2>"}


class DonutVLLM:
    def __init__(self):
        self.processor: DonutProcessor = ...
        self.model: VisionEncoderDecoderModel = ...
        self.load_model()

    @lru_cache(maxsize=5)
    def load_model(self, model_id: str = model_id):
        self.model_id = model_id
        self.processor = DonutProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(DEVICE)

    def get_text_from_image(self, image, task_prompt="<s_receipt>"):
        image_tensor = self.processor(image, return_tensors="pt").pixel_values
        print(f"image_tensor shape: {image_tensor.shape}")
        task_prompt = task_prompt_dict[self.model_id]
        decoder_input_ids = self.processor.tokenizer(task_prompt,
                                                     add_special_tokens=False,
                                                     return_tensors="pt").input_ids
        outputs = self.model.generate(pixel_values=image_tensor.to(DEVICE),
                                      decoder_input_ids=decoder_input_ids.to(DEVICE),
                                      max_length=self.model.decoder.config.max_position_embeddings,
                                      early_stopping=True,
                                      num_beams=1,
                                      do_sample=False,
                                      pad_token_id=self.processor.tokenizer.pad_token_id,
                                      eos_token_id=self.processor.tokenizer.eos_token_id,
                                      use_cache=True,
                                      bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                      return_dict_in_generate=True,
                                      output_scores=True,
                                      )
        return outputs

    def generate_output_xml(self, image_path: Union[str, Image.Image]):
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        outputs = self.get_text_from_image(image)
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token,
                                                                                    "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        return sequence

    def generate_output_json(self, image_path):
        xml = self.generate_output_xml(image_path)
        result = self.processor.token2json(xml)
        return result


if __name__ == '__main__':
    donut = DonutVLLM()
    print(donut.generate_output_json("/home/extravolatile/Desktop/datasets/receipt.jpeg"))
