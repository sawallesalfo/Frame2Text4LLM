import io
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor, 
    AutoModelForCausalLM
)



class VLMService:
    def __init__(
        self,
        model_name: str = 'microsoft/Florence-2-base-ft',
    ):    
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model_name = model_name

        if model_name == 'microsoft/Florence-2-base-ft' or model_name == 'microsoft/Florence-2-base':
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=self.torch_dtype, 
                trust_remote_code=True
            ).to(self.device).eval()

            self.task_prompt = "<OCR>"

        elif model_name == 'OpenGVLab/InternVL2-1B':
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                use_fast=False
            )

            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True
            ).eval().cuda()

            self.PROMPT_INTERNVL2 = """Extract and return only the text from the image without any additional commentary or introductory phrases. 
            OCR output shoulb be beetween xml tags <ocr> and </ocr>. """
            self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
            self.IMAGENET_STD = (0.229, 0.224, 0.225)

            

    def _build_transform_internvl2(self, input_size=448):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])


    def vlm_infer_florence(
            self,
            frame: bytes,
        ) -> str:
        """
        Run the model on a frame.
        """
        image = Image.open(io.BytesIO(frame)).convert('RGB')

        inputs = self.processor(
            text=self.task_prompt, 
            images=image, 
            return_tensors='pt'
        ).to(self.device)
        # Caster pixel_values en self.torch_dtype pour éviter le mismatch float/half
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.torch_dtype)

        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=1024, 
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        parsed_answer  = self.processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer


    def vlm_infer_internvl2(
            self,
            image_bytes,
            prompt,
            input_size=448,
            max_num=12
        ):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transform = self._build_transform_internvl2(input_size=input_size)
        image = image.resize((input_size, input_size))
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

        question = f'<image>\n{prompt}'
        generation_config = dict(max_new_tokens=256, do_sample=False)
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        return response 
    

    def perform_ocr(
            self, 
            frame: bytes,
        ) -> str:
        """
        Perform OCR on a frame.
        """
        if self.model_name == 'microsoft/Florence-2-base-ft' or self.model_name == 'microsoft/Florence-2-base':
            return self.vlm_infer_florence(
                frame, 
            )
        
        elif self.model_name == 'OpenGVLab/InternVL2-1B':
            return self.vlm_infer_internvl2(
                frame, 
                self.PROMPT_INTERNVL2
            )
