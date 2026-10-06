import io
import cv2
import numpy as np
from loguru import logger

try:  # the vlm extra, so that importing the other OCR tools does not need torch
    from PIL import Image
    import torch
    from torchvision.transforms.functional import InterpolationMode
    import torchvision.transforms as T
except ImportError:
    torch = None


PROMPT = ("Copy exactly the subtitle burned into this image, on one line, adding nothing. "
          "If there is no subtitle, answer only: -")


class VLMOCR:
    """
    VLM-based OCR implementation (using lightweight models)
    """
    
    def __init__(self, model_name: str = 'Qwen/Qwen3-VL-2B-Instruct', prompt: str = PROMPT):
        """
        Initialize VLM OCR with specified model.
        
        Args:
            model_name: Florence-2 ('microsoft/Florence-2-base-ft', 'microsoft/Florence-2-base'),
                'OpenGVLab/InternVL2-1B', or any image-text-to-text model of transformers, such as
                'Qwen/Qwen3-VL-2B-Instruct', which reads subtitles best (docs: Choosing an OCR engine).
            prompt: What an image-text-to-text model is asked; it answers "-" for no subtitle.
        """
        if torch is None:
            raise ImportError("VLMOCR needs the vlm extra: pip install 'frame2text4llm[vlm]'")
        self.model_name = model_name
        self.prompt = prompt
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    def _initialize_model(self):
        """Initialize the VLM model if not already done."""
        if self.model is None:
            try:
                from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
                
                logger.info(f"Initializing VLM model: {self.model_name}")
                
                if self.model_name in ['microsoft/Florence-2-base-ft', 'microsoft/Florence-2-base']:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name, 
                        trust_remote_code=True
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, 
                        torch_dtype=self.torch_dtype, 
                        trust_remote_code=True
                    ).to(self.device).eval()
                    
                    self.task_prompt = "<OCR>"
                    
                elif self.model_name == 'OpenGVLab/InternVL2-1B':
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
                    
                    self.prompt_internvl2 = """Extract and return only the text from the image without any additional commentary or introductory phrases. 
                    OCR output shoulb be beetween xml tags <ocr> and </ocr>. """
                    self.imagenet_mean = (0.485, 0.456, 0.406)
                    self.imagenet_std = (0.229, 0.224, 0.225)
                    
                else:  # any image-text-to-text model of transformers, such as Qwen3-VL
                    from transformers import AutoModelForImageTextToText

                    # bfloat16 where the GPU has it (compute capability 8 and up), float32 elsewhere, as measured on a T4
                    bf16 = self.device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name, dtype=torch.bfloat16 if bf16 else torch.float32, device_map=self.device
                    ).eval()
                    
                logger.info("VLM model initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize VLM model: {str(e)}")
                raise RuntimeError(f"Failed to initialize VLM model: {str(e)}")
    
    def _numpy_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy array to bytes."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    def _build_transform_internvl2(self, input_size=448):
        """Build transform for InternVL2."""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])
    
    def _infer_florence(self, frame_bytes: bytes) -> str:
        """Run Florence model inference."""
        image = Image.open(io.BytesIO(frame_bytes)).convert('RGB')
        
        inputs = self.processor(
            text=self.task_prompt, 
            images=image, 
            return_tensors='pt'
        ).to(self.device)
        
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
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer
    
    def _infer_internvl2(self, frame_bytes: bytes, input_size=448) -> str:
        """Run InternVL2 model inference."""
        image = Image.open(io.BytesIO(frame_bytes)).convert('RGB')
        transform = self._build_transform_internvl2(input_size=input_size)
        image = image.resize((input_size, input_size))
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()
        
        question = f'<image>\n{self.prompt_internvl2}'
        generation_config = dict(max_new_tokens=256, do_sample=False)
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        return response
    
    def _infer_chat(self, images: list) -> list:
        """An image-text-to-text model through its chat template, on the images doubled (BGR, as
        OpenCV reads them), read in one batch padded on the left. An answer of "-" means no subtitle."""
        messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": Image.fromarray(cv2.cvtColor(
                    cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))},
                {"type": "text", "text": self.prompt}]}]
            for image in images
        ]
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        texts = self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return ["" if t.strip() == "-" else t.strip() for t in texts]

    def process_images(self, images: list, lang: str = "en", batch_size: int = 16) -> list:
        """process_image on many images: an image-text-to-text model reads `batch_size` at a time,
        which keeps the GPU busy; Florence-2 and InternVL2 still read one by one."""
        self._initialize_model()
        if self.model_name in ['microsoft/Florence-2-base-ft', 'microsoft/Florence-2-base', 'OpenGVLab/InternVL2-1B']:
            return [self.process_image(image, lang) for image in images]
        return [t for i in range(0, len(images), batch_size) for t in self._infer_chat(images[i:i + batch_size])]

    def process_image(self, image: np.ndarray, lang: str = "en") -> str:
        """
        Extract text from image using VLM models.
        
        Args:
            image: NumPy array containing the image
            lang: Language code (not used by VLM models)
            
        Returns:
            Extracted text from the image
        """
        try:
            self._initialize_model()
            
            #convert np array to bytes
            frame_bytes = self._numpy_to_bytes(image)
            
            #run inference based on model type
            if self.model_name in ['microsoft/Florence-2-base-ft', 'microsoft/Florence-2-base']:
                result = self._infer_florence(frame_bytes)
            elif self.model_name == 'OpenGVLab/InternVL2-1B':
                result = self._infer_internvl2(frame_bytes)
            else:
                result = self._infer_chat([image])[0]
            
            return str(result) if result else ""
            
        except Exception as e:
            logger.error(f"VLM OCR failed: {str(e)}")
            raise RuntimeError(f"VLM OCR failed: {str(e)}") 