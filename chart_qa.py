
import logging
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if  i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width  = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width  // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width  // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def preprocess_image(image: Image.Image, input_size=448, max_num=6):
    image = image.convert("RGB")
    image = image.resize((input_size, input_size))
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


class ChartQA:
    def __init__(self, model_path: str = "./models_local/vintern", device: str = "cuda"):
        self.device = "cuda"
        self.model = None
        self.tokenizer = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            logger.info(f"Loading Vintern from {model_path} ...")
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval().to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
            logger.info("Vintern loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Vintern: {e}")
            raise

    def _build_prompt(self, question: str, chart_type: str, extracted_data: str,
                      is_first_turn: bool) -> str:
        if is_first_turn:
            context_parts = []
            if chart_type and chart_type != "unknown":
                context_parts.append(f"Loại biểu đồ: {chart_type}")
            if extracted_data:
                context_parts.append(f"Dữ liệu trích xuất từ biểu đồ:\n{extracted_data}")
            context_str = "\n".join(context_parts)

            if context_str:
                return (
                    f"<image>\n{context_str}\n\n"
                    f"Câu hỏi: {question}\n\n"
                    f"Hãy dựa vào bảng trích xuất và trả lời câu hỏi\n"
                )
            else:
                return f"<image>\nCâu hỏi: {question}"
        else:
            return question

    def answer_with_history(
        self,
        image: Image.Image,
        question: str,
        chart_type: str = "unknown",
        extracted_data: str = "",
        history: list = None,
        max_new_tokens: int = 1024,
    ) -> tuple[str, list]:

        if self.model is None:
            return "Model chưa được load.", []

        try:
            pixel_values = preprocess_image(image).to(torch.bfloat16).to(self.device)

            prompt = self._build_prompt(question, chart_type, extracted_data, is_first_turn=True)

            generation_config = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=2.0,
                num_beams=3,
            )

            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
                history=None,
                return_history=True,
            )

            return response.strip(), []

        except Exception as e:
            logger.error(f"Vintern inference error: {e}")
            return f"[Lỗi: {e}]", []


    def answer(self, image: Image.Image, question: str,
               chart_type: str = "unknown", extracted_data: str = "",
               max_new_tokens: int = 1024) -> str:
        response, _ = self.answer_with_history(
            image=image, question=question,
            chart_type=chart_type, extracted_data=extracted_data,
            history=None, max_new_tokens=max_new_tokens,
        )
        return response
