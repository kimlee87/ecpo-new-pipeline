from utils import discover_images

import asyncio
import base64
import cv2
import json
import numpy as np
import openai
import pathlib
import tqdm


OCR_PROMPT_TEMPLATE = """
Your task is to OCR this image written in traditional chinese.
The reading order is {reading_order}.

Your result needs to fulfill **all** of these constraints:
* Give the result **exactly** as it appears on the image
* Keep line breaks from the original
* Do not modify to modern chinese, keep exactly as is.
* Denote numbers exactly like in the image, not in english writing.
* The result needs to follow the {reading_order} reading order.
* If the text does not start in the top-right corner, start it in the right-most column.

Double-check your response so that it fulfills all constraints.
"""


READING_ORDER_PROMPT = """
You are an expert in classical Chinese paleography and historic East Asian page layouts.
Analyze the image and determine the correct reading order of the text.

Tasks:
* Identify the text line orientation: "vertical" or "horizontal"
* Identify column reading direction: "right-to-left" or "left-to-right" or "top-to-bottom"

Output format (no explanations, no reasoning):

<orientation>,  <direction>

Only produce the structured fields. No commentary or explanation.”
"""


_client = openai.AsyncOpenAI(
    api_key="",
    base_url="http://localhost:8080/v1",
)
_semaphore = asyncio.Semaphore(100)


_bar_ocr = None
_bar_reading_order = None


def preprocess(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel_sharp)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharp)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=18)

    return denoised


async def vllm_request(prompt, image):
    async with _semaphore:
        response = await _client.chat.completions.create(
            model="Qwen/Qwen3-VL-32B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image},
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0,
            extra_body={"repetition_penalty": 1.0, "top_k": 0, "top_p": 1.0},
        )

        return response.choices[0].message.content


async def handle_image(image):
    img = cv2.imread(str(image))
    img = preprocess(img)
    _, im_arr = cv2.imencode(".png", img)
    image_base64 = base64.b64encode(im_arr).decode("utf-8")

    # Build a data URL for the image
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    reading_order = await vllm_request(READING_ORDER_PROMPT, image_data_url)

    _bar_reading_order.update(1)

    result = await vllm_request(
        OCR_PROMPT_TEMPLATE.format(reading_order=reading_order), image_data_url
    )

    _bar_ocr.update(1)

    return result


async def perform_vllm_ocr(images: list[pathlib.Path]):
    image_tasks = {
        i.stem: asyncio.create_task(handle_image(i)) for i in discover_images(images)
    }

    global _bar_ocr, _bar_reading_order
    _bar_ocr = tqdm.tqdm(total=len(image_tasks), desc="OCR images", position=1)
    _bar_reading_order = tqdm.tqdm(
        total=len(image_tasks), desc="Determining reading order", position=0
    )

    await asyncio.gather(*image_tasks.values())

    results = {}
    for stem, it in image_tasks.items():
        results[stem] = it.result()

    return results


if __name__ == "__main__":
    result = asyncio.run(
        perform_vllm_ocr(
            discover_images(pathlib.Path("/home/dkempf/ecpo-ocr-bench/data"))
        )
    )

    with open("result.json", "w") as f:
        json.dump(result, f)
