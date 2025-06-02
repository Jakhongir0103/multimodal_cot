# Smart Resize: https://arc.net/l/quote/yzayryvr

from typing import Dict, Any
from qwen_vl_utils import smart_resize

def xywh_to_xyxy(box):
    """Convert COCO format [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]."""
    x_min, y_min, width, height = box
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]

def resize_bbox(bbox, width, height, input_width, input_height):
    abs_x1, abs_y1, abs_x2, abs_y2 = bbox
    # Convert actual coords [abs_x1, abs_y1, abs_x2, abs_y2] to model input/output coords
    input_x1 = int(abs_x1 / width * input_width)
    input_y1 = int(abs_y1 / height * input_height)
    input_x2 = int(abs_x2 / width * input_width)
    input_y2 = int(abs_y2 / height * input_height)

    return [input_x1, input_y1, input_x2, input_y2]

def format_prompt(sample: Dict[str, Any], max_pixels: int=1920*1920, min_pixels: int=4*4) -> Dict[str, Any]:
    """
    Sample example:
    {'question': 'what is the contact person name mentioned in letter?',
    'answer': 'P. Carter',
    'possible_answers': ['P. Carter', 'p. carter'],
    'image': 'xnbl0037_1.png',
    'width': 1695,
    'height': 2025,
    'bboxs': [[429, 511, 666, 578], [429, 511, 666, 578]],
    'dataset': 'docvqa',
    'split': 'train'
    'image_path': <image_path>}
    """
    # Main instruction
    main_instruction_prompt = "First think about the question in the mind using all relevant entities from the scene that are necessary to answer the question, along with their bounding boxes coordinates in plain text format 'x1,y1,x2,y2 object'. Then, provide with the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    # Reformat bbox
    img_width, img_height = sample['width'], sample['height']
    input_height, input_width = smart_resize(img_height, img_width, min_pixels=min_pixels, max_pixels=max_pixels)
    # print(f"{img_width}x{img_height} -> {input_width}x{input_height}")

    bboxes = sample['bboxs']
    # bboxes_resized = [resize_bbox(xywh_to_xyxy(bbox), width=img_width, height=img_height, input_width=input_width, input_height=input_height) for bbox in bboxes]
    bboxes_resized = [resize_bbox(bbox, width=img_width, height=img_height, input_width=input_width, input_height=input_height) for bbox in bboxes]

    # prompt
    prompt = main_instruction_prompt + '\n' + f"Question: {sample['question']}"
    possible_answers = sample['possible_answers'] if 'possible_answers' in sample else [sample['answer']]

    return {
        'prompt': prompt,
        'bboxes': bboxes_resized,
        'possible_answers': possible_answers
    }