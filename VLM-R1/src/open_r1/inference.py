from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize

model_name_or_path = "/lid/home/saydalie/multimodal_cot/VLM-R1/output/merged/Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format/checkpoint-79768"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path, torch_dtype="auto", device_map="auto"
)

# default processor
processor = AutoProcessor.from_pretrained(model_name_or_path, min_pixels=56*56, max_pixels=1920*1920)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///lid/home/saydalie/multimodal_cot/VLM-R1/data/aokvqa/images/train2017/000000323820.jpg",
            },
            {"type": "text", "text": "First think about the question in the mind using all relevant entities from the scene that are necessary to answer the question, along with their bounding boxes coordinates in plain text format 'x1,y1,x2,y2 object'. Then, provide with the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.\nQuestion: What food here comes from outside a farm?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)