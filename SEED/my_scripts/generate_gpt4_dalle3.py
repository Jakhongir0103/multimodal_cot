import os
import re
import time
import base64
from ast import literal_eval

from tqdm import tqdm
from dotenv import load_dotenv

import pandas as pd

load_dotenv()

from openai import OpenAI
# api_key = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = OpenAI()

DATA_DIR_PATH = "/lid/home/saydalie/multimodal_cot/SEED/data/ReSQ/"

# SYSTEM_MESSAGE = (
#     "Given the question, think step by step using the provided context and the image to answer the question, then, provide with the final answer. "
#     "The final answer is either 'Yes' or 'No'. If image and text contradict each other, please follow the text. "
#     "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
#     "i.e., <think> reasoning process here </think> <answer> the final answer here </answer>."
# )

SYSTEM_MESSAGE = (
    "Given the question, think step by step using the provided context to answer the question, then, provide with the final answer. "
    "The final answer is either 'Yes' or 'No'. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> the final answer here </answer>."
)

def capitalize_sentences(text: str) -> str:
    """
    Capitalizes the first letter of every sentence in the given text.

    A sentence is considered to end with '.', '!', or '?'. Handles multiple spaces and newlines.
    """
    # Pattern to match sentence endings followed by spaces/newlines
    sentence_endings = re.compile(r'([.!?]\s*)')
    parts = sentence_endings.split(text)
    
    # Recombine parts while capitalizing the first character of each sentence
    result = ''
    capitalize_next = True
    for part in parts:
        if capitalize_next and part.strip():
            result += part.lstrip().capitalize()
            capitalize_next = False
        else:
            result += part
        if re.match(sentence_endings, part):
            capitalize_next = True
    return result

def load_data():
    # data_resq_clean = pd.read_csv(DATA_DIR_PATH + "train_resq_clean.csv")
    # data_resq_images = pd.read_csv(DATA_DIR_PATH + "train_resq_images_paths.csv")
    # data = pd.merge(data_resq_clean, data_resq_images, on=['index', 'scene'], how='left')

    # data['images_paths'] = data.apply(lambda sample: [sample['image_0'], sample['image_1'], sample['image_2']], axis=1)
    # data.drop(columns=['image_0', 'image_1', 'image_2'], inplace=True)
    # data['scene'] = data['scene'].map(capitalize_sentences)

    data = pd.read_csv(DATA_DIR_PATH + "train_resq_with_answers.csv")
    data['images_paths'] = data['images_paths'].apply(lambda x: literal_eval(x))
    
    return data

# Define a function to get paraphrases using OpenAI API or using API from third-party service providers
def get_gpt4answer(input_text, base64_image=None):
    user_content = [{"type": "input_text", "text": input_text}]
    if base64_image:
        user_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"})

    for _ in range(10):
        try:
            # Call the OpenAI API
            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue

        answer = response.output[0].content[0].text
        if not answer:
            continue
        if 'I\'m sorry' in answer:
            print(answer)
            continue
        return answer
    return ""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    data = load_data()

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):

        if idx >= 350:
            break
        
        # Check if answers already exist for this row
        # if 'answers_with_image' in data.columns and pd.notna(row.get('answers_with_image')):
        if 'answers_without_image' in data.columns and pd.notna(row.get('answers_without_image')):
            print(f"Skipping {idx}...")
            continue
        
        input_text = row['scene'] + " " + row["question"]
        outputs = []

        for img_path in row['images_paths']:
            base64_image = encode_image(img_path)

            for _ in range(3):
                time.sleep(1)
                # output = get_gpt4answer(input_text, base64_image)
                output = get_gpt4answer(input_text)
                outputs.append(output)
        
        # data.at[idx, 'answers_with_image'] = f"{outputs}"
        data.at[idx, 'answers_without_image'] = f"{outputs}"
        data.to_csv(DATA_DIR_PATH + 'train_resq_with_answers.csv', index=False)