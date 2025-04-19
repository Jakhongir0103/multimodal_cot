import os
import re
import json
import requests
from tqdm import tqdm
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

resq_dir_path = "/lid/home/saydalie/multimodal_cot/SEED/data/ReSQ/"

client = OpenAI()

def remove_repetitive_sentences(text):
    """
    Removes repetitive sentences from the given text while preserving order.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique_sentences = []

    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)

    cleaned_text = ' '.join(unique_sentences)

    if cleaned_text and not re.search(r'[.!?]$', cleaned_text):
        cleaned_text += '.'
        
    return cleaned_text

def load_data(data_path, out_data_path):
    with open(data_path, 'r')as f:
        data_questions = json.load(f)['data']

    data_all = []   # not used
    unique_scenes = []

    for d in data_questions:
        story = d['story']
        questions = d['questions']
        
        for q in questions:
            question = q['question']
            answer = q['answer'][0]
            candidate_answers = q['candidate_answers']
            num_1st_context_sentences = q['num_1st_context_sentences']

            scene = remove_repetitive_sentences(' '.join(story[:num_1st_context_sentences]))
            data_all.append({
                'scene': scene,
                'question': question,
                'candidate_answers': candidate_answers,
                'answer': answer
            })

            if scene not in unique_scenes:
                unique_scenes.append(scene)

    data = pd.DataFrame({'scene': unique_scenes})
    data.reset_index(inplace=True)

    data_all_df = pd.merge(pd.DataFrame(data_all), data, on='scene', how='left')
    data_all_df.to_csv(out_data_path, index=False)

    return data

def get_image_from_DALL_E_3_API(
    prompt,
    model="dall-e-3",
    style='natural',
    quality='standard',
    size="1024x1024",
    n=1,
):
  if model == "dall-e-3":
    response = client.images.generate(
      model=model,
      prompt=prompt,
      style=style,
      quality=quality,
      n=n,
      size=size
    )
  else:
    response = client.images.generate(
      model=model,
      prompt=prompt,
      n=n,
      size=size
    )

  image_url = response.data[0].url
  return image_url

if __name__=='__main__':
    # set a directory to save DALL·E images to
    image_dir_name = resq_dir_path + "images"
    image_dir = os.path.join(os.curdir, image_dir_name)

    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # load the data
    data = load_data(
        data_path = resq_dir_path+'train_resq.json',
        out_data_path = resq_dir_path+'train_resq_clean.csv'
    )

    # generate and store images
    generated_images_filepaths = {0: [], 1: [], 2: []}
    for _, row in tqdm(data.iterrows(), desc="Generating Images", total=data.shape[0]):
        prompt = (
            f"{row['scene']} "
            "a clear spatial relationship, minimalistic and natural perspective, full wide view."
        )

        # save the image
        generated_image_dir_name = f"{row['index']:04}"
        generated_image_dir = os.path.join(image_dir, generated_image_dir_name)

        if not os.path.isdir(generated_image_dir):
            os.mkdir(generated_image_dir)

        for idx in range(3):
            generated_image_name = f"{idx:02}.png"
            generated_image_filepath = os.path.join(generated_image_dir, generated_image_name)

            if os.path.exists(generated_image_filepath):
                generated_images_filepaths[idx].append(generated_image_filepath)
                continue

            try:
                generated_image_url = get_image_from_DALL_E_3_API(prompt=prompt, quality='standard')
                generated_image = requests.get(generated_image_url).content

                with open(generated_image_filepath, "wb") as image_file:
                    image_file.write(generated_image)
                
                generated_images_filepaths[idx].append(generated_image_filepath)
            except Exception as e:
                generated_images_filepaths[idx].append(e)
    
    data['image_0'] = generated_images_filepaths[0]
    data['image_1'] = generated_images_filepaths[1]
    data['image_2'] = generated_images_filepaths[2]

    data.to_csv(resq_dir_path + 'train_resq_images_paths.csv', index=False)