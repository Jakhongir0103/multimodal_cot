from typing import Dict, Any

def format_prompt(sample: Dict[str, Any], answers: Dict[int, Any]) -> Dict[str, Any]:
    """
    `sample` example:
    {
        "question_id": 262144000,
        "image": "COCO_test2015_000000262144.jpg",
        "text": "Is the ball flying towards the batter?\nAnswer the question using a single word or phrase.",
        "category": "default"
    }
    
    `answers` example:
    {
        262144000: 'Yes',
        ...
    }
    """
    # Main instruction
    main_instruction_prompt = "First think about the question in the mind using all relevant entities from the scene that are necessary to answer the question, along with their bounding boxes coordinates in plain text format 'x1,y1,x2,y2 object'. Then, provide with the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    # prompt
    prompt = main_instruction_prompt + '\n' + f"Question: {sample['text']}"
    answer = answers[sample['question_id']] # `question_id` must exist in answers!

    return {
        'prompt': prompt,
        'answer': answer
    }

def format_prompt_sqa(sample: Dict[str, Any], answers: Dict[int, Any]) -> Dict[str, Any]:
    """
    `sample` example:
    {
        "question_id": 262144000,
        "image": "COCO_test2015_000000262144.jpg",
        "text": "Is the ball flying towards the batter?\nAnswer the question using a single word or phrase.",
        "category": "default"
    }
    
    `answers` example:
    {
        262144000: 'Yes',
        ...
    }
    """
    # Main instruction
    main_instruction_prompt = "First think about the question in the mind using all relevant entities from the scene that are necessary to answer the question, along with their bounding boxes coordinates in plain text format 'x1,y1,x2,y2 object'. Then, provide with the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    # prompt
    prompt = main_instruction_prompt + '\n' + f"Question: {sample['text']}" + "\nAs the final answer, give directly the option's letter from the given choices."
    answer = answers[sample['question_id']] # `question_id` must exist in answers!

    return {
        'prompt': prompt,
        'answer': answer
    }