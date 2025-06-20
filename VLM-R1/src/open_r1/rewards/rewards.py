import os
import re
from datetime import datetime

def xywh_to_xyxy(box):
    """Convert COCO format [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]."""
    x_min, y_min, width, height = box
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]

def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (list of int): [x_min, y_min, x_max, y_max] for the first box.
        box2 (list of int): [x_min, y_min, x_max, y_max] for the second box.
    
    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection coordinates
    x_int_min = max(x1_min, x2_min)
    y_int_min = max(y1_min, y2_min)
    x_int_max = min(x1_max, x2_max)
    y_int_max = min(y1_max, y2_max)

    # Compute intersection area
    int_width = max(0, x_int_max - x_int_min)
    int_height = max(0, y_int_max - y_int_min)
    intersection_area = int_width * int_height

    # Compute area of both boxes
    area1 = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    area2 = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)

    # Compute union area
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def extract_bounding_boxes(text):
    """
    Extracts bounding boxes and their labels from a text string.
    
    Args:
        text (str): Input text containing patterns like "[x1,y1,x2,y2 label]".
    
    Returns:
        dict: A dictionary mapping label (str) to bounding box coordinates (list of int).
              Only correctly formatted entries are included.
    """
    pattern = r'\[(\d+),(\d+),(\d+),(\d+)\s+([^\[\]]+?)\]'
    matches = re.findall(pattern, text)
    
    results = {}
    for match in matches:
        x, y, w, h, label = match
        try:
            box = [int(x), int(y), int(w), int(h)]
            results[label.strip().lower()] = box
        except ValueError:
            continue  # skip if conversion to int fails
    
    return results

def bbox_reward(completions, **kwargs):
    """Computes the IoU reward between generated bounding box and the ground truth bounding boxes."""
    contents = [completion[0]["content"] for completion in completions]
    solutions = kwargs['solution']
    rewards = []
    for idx, (content, solution) in enumerate(zip(contents, solutions)):
        reward = 0.0
        predicted_bboxes = None
        try:
            # Extract answer from solution if it has think tags
            sol_match = re.search(r'<think>(.*?)</think>', solution, re.DOTALL)
            true_think = sol_match.group(1).strip() if sol_match else solution.strip().lower()
            true_bboxes = extract_bounding_boxes(true_think)
            
            # Extract answer from content if it has think tags
            content_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            predicted_think = content_match.group(1).strip() if content_match else content.strip().lower()
            predicted_bboxes = extract_bounding_boxes(predicted_think)

            if predicted_bboxes:
                for label, predicted_bbox in predicted_bboxes.items():
                    if label in true_bboxes:
                        # If the exact label is found from true, use its bbox
                        iou = compute_iou(predicted_bbox, true_bboxes[label])
                        reward += iou
                    elif true_bboxes:
                        # Otherwise, use the bbox from the true answer with the maximum iou value
                        iou = max([compute_iou(predicted_bbox, true_bbox) for true_bbox in true_bboxes.values()])
                        reward += iou
                reward = reward / len(predicted_bboxes)
        except Exception:
            pass

        rewards.append(reward)
    return rewards

def normalize_answer(answer):
    """Normalizes an answer by stripping whitespace, converting to lowercase, and removing punctuation."""
    return re.sub(r'[^a-zA-Z0-9]', '', answer).lower()

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    solution = kwargs['solution']
    rewards = []
    for idx, (content, sol) in enumerate(zip(contents, solution)):
        reward = 0.0
        ground_truth = None
        predicted_answer = None
        if reward == 0.0:
            try:
                # Extract answer from solution if it has answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip().lower()
                
                # Extract answer from content if it has answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                predicted_answer = content_match.group(1).strip() if content_match else content.strip().lower()
                
                # Compare the extracted answers
                if predicted_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Image Path: {kwargs['image_path'][idx]}\n")
                f.write(f"Completion: {content}\n")
                f.write(f"Solution: {sol}\n")
                f.write(f"Golden: `{ground_truth}`\tPredicted: `{predicted_answer}`\n")
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    # return [1.0 if match else 0.0 for match in matches]
    return [0.5 if match else 0.0 for match in matches]