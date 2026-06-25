"""Helpers to build LLM/VLM request payloads and encode images for prompts."""

from caragent_agent.third_party.from_vlm_grounder.grid_image_generator import dynamic_stitch_images_fix_v2
from caragent_agent.config.config import config

from io import BytesIO
import base64
import os
import re
from PIL import Image
import yaml
from typing import List, Dict
import numpy as np
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent
PROMPT_DIR = base_path / "prompts"

if not (PROMPT_DIR / "scene_memory_prompts.yaml").exists():
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        candidate = Path(prefix) / "share" / "caragent_agent" / "prompts"
        if (candidate / "scene_memory_prompts.yaml").exists():
            PROMPT_DIR = candidate
            break
    else:
        for parent in base_path.parents:
            candidate = parent / "src" / "caragent_agent" / "caragent_agent" / "prompts"
            if (candidate / "scene_memory_prompts.yaml").exists():
                PROMPT_DIR = candidate
                break

with open(PROMPT_DIR / "scene_memory_prompts.yaml", "r", encoding="utf-8") as f:
    scene_memory_prompts = yaml.safe_load(f)

with open(PROMPT_DIR / "tool_prompts.yaml", "r", encoding="utf-8") as f:
    tool_prompts = yaml.safe_load(f)

with open(PROMPT_DIR / "agent_prompts.yaml", "r", encoding="utf-8") as f:
    agent_prompts = yaml.safe_load(f)

def encode_PIL_image_to_base64(image):
    """Encode a PIL image to a base64 JPEG string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def stitch_images_to_base64(image_paths, image_ids):
    """Create stitched grids with ids overlaid and return base64 strings."""
    images = [Image.open(path) for path in image_paths]

    grid_images = dynamic_stitch_images_fix_v2(
        images,
        ID_array=image_ids,
        relative_ID_size=0.05,
        ID_color="red"
    )

    return [encode_PIL_image_to_base64(img) for img in grid_images]

def divide_nodes_into_subsets(nodes_dict: Dict, subset_size: int) -> List[Dict]:
    """Split a node dictionary into fixed-size subsets."""
    result = []
    subset = {}
    count = 0
    
    for key, value in nodes_dict.items():
        subset[key] = value
        count += 1
        
        if count == subset_size:
            result.append(subset)
            subset = {}
            count = 0
            
    if subset:
        result.append(subset)
        
    return result

def extract_and_convert_ids(text):
    """Extract numeric ids from <answer>[...]</answer> tags."""
    if not isinstance(text, str) or not text.strip():
        return []

    pattern = r'<answer>\s*\[(.*?)\]\s*</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    id_lists = []
    for match in matches:
        ids = []
        for id_str in match.split(','):
            stripped = id_str.strip()
            if stripped.isdigit():
                ids.append(int(stripped))

        if ids:
            id_lists += ids
    
    return id_lists

def extract_answer_tags(text):
    """Return content inside <answer> tags or the stripped text."""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    answer = None
    if match:
        answer = match.group(1).strip()
    else:
        answer = text.strip()
    return answer

def vlm_single_image_request_message_kf(kf):
    """Build VLM request for a single keyframe image."""
    image_path = kf.rgb_path
    image_id = kf.kf_id
    image = Image.open(image_path)
    base64_image = encode_PIL_image_to_base64(image)
    messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": scene_memory_prompts['vlm_give_a_kf_image_semantic']}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            },
        ]
    return messages

def generate_far_near_mask_image(depth_image, near_mask_distance_threshold=5.0, far_mask_distance_threshold=10.0):
    """Generate near/far binary masks from a depth image (meters thresholds)."""
    depth_array = np.array(depth_image)

    mask_near = (depth_array <= near_mask_distance_threshold * 1000).astype(np.uint8) * 255
    mask_far = (depth_array <= far_mask_distance_threshold * 1000).astype(np.uint8) * 255

    mask_near_image = Image.fromarray(mask_near, mode='L')
    mask_far_image = Image.fromarray(mask_far, mode='L')

    return mask_near_image, mask_far_image

def vlm_single_image_request_message_kf_far_near_mask(kf):
    """Build VLM request with near/far masked RGB variants stitched together."""
    rgb_path = kf.rgb_path
    depth_path = kf.depth_path

    image_id = kf.kf_id
    original_image = Image.open(rgb_path)
    depth_image = Image.open(depth_path)

    near_mask_distance_threshold = config['near_mask_distance_threshold']
    far_mask_distance_threshold = config['far_mask_distance_threshold']

    mask_near, mask_far = generate_far_near_mask_image(depth_image=depth_image,
                                                       near_mask_distance_threshold=near_mask_distance_threshold, 
                                                       far_mask_distance_threshold=far_mask_distance_threshold)

    masked_image_near = Image.composite(original_image, Image.new("RGB", original_image.size, (0, 0, 0)), mask_near)
    masked_image_far = Image.composite(original_image, Image.new("RGB", original_image.size, (0, 0, 0)), mask_far)

    stitch_far_near = Image.new('RGB', (original_image.width * 3, original_image.height))
    stitch_far_near.paste(masked_image_near, (0, 0))
    stitch_far_near.paste(masked_image_far, (original_image.width, 0))
    stitch_far_near.paste(original_image, (original_image.width * 2, 0))

    stitch_base64 = encode_PIL_image_to_base64(stitch_far_near)

    messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": scene_memory_prompts['vlm_give_a_kf_image_semantic_far_near_mask']}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{stitch_base64}"
                        },
                    },
                ],
            },
        ]
    return messages

def llm_search_keywords_on_kf_request_message(keyframe_nodes_dict, keywords, logic):
    """Build LLM request to search keyframes by keywords with AND/OR logic."""
    id_keyframe_dict = {}
    for keyframe_node in keyframe_nodes_dict.values():
        id_keyframe_dict[keyframe_node.kf_id] = keyframe_node.semantic

    user_content = f'<keyframe node dictionary>{id_keyframe_dict}</keyframe node dictionary>, <keywords list>{keywords}</keywords list>'

    if logic == "and":
        system_prompt = tool_prompts['llm_search_keywords_on_kf_request_message_system_and']
    elif logic == "or":
        system_prompt = tool_prompts['llm_search_keywords_on_kf_request_message_system_or']
    else:
        raise ValueError("Invalid logic. Logic should be 'and' or 'or'.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages

def llm_search_requirement_on_kf_request_message(keyframe_nodes_dict, requirement):
    """Build LLM request to search keyframes matching a requirement string."""
    id_keyframe_dict = {}
    for keyframe_node in keyframe_nodes_dict.values():
        id_keyframe_dict[keyframe_node.kf_id] = keyframe_node.semantic

    user_content = f'<keyframe node dictionary>{id_keyframe_dict}</keyframe node dictionary>, <requirement string>{requirement}</requirement string>'
    system_prompt = tool_prompts['llm_search_requirement_on_kf_request_message_system']
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages

def vlm_analyse_on_each_kf_images_request_message(image, question):
    """Build VLM request to analyze a single image with a question."""
    base64_image = encode_PIL_image_to_base64(image)
    messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": tool_prompts['vlm_model_analyse_each_image_system']}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": f"<question>{question}</question>"
                    },
                ],
            },
        ]
    return messages

def vlm_multi_images_request_message_co_analysis(kf_set, question):
    """Build VLM request to co-analyze multiple keyframe images with a question."""
    image_paths = [kf_node.rgb_path for kf_node in kf_set]
    image_ids = [kf_node.kf_id for kf_node in kf_set]
    base64_images = stitch_images_to_base64(image_paths, image_ids)
    user_image_contents = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            },
        }
        for img_base64 in base64_images
    ]
    user_image_contents.append({
        "type": "text",
        "text": f"<question>{question}</question>"
    })
    messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": tool_prompts['vlm_model_co_analyse_images_system']}],
            },
            {
                "role": "user",
                "content": user_image_contents
            },
        ]
    return messages

def get_react_agent_system_prompt():
    """Return the system prompt for the agent."""
    return agent_prompts['react_agent_system']
