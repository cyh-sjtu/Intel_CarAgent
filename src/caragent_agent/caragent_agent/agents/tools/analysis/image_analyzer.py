from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image
import asyncio

from caragent_agent.agents.tools.base.tool_base import ToolBase
from caragent_agent.utils.llm_handler import UnifiedLLMClient
from caragent_agent.utils.llm_request_generator import vlm_analyse_on_each_kf_images_request_message, vlm_multi_images_request_message_co_analysis, extract_answer_tags
from caragent_agent.config.config import config

class ImageAnalyzerTool(ToolBase):
    def __init__(self):
        super().__init__(
            name="analyse_on_each_kf_images", 
            description="""
                Analyzes selected keyframe images using vision-language models to answer questions.
                 
                This is supplemental visual evidence, not the default next step
                after keyframe search. Prefer keyframe semantics, positions, and
                compact candidate evidence first. Use this tool only when those
                metadata fields cannot distinguish a small set of plausible
                candidates, or when the active task explicitly requires visual
                inspection of stored keyframe images.

                Processes multiple keyframe images in parallel through batch requests to a VLM 
                (Vision-Language Model) endpoint. Returns structured answers for each analyzed frame.

                Args:
                    kf_id_list (List[int]): 
                        Target keyframe IDs from scene memory. Valid IDs should exist in 
                        `self.scene_memory.keyframe_nodes`
                    question (str): 
                        Natural language query to analyze against visual content. Should be 
                        specific enough for visual understanding (e.g., "What objects are visible?" 
                        or "Describe the scene layout")

                Returns:
                    Dict[int, str]: 
                        Analysis results mapping with:
                        - Keys: Original keyframe IDs from input list
                        - Values: Model-generated answers in natural language

                Example:
                    >>> toolkit.analyse_on_each_kf_images([6], "How many books are included in the image? And discribe where they are.")
                    {6: 'There are three books in the image. They are all located on the desk:\n- The first book, with a white cover featuring "OpenCV," is on the left.\n- The second book, with a blue cover and technical imagery, is in the middle.\n- The third book, also with a blue cover and similar technical imagery, is on the right.'}
                """,
            capability_tags=("scene_memory_search", "background_safe"),
        )
        self.llm_client = UnifiedLLMClient()
        
    def execute(self, kf_id_list: List[int], question: str) -> Dict[int, str]:
        try:
            requests_list = []
            for kf_id in kf_id_list:
                request_metadata = {}
                request_metadata["request_id"] = kf_id
                request_metadata['model'] = config['vlm_model_analyse_images']

                image_path = self.scene_memory.keyframe_nodes[kf_id].rgb_path
                image = Image.open(image_path)

                messages = vlm_analyse_on_each_kf_images_request_message(image, question)
                request_metadata['messages'] = messages
                requests_list.append(request_metadata)

            client = UnifiedLLMClient()
            results = asyncio.run(client.batch_chat_completion(requests_list))

            analyse_answer = {}
            for req_id, response in results.items():
                analyse_answer[int(req_id)] = extract_answer_tags(response[req_id])

            return self.ok(
                "Analyzed the requested keyframe images.",
                data={
                    "question": question,
                    "requested_keyframe_ids": [int(kf_id) for kf_id in kf_id_list],
                    "answers_by_keyframe": analyse_answer,
                },
                provenance={"source_type": "scene_memory"},
            )
        except Exception as exc:
            return self.error_result(
                "Keyframe image analysis failed.",
                data={
                    "question": question,
                    "requested_keyframe_ids": [int(kf_id) for kf_id in kf_id_list],
                },
                error={
                    "code": "image_analysis_failed",
                    "message": str(exc),
                },
                provenance={"source_type": "scene_memory"},
            )

# TODO: Some bugs inside
class MultiImageAnalyzerTool(ToolBase):
    def __init__(self):
        super().__init__(
            name="co_analyse_on_kf_images",
            description="""
                Analyse multiple keyframe images together by stitching them into a grid and asking a question using a vision-language model (VLM).
                
                This method combines the RGB images of the specified keyframes into a single grid image, then sends this image along with a natural language question to a VLM for joint analysis. The model's answer reflects the similarities, differences, or other relationships among the provided images.

                Args:
                    kf_id_list (List[int]):
                        List of keyframe node IDs whose images will be stitched and analyzed together. Each ID should correspond to a valid keyframe in `self.scene_memory.keyframe_nodes`.
                    question (str):
                        Natural language question to ask about the combined images. Example: "What are the differences and similarities among these images?"

                Returns:
                    str:
                        The answer generated by the VLM, typically a natural language description comparing or analyzing the input images.

                Example:
                    >>> toolkit.co_analyse_on_kf_images([0, 1, 2, 3, 4], "What are the differences and similarities among these images?")
                    'All images show a desk, but the objects on the desk differ: image 0 contains a laptop, image 1 contains a book, ...'
                """,
            capability_tags=("scene_memory_search", "background_safe"),
        )
        self.llm_client = UnifiedLLMClient()
        
    def execute(self, kf_id_list: List[int], question: str) -> str:
        try:
            kf_set = [self.scene_memory.keyframe_nodes[kf_id] for kf_id in kf_id_list]
            client = UnifiedLLMClient()
            requests_list = []
            request_metadata = {}
            request_metadata["request_id"] = 0
            request_metadata['model'] = config['vlm_model_analyse_images']
            messages = vlm_multi_images_request_message_co_analysis(kf_set, question)
            request_metadata['messages'] = messages
            requests_list.append(request_metadata)
            results = asyncio.run(client.batch_chat_completion(requests_list))
            for req_id, response in results.items():
                print(response)
                return self.ok(
                    "Analyzed the requested keyframe images jointly.",
                    data={
                        "question": question,
                        "requested_keyframe_ids": [int(kf_id) for kf_id in kf_id_list],
                        "joint_answer": extract_answer_tags(response[req_id]),
                    },
                    provenance={"source_type": "scene_memory"},
                )
        except Exception as e:
            return self.error_result(
                "Collaborative keyframe analysis failed.",
                data={
                    "question": question,
                    "requested_keyframe_ids": [int(kf_id) for kf_id in kf_id_list],
                },
                error={
                    "code": "collaborative_analysis_failed",
                    "message": str(e),
                },
                provenance={"source_type": "scene_memory"},
            )
        
class CurrentImageAnalyzerTool(ToolBase):
    def __init__(self, controller):
        super().__init__(
            name="analyse_on_current_image", 
            description="""
                Analyzes the current image from the robot controller using a vision-language model (VLM) to answer questions.
                
                Captures the current RGB image and processes it through a VLM to provide insights based on the visual content.

                Args:
                    question (str): 
                        Natural language query to analyze against the current visual content. Should be 
                        specific enough for visual understanding (e.g., "What objects are visible?" 
                        or "Describe the scene layout")

                Returns:
                    str: 
                        Model-generated answer in natural language

                Example:
                    >>> toolkit.analyse_on_current_image("What objects are visible in the current view?")
                    'In the current view, there is a tree on the left side and a building in the background.'
                """,
            capability_tags=("live_view", "background_unsafe"),
        )
        self.llm_client = UnifiedLLMClient()
        self.controller = controller
        
    def execute(self, question: str) -> str:
        try:
            if self.controller is None:
                return self.blocked(
                    "Controller is unavailable, so the current image cannot be analyzed.",
                    data={"question": question},
                    error={
                        "code": "controller_unavailable",
                        "message": "Controller reference is missing.",
                    },
                    provenance={"source_type": "controller"},
                )

            image = self.controller.get_current_image()
            if image is None:
                return self.blocked(
                    "Current image is unavailable, so live image analysis cannot proceed.",
                    data={"question": question},
                    error={
                        "code": "current_image_unavailable",
                        "message": "Controller returned no current image.",
                    },
                    provenance={"source_type": "controller"},
                )
            image_metadata = {}
            try:
                get_metadata = getattr(self.controller, "get_current_image_metadata", None)
                if callable(get_metadata):
                    image_metadata = get_metadata() or {}
            except Exception:
                image_metadata = {}

            request_metadata = {}
            request_metadata["request_id"] = 0
            request_metadata['model'] = config['vlm_model_analyse_images']

            messages = vlm_analyse_on_each_kf_images_request_message(image, question)
            request_metadata['messages'] = messages

            client = UnifiedLLMClient()
            results = asyncio.run(client.batch_chat_completion([request_metadata]))
            
            for req_id, response in results.items():

                return self.ok(
                    (
                        "Analyzed the current image."
                        if image_metadata.get("source_type") != "simulated_keyframe_view"
                        else "Analyzed a simulated current image from historical keyframe memory."
                    ),
                    data={
                        "question": question,
                        "answer": extract_answer_tags(response[req_id]),
                        "image_source": image_metadata,
                    },
                    provenance=image_metadata or {"source_type": "live_view"},
                )
        except Exception as e:
            print("question", question)
            return self.error_result(
                "question" + str(question) + "Current image analysis failed." + str(e),
                data={"question": question},
                error={
                    "code": "current_image_analysis_failed",
                    "message": str(e),
                },
                provenance={"source_type": "live_view"},
            )

# Smallest test unit
if __name__ == "__main__":
    from pathlib import Path
    from caragent_agent.config.runtime_paths import get_default_scene_dataset_dir
    from caragent_agent.impression_graph.scene_memory import SceneMemory
    dataset_dir = get_default_scene_dataset_dir()
    scene_memory = SceneMemory(dataset_dir)
    scene_memory.load_keyframe_nodes()
    scene_memory.load_keyframe_graph()
    tool = ImageAnalyzerTool()
    tool.scene_memory = scene_memory
    print(tool.execute([0, 1, 2], "describe the scene"))
    # tool = MultiImageAnalyzerTool()
    # tool.scene_memory = scene_memory
    # print(tool.execute([0, 1, 2, 3, 4], "What are the differences and similarities among the keyframes?"))
