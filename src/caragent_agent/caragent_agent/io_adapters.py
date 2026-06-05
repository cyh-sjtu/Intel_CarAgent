"""Boundary adapters for multilingual text and image I/O."""

from __future__ import annotations

import asyncio
import base64
import re
from io import BytesIO
from typing import Any

from PIL import Image


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def detect_language(text: str) -> str:
    """Return a coarse language tag used by the UI/ROS boundary."""

    return "zh" if _CJK_RE.search(str(text or "")) else "en"


def language_name(language: str) -> str:
    """Return the human-readable language name expected in prompts."""

    normalized = str(language or "auto").lower()
    if normalized in {"zh", "zh-cn", "chinese"}:
        return "Chinese"
    if normalized in {"en", "english"}:
        return "English"
    return str(language or "the user's language")


def prepare_user_message_for_agent(
    message: str,
    *,
    input_language: str = "auto",
    output_language: str = "auto",
    translate_boundary: bool = True,
) -> str:
    """Translate the external user message into the agent's English work language.

    This intentionally returns only the translated task, not an instruction
    wrapper, so the planner never sees the language policy as a task.
    """

    clean_message = str(message or "").strip()
    if not clean_message or not translate_boundary:
        return clean_message

    detected = detect_language(clean_message)
    source = normalize_language(input_language, fallback=detected)
    if source == "en":
        return clean_message

    return translate_text_for_agent(clean_message, source_language=source)


def normalize_language(language: str | None, *, fallback: str = "en") -> str:
    """Normalize UI/config language values to compact tags."""

    raw = str(language or "auto").strip().lower()
    if raw in {"auto", ""}:
        return fallback
    if raw in {"zh", "zh-cn", "chinese"}:
        return "zh"
    if raw in {"en", "english"}:
        return "en"
    return raw


def _run_text_llm(system_prompt: str, user_text: str) -> str:
    """Run one configured text LLM request and return stripped text."""

    from caragent_agent.config.config import config
    from caragent_agent.utils.llm_handler import UnifiedLLMClient

    model = (
        (config.get("llm_routing", {}) or {}).get("orchestrate")
        or config.get("llm_model")
        or "deepseek-chat"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return asyncio.run(UnifiedLLMClient().chat_completion(model, messages)).strip()


def translate_text_for_agent(text: str, *, source_language: str = "zh") -> str:
    """Translate a user request into concise English for planning/search."""

    clean_text = str(text or "").strip()
    if not clean_text:
        return ""
    try:
        translated = _run_text_llm(
            (
                "Translate the user's robot navigation request into concise English. "
                "Return only the translated request, with no explanation, no markdown, "
                "and no added policy text. Preserve room labels, visible text, ids, "
                "numbers, coordinates, and named objects exactly when possible."
            ),
            clean_text,
        )
        return translated or clean_text
    except Exception:
        return clean_text


def translate_text_for_user(text: str, *, target_language: str = "zh") -> str:
    """Translate an agent response for display, preserving technical payloads."""

    clean_text = str(text or "").strip()
    target = normalize_language(target_language, fallback="en")
    if not clean_text or target == "en":
        return clean_text
    try:
        translated = _run_text_llm(
            (
                f"Translate the assistant response into {language_name(target)}. "
                "Return only the translated response. Make Chinese output natural, concise, "
                "and conversational, like a robot assistant reporting to its user. Avoid stiff "
                "literal translation and phrases such as \"根据导航记忆\" when a simpler phrase "
                "like \"我去过\" is enough. Preserve JSON snippets, tool names, keyframe ids, "
                "coordinates, topic names, file paths, and error codes exactly."
            ),
            clean_text,
        )
        return translated or clean_text
    except Exception:
        return clean_text


def adapt_turn_result_language(
    turn_result: dict[str, Any],
    *,
    output_language: str = "auto",
    original_input_language: str = "auto",
) -> dict[str, Any]:
    """Translate final response fields for the UI/ROS boundary when requested."""

    target = normalize_language(
        output_language,
        fallback=normalize_language(original_input_language, fallback="en"),
    )
    if target == "en":
        return turn_result

    adapted = dict(turn_result)
    adapted["output_language"] = target
    adapted["language_adapted"] = True
    if adapted.get("turn_response_text"):
        adapted["turn_response_text"] = translate_text_for_user(
            str(adapted["turn_response_text"]),
            target_language=target,
        )
    elif isinstance(adapted.get("state"), dict):
        state_response = str(adapted["state"].get("user_facing_response") or "").strip()
        if state_response:
            adapted["turn_response_text"] = translate_text_for_user(
                state_response,
                target_language=target,
            )

    response_items = []
    for item in adapted.get("response_items", []) or []:
        if not isinstance(item, dict):
            response_items.append(item)
            continue
        new_item = dict(item)
        if new_item.get("response_text"):
            new_item["response_text"] = translate_text_for_user(
                str(new_item["response_text"]),
                target_language=target,
            )
        response_items.append(new_item)
    adapted["response_items"] = response_items
    return adapted


def image_to_data_url(image: Image.Image, *, image_format: str = "JPEG") -> str:
    """Encode a PIL image as a browser-friendly data URL."""

    buffered = BytesIO()
    image.convert("RGB").save(buffered, format=image_format)
    encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
    mime = "image/jpeg" if image_format.upper() in {"JPEG", "JPG"} else "image/png"
    return f"data:{mime};base64,{encoded}"


def image_from_data_url(data_url: str) -> Image.Image:
    """Decode a browser data URL or bare base64 payload into a PIL RGB image."""

    payload = str(data_url or "").strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    raw = base64.b64decode(payload)
    return Image.open(BytesIO(raw)).convert("RGB")


def describe_image_for_navigation(
    image: Image.Image,
    *,
    question: str | None = None,
) -> str:
    """Use the configured VLM to turn a target image into navigation-search text."""

    from caragent_agent.config.config import config
    from caragent_agent.utils.llm_handler import UnifiedLLMClient
    from caragent_agent.utils.llm_request_generator import (
        extract_answer_tags,
        vlm_analyse_on_each_kf_images_request_message,
    )

    prompt = question or (
        "Convert this target image into a compact scene-memory search query for an indoor robot. "
        "The image may be taken from a different angle, height, or crop than the robot keyframes, "
        "so do not require an exact visual match. "
        "Keep only stable navigation landmarks and spatial cues: object categories, distinctive colors, "
        "doors/corridors/junctions, signs or readable text, stairs, fire extinguishers, furniture clusters, "
        "and unusual wall/floor features. "
        "Ignore fragile details such as viewpoint, camera height, exact object counts, small decorations, "
        "lighting, reflections, or newly introduced objects unless they are the main landmark. "
        "Output 1-2 concise English sentences suitable for approximate best-candidate keyframe retrieval. "
        "No JSON, no markdown, no exhaustive description."
    )
    request = {
        "request_id": 0,
        "model": config.get("vlm_model_analyse_images", config.get("llm_model", "deepseek-chat")),
        "messages": vlm_analyse_on_each_kf_images_request_message(image, prompt),
    }
    results = asyncio.run(UnifiedLLMClient().batch_chat_completion([request]))
    response = results.get(0, {}).get(0)
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(str(response["error"]))
    if response is None:
        raise RuntimeError("VLM image description returned no response.")
    return extract_answer_tags(str(response)).strip()


def current_controller_image(controller: Any) -> Image.Image | None:
    """Read the latest controller image without blocking the navigation loop."""

    if controller is None or not hasattr(controller, "get_current_image"):
        return None
    image = controller.get_current_image()
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return None
