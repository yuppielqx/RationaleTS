import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import os

from pandas.core.config_init import pc_nb_repr_h_doc
from utils import encode_image_to_base64, call_qwen_api

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, model_name, dataset_name, image_url_format='string'):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.prompts = self._load_prompts()
        self.image_url_format = image_url_format

    def _load_prompts(self):
        with open("prompts.json", 'r') as f:
            all_prompts = json.load(f)
            return all_prompts.get(self.dataset_name, {})

    def _create_image_content(self, image_b64_string: str) -> Dict[str, Any]:
        """根据指定的格式创建图片内容部分。"""
        image_url_content = f"data:image/jpeg;base64,{image_b64_string}"
        if self.image_url_format == 'dict':
            return {"type": "image_url", "image_url": {"url": image_url_content}}
        # 默认为 'string' 格式
        return {"type": "image_url", "image_url": image_url_content}

    @abstractmethod
    def execute(self, **kwargs):
        """The main method to run the agent's logic."""
        pass

class PredictionAgent(BaseAgent):
    """Agent responsible for making the initial weather prediction."""
    def execute(self, image_path, text_content):
        prompt_config = self.prompts['prediction_agent']
        system_prompt = prompt_config['system_prompt']
        user_prompt_part_1 = prompt_config['user_prompt_part_1'].format(text_content=text_content)
        user_prompt_part_2 = prompt_config['user_prompt_part_2']

        content_parts = []
        content_parts.append({"type": "text", "text": user_prompt_part_1})
        image_b64 = encode_image_to_base64(image_path)
        content_parts.append(self._create_image_content(image_b64))
        content_parts.append({"type": "text", "text": user_prompt_part_2})
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts}
        ]

        return call_qwen_api(messages, self.model_name)


class EvaluatorAgent(BaseAgent):
    """
    金融评估代理。
    根据真实标签，评估预测的准确性，并提供一个“黄金标准”的推理过程。
    """
    def execute(self, true_label: int, true_label_meaning: str, llm_prediction: int, reasoning: str, image_path: str, **kwargs) -> str:
        """
        构建评估prompt并调用VLM。

        Args:
            true_label (int): 真实的S&P 500变动标签 (0, 1, 2)。
            true_label_meaning (str): 真实标签的文本含义。
            llm_prediction (int): VLM预测的标签。
            reasoning (str): VLM的初步推理过程。
            image_path (str): 图表的路径。

        Returns:
            str: VLM返回的评估和优化推理。
        """
        try:
            prompt_config = self.prompts['evaluator_agent']
        except KeyError:
            raise ValueError(f"Could not find prompts for 'evaluator_agent' in prompts.json for dataset '{self.dataset_name}'")

        system_prompt = prompt_config['system_prompt']
        user_prompt = prompt_config['user_prompt'].format(
            true_label_meaning=true_label_meaning
        )

        image_b64 = encode_image_to_base64(image_path)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, self._create_image_content(image_b64)]}
        ]

        return call_qwen_api(messages, self.model_name, max_tokens=3000)


class AnalysisAgent(BaseAgent):
    """
    分析代理。
    对新的图表进行初步分析，生成简短的文本摘要，用于后续的检索查询。
    """
    def execute(self, image_path: str, **kwargs) -> str:
        """
        构建分析prompt并调用VLM。

        Args:
            image_path (str): 图表的路径。

        Returns:
            str: VLM返回的简短分析文本。
        """
        try:
            prompt_config = self.prompts['analysis_agent']
        except KeyError:
            raise ValueError(f"Could not find prompts for 'analysis_agent' in prompts.json for dataset '{self.dataset_name}'")

        system_prompt = prompt_config['system_prompt']
        user_prompt = prompt_config['user_prompt']

        image_b64 = encode_image_to_base64(image_path)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, self._create_image_content(image_b64)]}
        ]

        return call_qwen_api(messages, self.model_name, max_tokens=512)


class RAGPredictionAgent(BaseAgent):
    """
    基于检索增强生成的金融预测代理。
    """
    def execute(self, image_path: str, examples: List[Dict], **kwargs) -> str:
        """
        构建包含相似案例的prompt并调用VLM进行最终预测。

        Args:
            image_path (str): 测试图表的路径。
            examples (List[Dict]): 检索到的相似案例列表。

        Returns:
            str: VLM返回的包含预测和推理的JSON字符串。
        """
        try:
            prompt_config = self.prompts['rag_prediction_agent']
        except KeyError:
            raise ValueError(f"Could not find prompts for 'rag_prediction_agent' in prompts.json for dataset '{self.dataset_name}'")

        system_prompt = prompt_config['system_prompt']
        user_prompt = prompt_config['user_prompt'].format(examples="\n\n".join([f"### Example {i+1} (ID: {ex['id']}):\n{ex['reasoning_path']}" for i, ex in enumerate(examples)]))
        image_b64 = encode_image_to_base64(image_path)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, self._create_image_content(image_b64)]}
        ]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)


class ImageOnlyPredictionAgent(BaseAgent):
    """
    Based ONLY on the image, makes a prediction.
    """
    def execute(self, image_path: str, **kwargs) -> str:
        """
        Builds a prompt using only the image and calls the VLM.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: VLM response containing prediction and reasoning in JSON format.
        """
        try:
            prompt_config = self.prompts['image_only_prediction_agent']
        except KeyError:
            raise ValueError(f"Could not find prompts for 'image_only_prediction_agent' in prompts.json for dataset '{self.dataset_name}'")

        system_prompt = prompt_config['system_prompt']
        user_prompt = prompt_config['user_prompt']
        image_b64 = encode_image_to_base64(image_path)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, self._create_image_content(image_b64)]}
        ]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)


class ImageZeroCOTPredictionAgent(BaseAgent):
    """
    Based ONLY on the image, makes a prediction.
    """
    def execute(self, image_path: str, **kwargs) -> str:
        """
        Builds a prompt using only the image and calls the VLM.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: VLM response containing prediction and reasoning in JSON format.
        """
        try:
            prompt_config = self.prompts['image_zero_cot_prediction_agent']
        except KeyError:
            raise ValueError(f"Could not find prompts for 'image_only_prediction_agent' in prompts.json for dataset '{self.dataset_name}'")

        system_prompt = prompt_config['system_prompt']
        user_prompt = prompt_config['user_prompt']
        image_b64 = encode_image_to_base64(image_path)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, self._create_image_content(image_b64)]}
        ]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)


class ImageFewShotPredictionAgent(BaseAgent):
    """Agent that performs few-shot in-context learning with images."""

    def __init__(self, model_name: str, dataset_name: str, image_url_format: str = 'path'):
        super().__init__(model_name, dataset_name, image_url_format)
        self.agent_name = 'image_few_shot_prediction_agent'

    def execute(self, test_image_path: str, examples: List[Dict], **kwargs):
        """
        Constructs a multi-image prompt for few-shot learning and calls the VLM.

        Args:
            test_image_path (str): Path to the test image.
            examples (List[Dict]): A list of few-shot examples, where each is a dict
                                   with 'image_path' and 'label_meaning'.

        Returns:
            A tuple of (VLM response string, usage dictionary).
        """
        try:
            prompt_config = self.prompts[self.agent_name]
        except KeyError:
            raise ValueError(
                f"Could not find prompts for '{self.agent_name}' in prompts.json for dataset '{self.dataset_name}'")

        system_prompt = prompt_config['system_prompt']
        user_prompt_intro = prompt_config['user_prompt_intro']
        user_prompt_example_template = prompt_config['user_prompt_example_template']
        user_prompt_task = prompt_config['user_prompt_task']

        # --- 构建多模态消息内容 ---
        user_content = [{"type": "text", "text": user_prompt_intro}]

        # 添加 few-shot 样本 (文本后跟图片)
        for i, ex in enumerate(examples):
            example_text = user_prompt_example_template.format(i=i + 1, label_meaning=ex['label_meaning'])
            user_content.append({"type": "text", "text": example_text})
            image_b64 = encode_image_to_base64(ex['image_path'])
            if image_b64:
                user_content.append(self._create_image_content(image_b64))

        # 添加最终任务提示和测试图片
        user_content.append({"type": "text", "text": user_prompt_task})
        test_image_b64 = encode_image_to_base64(test_image_path)
        user_content.append(self._create_image_content(test_image_b64))

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)


class LLMDirectPredictionAgent(BaseAgent):
    """
    Based ONLY on the text data, makes a prediction.
    """
    def execute(self, text_data: str, **kwargs) -> str:
        """
        Builds a prompt using only the text data and calls the LLM.

        Args:
            text_data (str): The time-series data formatted as a string.

        Returns:
            str: LLM response containing prediction and reasoning in JSON format.
        """
        try:
            prompt_config = self.prompts['llm_direct_prediction_agent']
        except KeyError:
            raise ValueError(f"Could not find prompts for 'llm_direct_prediction_agent' in prompts.json for dataset '{self.dataset_name}'")

        system_prompt = prompt_config['system_prompt']
        user_prompt = prompt_config['user_prompt'].format(time_series_data=text_data)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)

class LLMCoTPredictionAgent(BaseAgent):
    """
    Based ONLY on the text data, makes a prediction.
    """
    def execute(self, text_data: str, **kwargs) -> str:
        """
        Builds a prompt using only the text data and calls the LLM.

        Args:
            text_data (str): The time-series data formatted as a string.

        Returns:
            str: LLM response containing prediction and reasoning in JSON format.
        """
        try:
            prompt_config = self.prompts['llm_cot_prediction_agent']
        except KeyError:
            raise ValueError(f"Could not find prompts for 'llm_cot_prediction_agent' in prompts.json for dataset '{self.dataset_name}'")

        system_prompt = prompt_config['system_prompt']
        user_prompt = prompt_config['user_prompt'].format(time_series_data=text_data)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)

class LLMFewShotPredictionAgent(BaseAgent):
    """Agent that performs few-shot in-context learning with text time-series data."""

    def __init__(self, model_name: str, dataset_name: str):
        super().__init__(model_name, dataset_name)
        self.agent_name = 'llm_few_shot_prediction_agent'

    def execute(self, test_text_data: str, examples: List[Dict], **kwargs):
        """
        Constructs a prompt with few-shot text examples and calls the LLM.
        """
        try:
            prompt_config = self.prompts[self.agent_name]
        except KeyError:
            raise ValueError(
                f"Could not find prompts for '{self.agent_name}' in prompts.json for dataset '{self.dataset_name}'")

        system_prompt = prompt_config['system_prompt']
        user_prompt_intro = prompt_config['user_prompt_intro']
        user_prompt_example_template = prompt_config['user_prompt_example_template']
        user_prompt_task = prompt_config['user_prompt_task']

        user_content_str = user_prompt_intro

        for i, ex in enumerate(examples):
            example_str = user_prompt_example_template.format(
                i=i + 1,
                time_series_data=ex['text_data'],
                label_meaning=ex['label_meaning']
            )
            user_content_str += example_str

        task_str = user_prompt_task.format(time_series_data=test_text_data)
        user_content_str += task_str

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_content_str}]}
        ]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)


class RAGWithLabelsAgent(BaseAgent):
    """Agent that uses reasoning paths and their corresponding labels for prediction."""

    def __init__(self, model_name: str, dataset_name: str, image_url_format: str = 'path'):
        super().__init__(model_name, dataset_name, image_url_format)
        self.agent_name = 'rag_with_labels_agent'

    def execute(self, image_path: str, examples: List[Dict], LABEL_MEANINGS: Dict[int, str], **kwargs):
        prompt_config = self.prompts[self.agent_name]
        system_prompt = prompt_config['system_prompt']

        # Format examples to include reasoning and labels
        example_texts = []
        for i, ex in enumerate(examples):
            label_meaning = LABEL_MEANINGS.get(ex['true_label'], "Unknown")
            example_texts.append(
                f"### Example {i + 1} (ID: {ex['id']}):\nReasoning Path: {ex['reasoning_path']}\nActual Outcome: {label_meaning}"
            )
        examples_str = "\n\n".join(example_texts)

        user_prompt = prompt_config['user_prompt'].format(examples=examples_str)

        image_b64 = encode_image_to_base64(image_path)
        if not image_b64:
            return "Failed to encode image.", {}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, self._create_image_content(image_b64)]}
        ]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)


class RAGWithImagesAgent(BaseAgent):
    """Agent that uses reasoning paths and their corresponding images for prediction."""

    def __init__(self, model_name: str, dataset_name: str, image_url_format: str = 'path'):
        super().__init__(model_name, dataset_name, image_url_format)
        self.agent_name = 'rag_with_images_agent'

    def execute(self, image_path: str, examples: List[Dict], **kwargs):
        prompt_config = self.prompts[self.agent_name]
        system_prompt = prompt_config['system_prompt']
        user_prompt_intro = prompt_config['user_prompt_intro']
        user_prompt_example_template = prompt_config['user_prompt_example_template']
        user_prompt_task = prompt_config['user_prompt_task']

        user_content = [{"type": "text", "text": user_prompt_intro}]

        for i, ex in enumerate(examples):
            example_text = user_prompt_example_template.format(i=i + 1, reasoning_path=ex['reasoning_path'])
            user_content.append({"type": "text", "text": example_text})
            example_image_path = f"dataset/{self.dataset_name}/images/{ex['id']}.png"
            image_b64 = encode_image_to_base64(example_image_path)
            if image_b64:
                user_content.append(self._create_image_content(image_b64))

        user_content.append({"type": "text", "text": user_prompt_task})
        test_image_b64 = encode_image_to_base64(image_path)
        user_content.append(self._create_image_content(test_image_b64))

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)


class RAGWithImagesAndLabelsAgent(BaseAgent):
    """Agent that uses reasoning, images, and labels for prediction."""

    def __init__(self, model_name: str, dataset_name: str, image_url_format: str = 'path'):
        super().__init__(model_name, dataset_name, image_url_format)
        self.agent_name = 'rag_with_images_and_labels_agent'

    def execute(self, image_path: str, examples: List[Dict], LABEL_MEANINGS: Dict[int, str], **kwargs):
        prompt_config = self.prompts[self.agent_name]
        system_prompt = prompt_config['system_prompt']
        user_prompt_intro = prompt_config['user_prompt_intro']
        user_prompt_example_template = prompt_config['user_prompt_example_template']
        user_prompt_task = prompt_config['user_prompt_task']

        user_content = [{"type": "text", "text": user_prompt_intro}]

        for i, ex in enumerate(examples):
            label_meaning = LABEL_MEANINGS.get(ex['true_label'], "Unknown")
            example_text = user_prompt_example_template.format(
                i=i + 1,
                reasoning_path=ex['reasoning_path'],
                label_meaning=label_meaning
            )
            user_content.append({"type": "text", "text": example_text})
            example_image_path = f"dataset/{self.dataset_name}/images/{ex['id']}.png"
            image_b64 = encode_image_to_base64(example_image_path)
            if image_b64:
                user_content.append(self._create_image_content(image_b64))

        user_content.append({"type": "text", "text": user_prompt_task})
        test_image_b64 = encode_image_to_base64(image_path)
        user_content.append(self._create_image_content(test_image_b64))

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

        return call_qwen_api(messages, self.model_name, max_tokens=2048)