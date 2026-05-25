
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
