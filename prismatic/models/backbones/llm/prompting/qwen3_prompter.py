"""
qwen3_prompter.py

Prompt builder for Qwen3 chat models.
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder


class Qwen3PromptBuilder(PromptBuilder):
    """
    Prompt builder for Qwen3 chat models.
    
    Qwen3 uses the ChatML format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>
    """

    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        
        # Qwen3 special tokens
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        
        # Default system prompt for VLA task
        if system_prompt is None:
            self.system_prompt = (
                "You are a helpful assistant controlling a robot. "
                "Given an image of the robot's current view and a task instruction, "
                "predict the appropriate action."
            )
        
        # Role-specific wrap functions
        self.wrap_system = lambda msg: f"{self.im_start}system\n{msg}{self.im_end}\n"
        self.wrap_human = lambda msg: f"{self.im_start}user\n{msg}{self.im_end}\n{self.im_start}assistant\n"
        self.wrap_gpt = lambda msg: f"{msg}{self.im_end}\n"
        
        # Build initial prompt with system message
        self.prompt = self.wrap_system(self.system_prompt)
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        """Add a turn to the conversation."""
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        
        # Remove <image> placeholder as it's handled separately
        message = message.replace("<image>", "").strip()
        
        if role == "human":
            wrapped_message = self.wrap_human(message)
        else:
            wrapped_message = self.wrap_gpt(message)
        
        self.prompt += wrapped_message
        self.turn_count += 1
        
        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        """Get prompt without committing the turn."""
        prompt_copy = str(self.prompt)
        message = message.replace("<image>", "").strip()
        prompt_copy += self.wrap_human(message)
        return prompt_copy

    def get_prompt(self) -> str:
        """Get the current prompt."""
        return self.prompt
