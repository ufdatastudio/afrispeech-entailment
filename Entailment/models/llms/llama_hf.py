from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaHF:
    """
    Minimal HuggingFace Transformers wrapper for chat-style generation with Llama.
    """
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate text given chat-style messages.
        """
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback simple formatting
            system_msgs = "\n".join(m["content"] for m in messages if m["role"] == "system")
            user_msgs = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
            prompt_text = f"System: {system_msgs}\n\nUser: {user_msgs}\n\nAssistant:"

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()


def extract_json(text: str) -> Optional[Dict]:
    """
    Extract first JSON object from text, if present.
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    import json
    try:
        return json.loads(match.group(0))
    except Exception:
        return None











