import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLMProvider:
    def __init__(self):
        # Dictionaries to hold the models in VRAM so they only load once
        self.models = {}
        self.tokenizers = {}

    def preload_models(self, models_to_load: list):
        """Forces models into VRAM upfront to prevent chat latency."""
        logger.info(f"Preloading {len(models_to_load)} models into VRAM. Please wait...")
        for model_id in models_to_load:
            self._load_model(model_id)
        logger.info("All models loaded successfully! VRAM allocated.")

    def _load_model(self, model_id: str):
        """Downloads (if needed) and loads the Hugging Face model into VRAM in 4-bit."""
        if model_id not in self.models:
            logger.info(f"Loading {model_id} via Hugging Face. This may take a while on first run...")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            # Gemma 4 uses a Multimodal architecture, DeepSeek uses standard Causal
            if "gemma-4" in model_id.lower():
                from transformers import AutoModelForMultimodalLM, AutoProcessor
                self.tokenizers[model_id] = AutoProcessor.from_pretrained(model_id)
                self.models[model_id] = AutoModelForMultimodalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
                self.models[model_id] = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
        return self.models[model_id], self.tokenizers[model_id]

    def get_logit_ratio(self, prompt: str, model: str) -> float:
        """Calculates the P(High) / (P(High) + P(Low)) ratio for a given prompt."""
        hf_model, tokenizer = self._load_model(model)

        # Convert text to tensor without adding special tokens to the end
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)

        with torch.no_grad():
            outputs = hf_model(**inputs)
            # Isolate the logits for the very next token prediction
            next_token_logits = outputs.logits[0, -1, :]

        # Get the token IDs for the target words
        high_token = tokenizer.encode("High", add_special_tokens=False)[-1]
        low_token = tokenizer.encode("Low", add_special_tokens=False)[-1]

        # Convert raw logits to probabilities
        probs = torch.softmax(next_token_logits, dim=-1)
        p_high = probs[high_token].item()
        p_low = probs[low_token].item()

        # Failsafe if neither token is statistically probable
        if (p_high + p_low) == 0:
            return 0.5

        return p_high / (p_high + p_low)

    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 model: str,
                 temperature: float = 0.7,
                 json_mode: bool = False) -> str:

        hf_model, tokenizer = self._load_model(model)

        # Format the chat using Hugging Face templates
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 1. Format the chat into a single raw text string
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # 2. Convert the text string into PyTorch tensors
        inputs = tokenizer(
            text=formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(hf_model.device)

        logger.info(f"Generating tokens with {model}...")

        # 3. Run local inference using the unpacked inputs dictionary
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=temperature > 0.0
            )

        # 4. Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return response.strip()
