# part 4 - app/rag/foundation_model.py

import torch
import re
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FoundationModel:
    """
    LLM wrapper with:
      - history (multi-turn chat)
      - per-turn RAG context injection
      - tokenizer.apply_chat_template formatting
      - history trimming to stay within a token budget
    """

    def __init__(
        self,
        FOUND_MODEL_PATH: str,
        SYSTEM_PROMPT: str = (
            "You are a movie recommendation assistant. Ask clarifying questions when needed "
            "and recommend a small set of movies with brief reasons."
        ),
        TEMPERATURE: float = 0.7, # arbitraire
        MAX_NEW_TOKENS: int = 512,
        MAX_INPUT_TOKENS: int = 4096,
        DO_SAMPLE: bool = True,
    ):
        self.device = pick_device()
        self.system_prompt = SYSTEM_PROMPT
        self.max_input_tokens = MAX_INPUT_TOKENS

        self.model = AutoModelForCausalLM.from_pretrained(
            FOUND_MODEL_PATH,
            torch_dtype="auto",
            trust_remote_code=True,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            FOUND_MODEL_PATH,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )

        self.history = []  # [{"role": "user"/"assistant", "content": "..."}]

        self.num_parameters = self.model.num_parameters()
        print("Number of parameters in my model", "{:.2e}".format(self.num_parameters))

    def reset(self):
        self.history = []

    def _format_chat(self, messages):
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

    def _count_tokens(self, messages) -> int:
        text = self._format_chat(messages)
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _trim_history_to_fit(self, base_messages, turn_messages):
        hist = self.history.copy()

        def full(hist_local):
            return base_messages + hist_local + turn_messages

        while hist and self._count_tokens(full(hist)) > self.max_input_tokens:
            hist = hist[2:] if len(hist) >= 2 else hist[1:]
        return hist

    def chat(
        self,
        user_text: str,
        retrieved_context: Optional[List[str]] = None,
        store_context: bool = False,
    ) -> str:
        base = [{"role": "system", "content": self.system_prompt}]

        turn = []
        joined = None
        if retrieved_context:
            joined = "\n\n".join([f"[context {i+1}]\n{c}" for i, c in enumerate(retrieved_context)])
            turn.append({"role": "system", "content": joined})

        turn.append({"role": "user", "content": user_text})

        trimmed_hist = self._trim_history_to_fit(base, turn)
        messages = base + trimmed_hist + turn

        prompt_text = self._format_chat(messages)
        output = self.llm(prompt_text)
        assistant_text = output[0]["generated_text"]

        assistant_text_clean = re.sub(r"<think>.*?</think>", "", assistant_text, flags=re.DOTALL).strip()

        new_hist = trimmed_hist + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text_clean},
        ]
        if store_context and joined is not None:
            new_hist = trimmed_hist + [
                {"role": "system", "content": joined},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text_clean},
            ]

        self.history = new_hist
        return assistant_text_clean

    def generate_response(self, prompt: str) -> str:
        self.reset()
        return self.chat(prompt, retrieved_context=None)


def extract_response(text: str) -> str:
    return text

def short_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# f_model = FoundationModel("Qwen/Qwen3-0.6B")
# Example usage:
# # turn 1 with retrieved context (from your Annoy/CLIP retriever)
# answer1 = f_model.chat("I want a thriller with a strong female lead.", retrieved_context=ctx)
# # turn 2 (refines using conversation history)
# answer2 = f_model.chat("Prefer something after 2010, not too violent.", retrieved_context=ctx2)