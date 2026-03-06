import random
from typing import Generator, List, Optional
import numpy as np

from inference_perf.apis.base import InferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .base import DataGenerator


class ChatDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for ChatDataGenerator but was not initialized.")

        # Initialize vocab_size
        hf_tokenizer = self.tokenizer.get_tokenizer()
        if hasattr(hf_tokenizer, "vocab_size") and hf_tokenizer.vocab_size is not None:
            self.vocab_size: int = hf_tokenizer.vocab_size
        elif hasattr(hf_tokenizer, "get_vocab") and callable(hf_tokenizer.get_vocab):
            self.vocab_size = len(hf_tokenizer.get_vocab())
        else:
            try:
                self.vocab_size = len(hf_tokenizer)
            except TypeError as e:
                raise ValueError(
                    "Tokenizer does not have a 'vocab_size' attribute, 'get_vocab()' method, "
                    "or support len() for vocabulary size. Cannot use random token generation."
                ) from e
        if self.vocab_size <= 0:
            raise ValueError(f"Tokenizer vocabulary size must be positive, got {self.vocab_size}.")

        if self.chat is None:
            raise ValueError("Chat config is required for ChatDataGenerator")

        self.num_chats: int = self.chat.num_chats
        self.num_messages_per_chat: int = self.chat.num_messages_per_chat
        self.message_len: int = self.chat.message_len
        self.output_len: int = self.chat.output_len

        self.prompts: List[str] = []
        self._generate_prompts()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return True

    def get_request(self, n: int) -> InferenceAPIData:
        i = n % len(self.prompts)
        return CompletionAPIData(prompt=self.prompts[i], max_tokens=self.output_len)

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if not self.prompts:
            return

        i = 0
        while True:
            yield CompletionAPIData(prompt=self.prompts[i], max_tokens=self.output_len)
            i = (i + 1) % len(self.prompts)

    def _generate_random_token_ids(self, length: int) -> List[int]:
        """Generates a list of random token IDs of a specified length."""
        if length == 0:
            return []
        # np.random.randint's high parameter is exclusive
        return np.random.randint(0, self.vocab_size, size=length, dtype=np.int64).tolist()  # type: ignore[no-any-return]

    def _generate_prompts(self) -> None:
        """Pre-generates all prompts based on the configuration."""
        if self.tokenizer is None:
            # This check is defensive; __init__ should have already validated this.
            raise ValueError("Tokenizer is not available for generating prompts.")

        hf_tokenizer = self.tokenizer.get_tokenizer()

        for _ in range(self.num_chats):
            full_chat_token_ids = self._generate_random_token_ids(self.num_messages_per_chat * self.message_len)
            full_chat_text = hf_tokenizer.decode(full_chat_token_ids, skip_special_tokens=True)

            for j in range(1, self.num_messages_per_chat + 1):
                prefix_length = (len(full_chat_text) * j // self.num_messages_per_chat)
                chat_prefix = full_chat_text[:prefix_length]
                self.prompts.append(chat_prefix)

        random.shuffle(self.prompts)
