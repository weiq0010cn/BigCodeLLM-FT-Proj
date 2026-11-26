# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 2 Acceptable Use License.

import fire

from llama import Llama
from typing import List


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    max_gen_len: int = 64,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions = [
        [
            {
                "role": "user",
                "content": "what is the recipe of mayonnaise in one word?",
            }
        ],
        [
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?",
            }
        ],
        [
            {
                "role": "user",
                "content": "what is the greatest album of all time?",
            }
        ],
    ]
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for instruction, result in zip(instructions, results):
        for dialog_turn in instruction:
            print(f"role: {dialog_turn['role']}, content: {dialog_turn['content']}")
        print(
            f"> {result['generation']['role']}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
