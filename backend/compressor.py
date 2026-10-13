"""
Prompt compression using Groq's LLM API.

PURPOSE:
Make user prompts more concise and precise while preserving all important information.
Removes filler, informal language, tangents, and redundancy.

INPUTS:
Prompt text, Groq client instance

OUTPUTS:
Compressed prompt text, token estimates for original and compressed versions
"""

from openai import OpenAIError

from config import GROQ_MODEL

# System prompt for compression
COMPRESSION_SYSTEM_PROMPT = """You are a prompt compression system. Make the user's message more concise and precise while preserving all important information. Remove filler, informal language, tangents, and redundancy. Return ONLY the compressed message, nothing else."""


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using rough heuristic.

    Approximation: average word takes 1.3 tokens (based on typical tokenizers).
    This is a rough estimate suitable for display purposes.

    Args:
        text: Text to estimate tokens for

    Returns:
        Approximate token count
    """
    words = len(text.split())
    # Multiply by 1.3 as rough approximation
    estimated_tokens = int(words * 1.3)
    return max(1, estimated_tokens)  # Ensure at least 1 token


def compress(prompt: str, groq_client) -> dict[str, any]:
    """
    Compress a prompt to be more concise and precise.

    Algorithm:
    1. Call Groq API with compression system prompt
    2. Return compressed text
    3. Estimate tokens for both original and compressed versions
    4. Calculate compression ratio

    Args:
        prompt: Prompt text to compress
        groq_client: OpenAI client configured for Groq API

    Returns:
        Dictionary with keys:
        - compressed (str): Compressed prompt text
        - original_tokens (int): Estimated tokens in original
        - compressed_tokens (int): Estimated tokens in compressed
        - ratio (float): Compression ratio (compressed / original)

    Raises:
        Exception: If Groq API call fails
    """
    # Estimate original token count
    original_tokens = _estimate_tokens(prompt)

    # Call Groq API for compression
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": COMPRESSION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=2048  # Allow room for compressed output
        )
    except OpenAIError as e:
        # If compression fails, return the original prompt unchanged
        print(f"Error calling Groq API for compression: {e}")
        return {
            "compressed": prompt,
            "original_tokens": original_tokens,
            "compressed_tokens": original_tokens,
            "ratio": 1.0
        }

    # Extract compressed text
    compressed_text = response.choices[0].message.content.strip()

    # Estimate compressed token count
    compressed_tokens = _estimate_tokens(compressed_text)

    # Calculate compression ratio
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

    return {
        "compressed": compressed_text,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "ratio": ratio
    }
