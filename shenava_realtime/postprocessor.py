"""
Text post-processing for ASR transcriptions
"""

import re

from .config import ASRConfig


class PostProcessor:
    """
    Post-processes raw ASR output text

    Features:
    - Inverse Text Normalization (ITN) for Persian digits
    - Repetition removal
    - Whitespace cleanup
    """

    # Persian digits mapped to ASCII equivalents
    PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

    def __init__(self, config: ASRConfig):
        """
        Initialize post-processor

        Args:
            config: ASR configuration
        """
        self.config = config

    def process(self, text: str) -> str:
        """
        Process raw ASR text

        Args:
            text: Raw transcription text

        Returns:
            Cleaned text
        """
        if not text:
            return text

        text = text.strip()

        if self.config.remove_repetitions:
            text = self._remove_repeated_words(text)

        if self.config.apply_itn:
            text = self._apply_itn(text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _remove_repeated_words(self, text: str) -> str:
        """
        Remove consecutive repeated words (common ASR artifact)

        Args:
            text: Input text

        Returns:
            Text with consecutive duplicates collapsed
        """
        words = text.split()
        result = []

        for word in words:
            if not result or result[-1] != word:
                result.append(word)

        return " ".join(result)

    def _apply_itn(self, text: str) -> str:
        """
        Apply inverse text normalization

        Converts Persian digits to ASCII digits and fixes punctuation spacing

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Convert Persian/Arabic digits to ASCII
        text = text.translate(self.PERSIAN_DIGITS)
        text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

        # Fix spacing before punctuation
        text = re.sub(r"\s+([.,!?؟،؛:])", r"\1", text)

        return text
