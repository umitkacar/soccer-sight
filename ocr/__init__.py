"""
OCR Engine Module for Jersey Number Recognition.

Supports:
- EasyOCR: Fast, CPU-friendly, ~52% accuracy
- SoccerNet PARSeq: SOTA, ~92% accuracy (requires CUDA)

Usage:
    from ocr import create_ocr_engine

    # EasyOCR (fast, CPU)
    engine = create_ocr_engine('easyocr')

    # SoccerNet (SOTA, GPU preferred)
    engine = create_ocr_engine('soccernet')

    result = engine.recognize(image)
    print(f"Jersey: {result.text}, Confidence: {result.confidence}")
"""

from .base import OCREngine, OCRResult
from .soccernet_ocr import SoccerNetOCR, create_soccernet_ocr
from .temporal_filter import TemporalConsistencyFilter


def create_easyocr_engine(**kwargs):
    """Create EasyOCR engine (fast, CPU-friendly)."""
    from .deprecated.easyocr_engine import EasyOCREngine
    engine = EasyOCREngine(**kwargs)
    if engine.initialize():
        return engine
    return None


def create_ocr_engine(engine_type: str = 'easyocr', **kwargs):
    """
    Create OCR engine by type.

    Args:
        engine_type: 'easyocr' (fast) or 'soccernet' (SOTA)
        **kwargs: Additional arguments

    Returns:
        OCREngine instance
    """
    if engine_type == 'easyocr':
        return create_easyocr_engine(**kwargs)
    elif engine_type == 'soccernet':
        return create_soccernet_ocr(**kwargs)
    else:
        print(f"Unknown OCR engine: {engine_type}, falling back to easyocr")
        return create_easyocr_engine(**kwargs)


def create_best_available_engine(preferred: str = 'easyocr', **kwargs):
    """
    Create the best available OCR engine.

    Args:
        preferred: Preferred engine type ('easyocr' or 'soccernet')

    Returns:
        OCREngine instance
    """
    if preferred == 'easyocr':
        engine = create_easyocr_engine(**kwargs)
        if engine and engine.is_initialized:
            return engine
        # Fallback to soccernet
        return create_soccernet_ocr(**kwargs)
    else:
        engine = create_soccernet_ocr(**kwargs)
        if engine and engine.is_initialized:
            return engine
        # Fallback to easyocr
        return create_easyocr_engine(**kwargs)


def get_available_engines():
    """Get list of available OCR engines."""
    return ['easyocr', 'soccernet']


__all__ = [
    'OCREngine',
    'OCRResult',
    'SoccerNetOCR',
    'TemporalConsistencyFilter',
    'create_ocr_engine',
    'create_easyocr_engine',
    'create_soccernet_ocr',
    'create_best_available_engine',
    'get_available_engines',
]
