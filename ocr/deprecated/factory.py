"""
OCR Engine Factory - Creates and configures OCR engines.

Provides a simple interface to create OCR engines by name,
with automatic fallback and configuration handling.
"""

from typing import Optional, List, Dict, Any
from .base import OCREngine
from .easyocr_engine import EasyOCREngine
from .paddleocr_engine import PaddleOCREngine
from .mmocr_engine import MMOCREngine
from .parseq_engine import PARSeqEngine
from .ensemble_engine import EnsembleOCREngine


# Registry of available engines
ENGINE_REGISTRY = {
    'easyocr': EasyOCREngine,
    'paddleocr': PaddleOCREngine,
    'mmocr': MMOCREngine,
    'parseq': PARSeqEngine,
    'ensemble': EnsembleOCREngine,
}


def get_available_engines() -> List[str]:
    """
    Get list of available OCR engine names.

    Returns:
        List of engine names that can be created
    """
    return list(ENGINE_REGISTRY.keys())


def create_ocr_engine(
    engine_name: str,
    config: Dict[str, Any] = None,
    auto_init: bool = True
) -> Optional[OCREngine]:
    """
    Create an OCR engine by name.

    Args:
        engine_name: Name of engine ('easyocr', 'paddleocr', 'mmocr', 'parseq', 'ensemble')
        config: Optional configuration dict passed to engine constructor
        auto_init: Whether to automatically initialize the engine

    Returns:
        OCREngine instance or None if creation failed

    Examples:
        # Simple creation
        engine = create_ocr_engine('easyocr')

        # With config
        engine = create_ocr_engine('parseq', {'device': 'cuda'})

        # Ensemble with specific engines
        engine = create_ocr_engine('ensemble', {
            'engines': ['easyocr', 'parseq'],
            'voting_strategy': 'confidence_weighted'
        })
    """
    config = config or {}

    # Handle ensemble specially
    if engine_name == 'ensemble':
        return _create_ensemble_engine(config, auto_init)

    # Standard engine creation
    if engine_name not in ENGINE_REGISTRY:
        print(f"Unknown engine: {engine_name}")
        print(f"Available engines: {list(ENGINE_REGISTRY.keys())}")
        return None

    engine_class = ENGINE_REGISTRY[engine_name]

    try:
        engine = engine_class(**config)

        if auto_init:
            if not engine.initialize():
                print(f"Failed to initialize {engine_name}")
                return None

        return engine

    except Exception as e:
        print(f"Error creating {engine_name}: {e}")
        return None


def _create_ensemble_engine(
    config: Dict[str, Any],
    auto_init: bool
) -> Optional[EnsembleOCREngine]:
    """Create and configure an ensemble engine."""
    # Get engine names from config
    engine_names = config.pop('engines', ['easyocr', 'paddleocr'])
    voting_strategy = config.pop('voting_strategy', 'confidence_weighted')
    min_agreement = config.pop('min_agreement', 2)

    # Create individual engines
    engines = []
    for name in engine_names:
        engine_config = config.get(f'{name}_config', {})
        engine = create_ocr_engine(name, engine_config, auto_init=False)
        if engine:
            engines.append(engine)

    if not engines:
        print("Failed to create any engines for ensemble")
        return None

    # Create ensemble
    ensemble = EnsembleOCREngine(
        engines=engines,
        voting_strategy=voting_strategy,
        min_agreement=min_agreement
    )

    if auto_init:
        if not ensemble.initialize():
            print("Failed to initialize ensemble engine")
            return None

    return ensemble


def create_best_available_engine(
    preferred_order: List[str] = None,
    config: Dict[str, Any] = None
) -> Optional[OCREngine]:
    """
    Create the best available OCR engine by trying in order.

    Tries each engine in the preferred order, returning the first
    one that initializes successfully.

    Args:
        preferred_order: List of engine names to try (default: parseq, mmocr, easyocr, paddleocr)
        config: Optional configuration dict

    Returns:
        Best available OCREngine or None
    """
    if preferred_order is None:
        preferred_order = ['parseq', 'mmocr', 'easyocr', 'paddleocr']

    config = config or {}

    for engine_name in preferred_order:
        engine_config = config.get(engine_name, {})
        engine = create_ocr_engine(engine_name, engine_config, auto_init=True)
        if engine and engine.is_initialized:
            print(f"Using {engine_name} as OCR engine")
            return engine

    print("No OCR engine available!")
    return None


def create_recommended_ensemble(
    gpu: bool = False,
    accuracy_priority: bool = True
) -> Optional[EnsembleOCREngine]:
    """
    Create a recommended ensemble configuration.

    Args:
        gpu: Whether GPU is available
        accuracy_priority: If True, prioritize accuracy over speed

    Returns:
        Configured EnsembleOCREngine
    """
    if accuracy_priority:
        # High accuracy ensemble
        if gpu:
            engines = ['parseq', 'mmocr', 'easyocr']
        else:
            engines = ['easyocr', 'paddleocr']
    else:
        # Speed-focused ensemble
        engines = ['easyocr', 'paddleocr']

    config = {
        'engines': engines,
        'voting_strategy': 'confidence_weighted',
        'min_agreement': 2
    }

    if gpu:
        config['parseq_config'] = {'device': 'cuda'}
        config['mmocr_config'] = {'device': 'cuda:0'}

    return create_ocr_engine('ensemble', config)
