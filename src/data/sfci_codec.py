"""
Alias exports for the component-centric VACT-Seq codec to keep legacy imports.
"""

from .sfci_codec_componentscentric import (
    build_component_vocab,
    components_to_sfci,
    sfci_to_components,
)

__all__ = ["components_to_sfci", "sfci_to_components", "build_component_vocab"]
