  """Aegis-A11y Reasoning Agent Package

  Multi-modal semantic reasoning layer for educational content accessibility.
  """

  from .semantic_reasoner import SemanticReasoner
  from .schemas import ReasoningInput, ReasoningOutput

  __version__ = "0.1.0"
  __all__ = ["SemanticReasoner", "ReasoningInput", "ReasoningOutput"]
