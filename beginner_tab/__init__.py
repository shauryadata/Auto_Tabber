"""beginner_tab — Convert audio to beginner-friendly guitar tabs."""

from .audio_loader import AudioLoader, AudioLoadError
from .pitch_tracker import PitchTracker, BasicPitchTracker, PitchDetectionError
from .tab_simplifier import TabSimplifier
from .fretboard_mapper import FretboardMapper
from .tab_renderer import TabRenderer
from .tab_storage import TabStorage, TabStorageError

__all__ = [
    "AudioLoader",
    "AudioLoadError",
    "PitchTracker",
    "BasicPitchTracker",
    "PitchDetectionError",
    "TabSimplifier",
    "FretboardMapper",
    "TabRenderer",
    "TabStorage",
    "TabStorageError",
]
__version__ = "0.2.0"
