"""Pitch detection backends — pYIN (monophonic) and Basic-pitch (polyphonic).

Two tracker classes share the same ``NoteEvent`` output type so the rest of
the pipeline is backend-agnostic.

PitchTracker (pYIN)
    Classic probabilistic YIN algorithm via librosa.  Single note at a time.
    Fast, no model download.  Works well on isolated melody recordings.
    Optional HPSS pre-processing removes drum hits before detection.

BasicPitchTracker (Spotify Basic-pitch)
    Convolutional neural network trained on real music.  Returns *polyphonic*
    note events — multiple simultaneous notes (chords) are supported.
    Requires tensorflow (installed with basic-pitch).  First call loads the
    model (~200 MB) from disk; subsequent calls are fast.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Shared type alias
# (time_in_seconds, MIDI_note_0-127, confidence_0-1)
# Multiple events at the same timestamp = simultaneous chord notes.
# ---------------------------------------------------------------------------
NoteEvent = tuple[float, int, float]


class PitchDetectionError(Exception):
    """Raised when any pitch-detection backend fails."""


# ===========================================================================
# pYIN tracker (monophonic)
# ===========================================================================

class PitchTracker:
    """Monophonic pitch tracker using librosa's pYIN algorithm.

    Optional HPSS pre-processing separates harmonic content (melody, chords)
    from percussive content (drums) before running pYIN, significantly
    improving note detection on mixed recordings.

    Example::

        tracker = PitchTracker(audio, sr, use_hpss=True)
        notes  = tracker.track()
        tempo  = tracker.estimate_tempo()
    """

    FMIN: float = 65.41    # C2 Hz
    FMAX: float = 2093.0   # C7 Hz
    DEFAULT_CONFIDENCE: float = 0.45

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        min_confidence: float = DEFAULT_CONFIDENCE,
        use_hpss: bool = True,
    ) -> None:
        """Initialise PitchTracker.

        Args:
            audio: Mono float32 audio samples.
            sr: Sample rate in Hz.
            min_confidence: Voiced-probability threshold ``[0, 1]``.
                0.45 suits mixed recordings; raise to 0.7+ for clean tracks.
            use_hpss: Apply Harmonic-Percussive Source Separation before
                pYIN.  Strongly recommended for full-mix recordings.
        """
        self.audio = audio
        self.sr = sr
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.use_hpss = use_hpss
        self._tempo: Optional[float] = None

    def track(self) -> list[NoteEvent]:
        """Run pYIN and return voiced note events sorted by time.

        Returns:
            List of ``(time_sec, midi_note, confidence)`` tuples.

        Raises:
            PitchDetectionError: On any librosa failure.
        """
        try:
            import librosa

            src = self.audio
            if self.use_hpss:
                harmonic, _ = librosa.effects.hpss(src)
                src = harmonic

            f0, voiced_flag, voiced_probs = librosa.pyin(
                src, fmin=self.FMIN, fmax=self.FMAX, sr=self.sr
            )
        except ImportError as exc:
            raise PitchDetectionError("librosa not installed: pip install librosa") from exc
        except Exception as exc:
            raise PitchDetectionError(f"pYIN failed: {exc}") from exc

        times = librosa.times_like(f0, sr=self.sr)
        events: list[NoteEvent] = []
        for t, freq, voiced, prob in zip(times, f0, voiced_flag, voiced_probs):
            if voiced and freq and not np.isnan(freq) and float(prob) >= self.min_confidence:
                midi = int(np.clip(round(float(librosa.hz_to_midi(freq))), 0, 127))
                events.append((float(t), midi, float(prob)))
        return events

    def estimate_tempo(self) -> float:
        """Estimate tempo in BPM (cached).  Falls back to 120.0 on error."""
        if self._tempo is not None:
            return self._tempo
        try:
            import librosa
            raw = librosa.beat.beat_track(y=self.audio, sr=self.sr)[0]
            bpm = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
            self._tempo = bpm if bpm > 0 else 120.0
        except Exception:
            self._tempo = 120.0
        return self._tempo


# ===========================================================================
# Basic-pitch tracker (polyphonic, neural network)
# ===========================================================================

class BasicPitchTracker:
    """Polyphonic pitch tracker powered by Spotify's Basic-pitch model.

    Basic-pitch uses a lightweight convolutional neural network (CNN) trained
    on diverse real-music recordings.  It detects *multiple simultaneous
    notes*, making it suitable for guitar chords and full-mix audio.

    The returned ``NoteEvent`` list may contain several events sharing the
    same ``time_sec`` — these represent simultaneously sounding notes
    (a chord).  Pass the result to ``TabSimplifier.quantize_chords()`` to
    group them correctly.

    The TensorFlow model is loaded on the first call to :meth:`track` and
    cached for the lifetime of the process.  This adds ~5-10 s overhead on
    the first run; subsequent calls are fast.

    Example::

        tracker = BasicPitchTracker(audio, sr, onset_threshold=0.5)
        notes  = tracker.track()         # polyphonic NoteEvents
        tempo  = tracker.estimate_tempo()
    """

    FMIN: float = 65.41    # C2
    FMAX: float = 2093.0   # C7

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float = 80.0,
    ) -> None:
        """Initialise BasicPitchTracker.

        Args:
            audio: Mono float32 audio samples.
            sr: Sample rate in Hz.
            onset_threshold: Confidence required to start a new note
                ``[0, 1]``.  Higher = fewer but more confident notes.
                Default 0.5 is a good starting point for mixed recordings.
            frame_threshold: Confidence to sustain a note across frames
                ``[0, 1]``.  Usually kept below ``onset_threshold``.
            minimum_note_length_ms: Shortest note kept (milliseconds).
                80 ms ≈ a 32nd note at 93 BPM — filters sub-note blips.
        """
        self.audio = audio
        self.sr = sr
        self.onset_threshold = float(np.clip(onset_threshold, 0.0, 1.0))
        self.frame_threshold = float(np.clip(frame_threshold, 0.0, 1.0))
        self.minimum_note_length_ms = maximum = float(minimum_note_length_ms)
        self._tempo: Optional[float] = None

    def track(self) -> list[NoteEvent]:
        """Run Basic-pitch inference and return polyphonic note events.

        The audio is written to a temporary WAV file (required by the
        Basic-pitch API), inference is run, and the file is cleaned up.

        Each ``NoteEvent`` is ``(start_time_sec, midi_note, amplitude)``.
        Multiple events at the same ``start_time_sec`` represent a chord.

        Returns:
            List of :data:`NoteEvent` tuples sorted by time.

        Raises:
            PitchDetectionError: If basic-pitch or tensorflow are missing,
                or if inference fails.
        """
        try:
            import soundfile as sf
            from basic_pitch.inference import predict, Model
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError as exc:
            raise PitchDetectionError(
                "basic-pitch is not installed.  "
                "Run: pip install basic-pitch"
            ) from exc

        # Resolve the best available model backend.
        # basic-pitch 0.3 default path points to the TF saved-model dir
        # which may fail on newer TF versions.  Prefer the ONNX variant
        # if onnxruntime is installed, as it's lighter and more portable.
        import pathlib
        model_path = pathlib.Path(ICASSP_2022_MODEL_PATH)
        onnx_path = model_path.with_suffix(".onnx")
        try:
            import onnxruntime  # noqa: F401
            if onnx_path.exists():
                model_path = onnx_path
        except ImportError:
            pass  # fall back to default (TF / TFLite)

        # Write audio to a temporary WAV (basic-pitch requires a file path)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                pass  # close fd so soundfile can write
            sf.write(tmp_path, self.audio, self.sr)

            _, _, raw_notes = predict(
                tmp_path,
                model_path,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
                minimum_note_length=self.minimum_note_length_ms,
                minimum_frequency=self.FMIN,
                maximum_frequency=self.FMAX,
            )
        except PitchDetectionError:
            raise
        except Exception as exc:
            raise PitchDetectionError(f"Basic-pitch inference failed: {exc}") from exc
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # raw_notes: list of (start, end, midi_int, amplitude, pitch_bends)
        events: list[NoteEvent] = []
        for note in raw_notes:
            start_time = float(note[0])
            midi = int(note[2])
            amplitude = float(note[3])
            midi = int(np.clip(midi, 0, 127))
            events.append((start_time, midi, amplitude))

        return sorted(events, key=lambda e: e[0])

    def estimate_tempo(self) -> float:
        """Estimate tempo in BPM using librosa (cached).  Falls back to 120."""
        if self._tempo is not None:
            return self._tempo
        try:
            import librosa
            raw = librosa.beat.beat_track(y=self.audio, sr=self.sr)[0]
            bpm = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
            self._tempo = bpm if bpm > 0 else 120.0
        except Exception:
            self._tempo = 120.0
        return self._tempo
