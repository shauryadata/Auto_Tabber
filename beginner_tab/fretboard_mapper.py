"""Map MIDI note numbers to guitar fretboard positions.

Two mapping modes share the same tuning / max-fret constraints:

map()          — monophonic: one note → one (string, fret) position.
map_chords()   — polyphonic: a list of simultaneous MIDI notes → multiple
                 (string, fret) positions, one per string (no two notes on
                 the same string at the same time).
"""

from __future__ import annotations

from typing import Optional

from .tab_simplifier import QuantizedNote, ChordNote

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# Index 0 = high e (thinnest, highest pitch), index 5 = low E
STANDARD_TUNING_MIDI: list[int] = [64, 59, 55, 50, 45, 40]
STRING_NAMES: list[str] = ["e", "B", "G", "D", "A", "E"]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FretPosition = tuple[int, int]                              # (string_idx, fret)
TabNote = tuple[float, int, int, int]                       # (time, str, fret, midi)
ChordTabNote = tuple[float, list[FretPosition], list[int]]  # (time, positions, midis)


class FretboardMapper:
    """Map MIDI pitches to guitar fret positions.

    For monophonic use: call :meth:`map`.
    For polyphonic / chord use: call :meth:`map_chords`.

    Both methods respect ``max_fret`` and ``one_string_mode``.

    Example::

        mapper = FretboardMapper(max_fret=5)
        tab_notes   = mapper.map(quantized_notes)        # monophonic
        chord_notes = mapper.map_chords(chord_notes)     # polyphonic
    """

    def __init__(
        self,
        max_fret: int = 5,
        one_string_mode: bool = False,
        tuning: Optional[list[int]] = None,
    ) -> None:
        """Initialise FretboardMapper.

        Args:
            max_fret: Maximum fret number allowed (e.g. 3, 5, 7, 12).
            one_string_mode: Only use the high-e string (index 0).
            tuning: Open-string MIDI notes high → low.  Defaults to
                standard EADGBE tuning.
        """
        self.max_fret = int(max_fret)
        self.one_string_mode = one_string_mode
        self.tuning: list[int] = tuning if tuning is not None else STANDARD_TUNING_MIDI
        self._active_strings: list[int] = (
            [0] if one_string_mode else list(range(len(self.tuning)))
        )
        self._last_skipped: int = 0

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def get_positions(self, midi_note: int) -> list[FretPosition]:
        """Return all valid ``(string_idx, fret)`` pairs for *midi_note*.

        Args:
            midi_note: MIDI note number 0–127.

        Returns:
            List of ``(string_index, fret)`` within ``[0, max_fret]``.
        """
        positions: list[FretPosition] = []
        for s in self._active_strings:
            fret = midi_note - self.tuning[s]
            if 0 <= fret <= self.max_fret:
                positions.append((s, fret))
        return positions

    # ------------------------------------------------------------------
    # Monophonic mapping
    # ------------------------------------------------------------------

    def map(self, notes: list[QuantizedNote]) -> list[TabNote]:
        """Map quantised monophonic notes to fretboard positions.

        Minimises hand movement using cost ``|Δstring|×3 + |Δfret|``.
        Input is sorted by time defensively.

        Args:
            notes: Output of :meth:`TabSimplifier.quantize`.

        Returns:
            List of ``(time, string_idx, fret, midi)`` tuples.
        """
        self._last_skipped = 0
        result: list[TabNote] = []
        prev_string: Optional[int] = None
        prev_fret: Optional[int] = None

        for time, midi, _conf in sorted(notes, key=lambda n: n[0]):
            positions = self.get_positions(midi)
            if not positions:
                self._last_skipped += 1
                continue
            string_idx, fret = self._best_mono_position(positions, prev_string, prev_fret)
            result.append((time, string_idx, fret, midi))
            prev_string, prev_fret = string_idx, fret

        return result

    def _best_mono_position(
        self,
        positions: list[FretPosition],
        prev_string: Optional[int],
        prev_fret: Optional[int],
    ) -> FretPosition:
        if prev_string is None or prev_fret is None:
            return min(positions, key=lambda p: (p[0], p[1]))

        def cost(p: FretPosition) -> float:
            return abs(p[0] - prev_string) * 3.0 + abs(p[1] - prev_fret)

        return min(positions, key=cost)

    @property
    def skipped_count(self) -> int:
        """Notes skipped in the last :meth:`map` or :meth:`map_chords` call."""
        return self._last_skipped

    # ------------------------------------------------------------------
    # Polyphonic / chord mapping
    # ------------------------------------------------------------------

    def map_chords(self, chords: list[ChordNote]) -> list[ChordTabNote]:
        """Map polyphonic chord notes to fretboard positions.

        Each chord is assigned to a set of strings with no two notes on
        the same string.  Notes are processed high→low; each is placed on
        the lowest-index (highest-pitched) available string so the chord
        voicing spreads naturally from treble to bass strings.

        Notes that cannot be placed are silently dropped; the count is
        accumulated in :attr:`skipped_count`.

        Args:
            chords: Output of :meth:`TabSimplifier.quantize_chords`.

        Returns:
            List of ``(time, [(string_idx, fret), …], [midi, …])`` tuples.
        """
        self._last_skipped = 0
        result: list[ChordTabNote] = []

        for time, midi_list, _conf in sorted(chords, key=lambda c: c[0]):
            positions, placed_midis = self._assign_chord(midi_list)
            self._last_skipped += len(midi_list) - len(placed_midis)
            if positions:
                result.append((time, positions, placed_midis))

        return result

    def _assign_chord(
        self, midi_notes: list[int]
    ) -> tuple[list[FretPosition], list[int]]:
        """Greedily assign strings to chord notes (no string shared).

        Process notes from highest pitch to lowest.  For each note pick
        the lowest available string index (highest string = top of chord).
        Lower notes fall naturally to higher string indices (deeper strings).

        Returns:
            ``(positions, placed_midis)`` — parallel lists.
        """
        used: set[int] = set()
        pairs: list[tuple[FretPosition, int]] = []

        for midi in sorted(midi_notes, reverse=True):   # high → low
            candidates = [
                (s, f) for s, f in self.get_positions(midi) if s not in used
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda p: p[0])  # lowest string index
            pairs.append((best, midi))
            used.add(best[0])

        # Sort by string index so tab renders top-string first
        pairs.sort(key=lambda x: x[0][0])
        if pairs:
            positions_out, midis_out = zip(*pairs)
            return list(positions_out), list(midis_out)
        return [], []
