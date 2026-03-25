"""Quantise raw pitch events onto a beat grid and simplify for beginners.

Two quantisation modes:

quantize()
    Monophonic — one note per time slot (highest confidence wins).
    Use with :class:`PitchTracker` (pYIN) output.

quantize_chords()
    Polyphonic — ALL notes in a slot are kept as a chord group.
    Use with :class:`BasicPitchTracker` (Basic-pitch) output.

Both modes:
* Collapse *strictly adjacent* consecutive slots with the same pitch
  (sustained notes) into a single event.
* Preserve non-adjacent same-pitch events (repeated notes).
"""

from __future__ import annotations

from .pitch_tracker import NoteEvent

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# Monophonic: one MIDI note per time slot
QuantizedNote = tuple[float, int, float]    # (time, midi, confidence)

# Polyphonic: multiple MIDI notes per time slot (chord)
ChordNote = tuple[float, list[int], float]  # (time, [midi, …], avg_confidence)


class TabSimplifier:
    """Quantise and clean up a raw pitch-event sequence.

    Example::

        simp = TabSimplifier(raw_notes, tempo=98.0)

        # monophonic path
        notes = simp.quantize()
        notes = simp.simplify_range(notes)

        # polyphonic path
        chords = simp.quantize_chords()
        chords = simp.simplify_range_chords(chords)
    """

    def __init__(
        self,
        notes: list[NoteEvent],
        tempo: float = 120.0,
        beats_per_measure: int = 4,
        subdivision: int = 2,
    ) -> None:
        """Initialise TabSimplifier.

        Args:
            notes: Output of any pitch tracker's ``track()`` method.
            tempo: Song tempo in BPM.
            beats_per_measure: Time-signature numerator (default 4).
            subdivision: Slots per beat: 1 = quarter, 2 = 8th (default),
                4 = 16th notes.
        """
        self.notes = notes
        self.tempo = max(float(tempo), 1.0)
        self.beats_per_measure = beats_per_measure
        self.subdivision = subdivision

    @property
    def slot_duration(self) -> float:
        """Duration of one quantisation slot in seconds."""
        return 60.0 / (self.tempo * self.subdivision)

    # ------------------------------------------------------------------
    # Monophonic path
    # ------------------------------------------------------------------

    def quantize(self) -> list[QuantizedNote]:
        """Quantise to beat grid — one note per slot (highest confidence).

        Only strictly adjacent slots carrying the same pitch are collapsed
        (sustained note).  Repeated same-pitch events with a gap are kept.

        Returns:
            Sorted list of ``(beat_time, midi, confidence)`` tuples.
        """
        if not self.notes:
            return []

        slot_dur = self.slot_duration
        slots: dict[int, QuantizedNote] = {}

        for time, midi, conf in self.notes:
            idx = int(round(time / slot_dur))
            existing = slots.get(idx)
            if existing is None or conf > existing[2]:
                slots[idx] = (idx * slot_dur, midi, conf)

        sorted_items = sorted(slots.items(), key=lambda x: x[0])

        # Collapse only strictly adjacent slots with the same pitch
        merged: list[tuple[int, QuantizedNote]] = []
        for slot_idx, note in sorted_items:
            if (
                merged
                and merged[-1][0] == slot_idx - 1
                and merged[-1][1][1] == note[1]
            ):
                prev_idx, prev_note = merged[-1]
                merged[-1] = (
                    slot_idx,
                    (prev_note[0], prev_note[1], max(prev_note[2], note[2])),
                )
            else:
                merged.append((slot_idx, note))

        return [note for _, note in merged]

    def simplify_range(
        self,
        notes: list[QuantizedNote],
        target_midi_min: int = 40,
        target_midi_max: int = 88,
    ) -> list[QuantizedNote]:
        """Transpose out-of-guitar-range notes by octave (monophonic).

        Args:
            notes: Output of :meth:`quantize`.
            target_midi_min: Lowest acceptable MIDI (E2 = 40).
            target_midi_max: Highest acceptable MIDI (default 88).

        Returns:
            New list with all pitches inside the target range.
        """
        result: list[QuantizedNote] = []
        for time, midi, conf in notes:
            while midi < target_midi_min:
                midi += 12
            while midi > target_midi_max:
                midi -= 12
            result.append((time, midi, conf))
        return result

    # ------------------------------------------------------------------
    # Polyphonic / chord path
    # ------------------------------------------------------------------

    def quantize_chords(self) -> list[ChordNote]:
        """Quantise to beat grid keeping ALL simultaneous notes per slot.

        Unlike :meth:`quantize`, every note in a slot is preserved —
        multiple notes at the same slot index form a chord.  Adjacent
        slots whose *complete pitch sets* are identical are collapsed
        (sustained chord).

        Returns:
            Sorted list of ``(beat_time, [midi_notes], avg_confidence)``
            tuples.
        """
        if not self.notes:
            return []

        slot_dur = self.slot_duration

        # Accumulate all (midi, conf) pairs per slot
        slot_notes: dict[int, list[tuple[int, float]]] = {}
        for time, midi, conf in self.notes:
            idx = int(round(time / slot_dur))
            slot_notes.setdefault(idx, []).append((midi, conf))

        # Deduplicate within each slot (same MIDI pitch — keep highest conf)
        slot_chords: dict[int, tuple[float, list[int], float]] = {}
        for idx, note_list in slot_notes.items():
            best: dict[int, float] = {}
            for midi, conf in note_list:
                best[midi] = max(best.get(midi, 0.0), conf)
            midi_list = sorted(best.keys())
            avg_conf = sum(best.values()) / len(best)
            slot_chords[idx] = (idx * slot_dur, midi_list, avg_conf)

        sorted_items = sorted(slot_chords.items(), key=lambda x: x[0])

        # Collapse only strictly adjacent slots with identical pitch sets
        merged: list[tuple[int, ChordNote]] = []
        for slot_idx, chord in sorted_items:
            if (
                merged
                and merged[-1][0] == slot_idx - 1
                and merged[-1][1][1] == chord[1]
            ):
                prev_idx, prev_chord = merged[-1]
                merged[-1] = (
                    slot_idx,
                    (prev_chord[0], prev_chord[1], max(prev_chord[2], chord[2])),
                )
            else:
                merged.append((slot_idx, chord))

        return [chord for _, chord in merged]

    def simplify_range_chords(
        self,
        chords: list[ChordNote],
        target_midi_min: int = 40,
        target_midi_max: int = 88,
    ) -> list[ChordNote]:
        """Transpose each pitch in every chord into the guitar range.

        Args:
            chords: Output of :meth:`quantize_chords`.
            target_midi_min: Lowest acceptable MIDI (E2 = 40).
            target_midi_max: Highest acceptable MIDI (default 88).

        Returns:
            New list with all pitches transposed into the target range.
        """
        result: list[ChordNote] = []
        for time, midi_list, conf in chords:
            fixed: list[int] = []
            seen: set[int] = set()
            for midi in midi_list:
                while midi < target_midi_min:
                    midi += 12
                while midi > target_midi_max:
                    midi -= 12
                if midi not in seen:          # avoid octave-collision duplicates
                    fixed.append(midi)
                    seen.add(midi)
            if fixed:
                result.append((time, sorted(fixed), conf))
        return result
