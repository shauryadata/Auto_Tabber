# AutoTabber

Convert an audio recording of a **monophonic melody** into a beginner-friendly
ASCII guitar tab — entirely on your local machine.

---

## Features

| Feature | Detail |
|---|---|
| Audio input | MP3, WAV, M4A (WAV needs no extra tools) |
| Pitch detection | pYIN via librosa — robust for guitar/voice |
| Max-fret control | 3 / 5 / 7 / 12 (user-adjustable) |
| One-string mode | High-e only — for absolute beginners |
| Confidence filter | Adjustable voiced-probability threshold |
| Note grid | Quarter / 8th / 16th note quantisation |
| Output | ASCII tab + note-name legend |
| Download | `.txt` file with metadata header |

---

## Requirements

### Python

Python **3.11** is required (type annotations use 3.10+ syntax).

### ffmpeg (for MP3 / M4A)

pydub delegates audio decoding to **ffmpeg**.  WAV files work without it.

| Platform | Install command |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Windows | Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` folder to `PATH` |

Verify the install: `ffmpeg -version`

---

## Quick Start

```bash
# 1. Clone / download the project
cd AutoTabber

# 2. (Recommended) create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

---

## Usage

1. **Upload** an MP3, WAV, or M4A file containing a monophonic melody (solo
   guitar, whistle, a hummed tune, etc.).
2. **Optionally** paste a YouTube URL in the reference field — this is for
   your own notes only; the app does not download from YouTube.
3. Adjust **settings** in the sidebar:
   - *Max Fret* — limits how far up the neck the tab goes.
   - *One-String Mode* — all notes on the high-e string only.
   - *Pitch Detection Confidence* — raise for cleaner notes, lower for
     quieter/noisier recordings.
   - *Note Grid* — quantisation subdivision.
4. Click **Generate Tab**.
5. Read the ASCII tab in-browser or **Download** it as a `.txt` file.

### Reading the tab

```
e|--0---3---5---3---0--|
B|---------------------|
G|---------------------|
D|---------------------|
A|---------------------|
E|---------------------|

Notes:  E4  G4  A4  G4  E4
```

- Row labels (`e B G D A E`) are guitar strings, high → low.
- **Numbers** = fret to press.  **0** = open string (no fret).
- **Dashes** = that string is not played on this beat.

---

## Project Layout

```
AutoTabber/
├── app.py                      # Streamlit frontend
├── requirements.txt
├── README.md
├── beginner_tab/               # Core package
│   ├── __init__.py
│   ├── audio_loader.py         # AudioLoader  — load & convert audio
│   ├── pitch_tracker.py        # PitchTracker — pYIN pitch detection
│   ├── tab_simplifier.py       # TabSimplifier — quantise & clean notes
│   ├── fretboard_mapper.py     # FretboardMapper — MIDI → fret positions
│   └── tab_renderer.py         # TabRenderer  — ASCII tab output
└── tests/
    ├── __init__.py
    ├── test_fretboard_mapper.py
    ├── test_tab_simplifier.py
    └── test_tab_renderer.py
```

---

## Running the Tests

```bash
pytest tests/ -v
```

The unit tests cover `FretboardMapper`, `TabSimplifier`, and `TabRenderer`
without requiring any audio files or internet access.

---

## Architecture

```
AudioLoader
    │  (mono float32 array, sample_rate)
    ▼
PitchTracker
    │  list[NoteEvent]  (time, midi_note, confidence)
    ▼
TabSimplifier
    │  list[QuantizedNote]  (beat_time, midi_note, confidence)
    ▼
FretboardMapper
    │  list[TabNote]  (time, string_idx, fret, midi_note)
    ▼
TabRenderer
       str  (ASCII guitar tab)
```

Each class has full type hints and docstrings on all public methods.

---

## Known Limitations

- **Monophonic only** — chords and harmonies are not supported.
- **No rhythm notation** — the tab shows note order, not durations.
- **Octave errors** — pYIN can occasionally misidentify the octave; the
  `simplify_range` step corrects most cases automatically.
- **ffmpeg dependency** — MP3 / M4A files require ffmpeg; WAV does not.
- **Long files are slow** — pitch detection runs in real time; trim to the
  section you need.
- **Tab accuracy** — intended as a learning aid, not a professional
  transcription.

---

## Legal

Upload only audio you own or are authorised to use.  The YouTube URL field
is a labelling / reference field only — AutoTabber does not download or
stream from YouTube or any external service.
