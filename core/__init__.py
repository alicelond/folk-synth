"""
Folk Synth Core Module

A physical modelling Karplus-Strong synthesizer for acoustic guitar synthesis.

Main Classes:
    - KarplusStrong: Core synthesizer
    - AudioPlayer: Real-time playback
    - AudioExporter: WAV file export

Convenience Functions:
    - play_note(): Synthesize and play a note
    - save_note(): Synthesize and save to file

Usage:
    >>> from core import KarplusStrong, play_note, save_note
    >>> # Synthesize a note
    >>> synth = KarplusStrong(frequency=440, duration=1.0)
    >>> audio = synth.synthesize()
    >>> # Or play directly
    >>> play_note(440)
    >>> # Or save to file
    >>> save_note('note.wav', 440)
"""

from core.string import KarplusStrong
from core.audio_output import AudioPlayer, AudioExporter, play_note, save_note

__version__ = "0.1.0"
__author__ = "Folk Synth Contributors"

__all__ = [
    "KarplusStrong",
    "AudioPlayer",
    "AudioExporter",
    "play_note",
    "save_note",
]
