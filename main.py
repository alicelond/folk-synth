#!/usr/bin/env python3
"""
Main script to demonstrate the Folk Synth Karplus-Strong synthesizer.

This script showcases various synthesis features including:
- Single note synthesis with different parameters
- Parameter effects (pluck intensity, position, decay)
- Chord generation
- Audio file export
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core import KarplusStrong, play_note, save_note, AudioExporter


def demo_single_notes():
    """Play several single notes with different parameters."""
    print("\n🎵 Playing single notes with different parameters...\n")
    
    notes = [
        {"freq": 220, "name": "A3", "intensity": 0.3, "desc": "Soft pluck"},
        {"freq": 440, "name": "A4", "intensity": 0.5, "desc": "Medium pluck"},
        {"freq": 880, "name": "A5", "intensity": 0.8, "desc": "Hard pluck"},
    ]
    
    for note in notes:
        print(f"  Playing {note['name']} ({note['freq']} Hz) - {note['desc']}...")
        play_note(
            frequency=note['freq'],
            duration=0.5,
            pluck_intensity=note['intensity']
        )


def demo_pluck_position():
    """Demonstrate the effect of pluck position on timbre."""
    print("\n🎸 Demonstrating pluck position effects...\n")
    
    positions = [
        (0.2, "Near bridge (bright)"),
        (0.5, "Center string (balanced)"),
        (0.8, "Near nut (dark)"),
    ]
    
    for pos, description in positions:
        print(f"  Plucking at position {pos}: {description}...")
        play_note(
            frequency=440,
            duration=1.0,
            pluck_position=pos,
            pluck_intensity=0.7
        )


def demo_decay_stretching():
    """Demonstrate decay stretching for different sustain lengths."""
    print("\n⏱️  Demonstrating decay stretching...\n")
    
    stretches = [
        (0.5, "Shorter sustain (staccato)"),
        (1.0, "Normal sustain"),
        (2.0, "Longer sustain (resonant)"),
    ]
    
    for stretch, description in stretches:
        print(f"  Stretch factor {stretch}: {description}...")
        play_note(
            frequency=440,
            duration=2.0,
            decay_factor=0.98,
            stretch_factor=stretch,
            pluck_intensity=0.7
        )


def demo_save_notes():
    """Save several notes to WAV files."""
    print("\n💾 Saving synthesized notes to files...\n")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    notes = [
        ("A3_soft.wav", 220, "A3 - Soft pluck", 0.3),
        ("A4_medium.wav", 440, "A4 - Medium pluck", 0.5),
        ("A5_hard.wav", 880, "A5 - Hard pluck", 0.8),
        ("C_major.wav", 262, "C4 - Full sustain", 0.6),
    ]
    
    for filename, freq, description, intensity in notes:
        filepath = output_dir / filename
        print(f"  Saving {description}...")
        save_note(
            str(filepath),
            frequency=freq,
            duration=1.5,
            pluck_intensity=intensity,
            decay_factor=0.99
        )
    
    print(f"  ✅ Notes saved to '{output_dir}/' directory")


def demo_chord():
    """Generate a chord by synthesizing multiple notes sequentially."""
    print("\n🎼 Generating a chord (C Major)...\n")
    
    # C Major chord: C4, E4, G4
    chord_notes = [
        (262, "C4"),
        (330, "E4"),
        (392, "G4"),
    ]
    
    print("  Synthesizing chord notes:")
    audio_parts = []
    synth_sr = 44100
    
    for freq, note_name in chord_notes:
        print(f"    - {note_name} ({freq} Hz)")
        synth = KarplusStrong(
            frequency=freq,
            sample_rate=synth_sr,
            duration=1.0,
            pluck_intensity=0.6,
            decay_factor=0.99
        )
        audio_parts.append(synth.synthesize())
    
    # Mix notes together (simple average)
    import numpy as np
    mixed_audio = np.mean(audio_parts, axis=0)
    
    # Normalize
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 1:
        mixed_audio = mixed_audio / (1.1 * max_val)
    
    # Save the chord
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    exporter = AudioExporter(sample_rate=synth_sr)
    chord_path = output_dir / "chord_c_major.wav"
    exporter.save(str(chord_path), mixed_audio)
    print(f"\n  ✅ Chord saved to '{chord_path}'")


def demo_custom_synthesis():
    """Demonstrate fully custom synthesis with detailed parameters."""
    print("\n⚙️  Custom synthesis with detailed parameter control...\n")
    
    print("  Creating custom E4 with specific parameters:")
    print("    - Frequency: 330 Hz (E4)")
    print("    - Duration: 2.0 seconds")
    print("    - Pluck intensity: 0.7 (fairly hard)")
    print("    - Pluck position: 0.3 (toward bridge - brighter)")
    print("    - Decay factor: 0.985 (medium-long sustain)")
    print("    - Stretch factor: 1.5 (extended sustain)\n")
    
    synth = KarplusStrong(
        frequency=330,
        sample_rate=44100,
        duration=2.0,
        pluck_intensity=0.7,
        pluck_position=0.3,
        decay_factor=0.985,
        stretch_factor=1.5
    )
    
    audio = synth.synthesize()
    print("  Synthesizing audio...")
    
    # Save it
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    exporter = AudioExporter(sample_rate=44100)
    filepath = output_dir / "custom_e4.wav"
    exporter.save(str(filepath), audio)
    
    print(f"  ✅ Custom synthesis saved to '{filepath}'")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("🎸 Folk Synth - Karplus-Strong Synthesizer Demo")
    print("=" * 60)
    
    try:
        # Uncomment demos to run them
        # (Some may require audio output device)
        
        # demo_single_notes()
        # demo_pluck_position()
        # demo_decay_stretching()
        
        demo_save_notes()
        demo_chord()
        demo_custom_synthesis()
        
        print("\n" + "=" * 60)
        print("✅ Demo complete!")
        print("=" * 60)
        print("\nTo play audio notes, uncomment the demo functions in main().")
        print("Audio files have been saved to the 'output/' directory.\n")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
