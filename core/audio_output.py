"""
Audio output module for playback and export.

Provides real-time audio playback and WAV file export capabilities
for the synthesized audio.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from core.string import KarplusStrong


class AudioPlayer:
    """
    Real-time audio playback handler.
    
    Uses sounddevice library for low-latency playback to the system audio output.
    """
    
    def __init__(self, sample_rate=44100, channels=1):
        """
        Initialize the audio player.
        
        Args:
            sample_rate (float): Audio sample rate in Hz. Default 44100.
            channels (int): Number of channels (1=mono, 2=stereo). Default 1.
        """
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
    
    def play(self, audio_samples):
        """
        Play audio samples through the system audio device.
        
        Args:
            audio_samples (np.ndarray): Audio samples to play.
                Shape: (num_samples,) for mono or (num_samples, channels) for multi-channel.
                Values should be in roughly [-1.0, 1.0] range.
        
        Raises:
            ValueError: If audio_samples is invalid.
        """
        if not isinstance(audio_samples, np.ndarray):
            raise ValueError("audio_samples must be a numpy array")
        
        if audio_samples.ndim not in (1, 2):
            raise ValueError("audio_samples must be 1D or 2D array")
        
        if audio_samples.size == 0:
            raise ValueError("audio_samples cannot be empty")
        
        # Normalize to prevent clipping if needed
        max_val = np.max(np.abs(audio_samples))
        if max_val > 1.0:
            # Soft clipping: normalize to prevent distortion
            audio_samples = audio_samples / (1.1 * max_val)
        
        # Convert to float32 for playback
        audio_data = audio_samples.astype(np.float32)
        
        # Play using sounddevice
        sd.play(audio_data, samplerate=self.sample_rate, blocking=True)
    
    def stop(self):
        """Stop current playback if any."""
        sd.stop()


class AudioExporter:
    """
    WAV file export handler.
    
    Exports synthesized audio to standard WAV files for offline playback.
    """
    
    def __init__(self, sample_rate=44100):
        """
        Initialize the audio exporter.
        
        Args:
            sample_rate (float): Audio sample rate in Hz. Default 44100.
        """
        self.sample_rate = int(sample_rate)
    
    def save(self, filename, audio_samples, normalize=True):
        """
        Export audio samples to a WAV file.
        
        Args:
            filename (str): Path to output WAV file.
            audio_samples (np.ndarray): Audio samples to save.
                Shape: (num_samples,) for mono or (num_samples, channels) for multi-channel.
            normalize (bool): If True, normalize peak to -0.1 dB to prevent clipping.
                Default True.
        
        Raises:
            ValueError: If audio_samples is invalid.
            IOError: If file cannot be written.
        """
        if not isinstance(audio_samples, np.ndarray):
            raise ValueError("audio_samples must be a numpy array")
        
        if audio_samples.size == 0:
            raise ValueError("audio_samples cannot be empty")
        
        # Normalize if requested
        audio_data = audio_samples.astype(np.float64)
        
        if normalize:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                # Normalize to -0.1 dB (0.989 peak)
                target_level = 0.989
                audio_data = audio_data * (target_level / max_val)
        
        # Save using soundfile
        sf.write(filename, audio_data, samplerate=self.sample_rate)


def play_note(frequency, duration=1.0, sample_rate=44100, 
              pluck_intensity=0.5, pluck_position=0.5, decay_factor=0.99,
              stretch_factor=1.0, rng_seed=None):
    """
    Synthesize and play a single note in real-time.
    
    Convenience function that creates a KarplusStrong synthesizer, synthesizes
    the note, and plays it immediately.
    
    Args:
        frequency (float): Note frequency in Hz.
        duration (float): Note duration in seconds. Default 1.0.
        sample_rate (float): Audio sample rate in Hz. Default 44100.
        pluck_intensity (float): Pluck hardness in [0, 1]. Default 0.5.
        pluck_position (float): Pluck position in [0, 1]. Default 0.5.
        decay_factor (float): Sustain decay in [0, 1). Default 0.99.
        stretch_factor (float): Decay stretching factor. Default 1.0.
        rng_seed (int): Random seed for reproducibility. Default None.
    
    Raises:
        ValueError: If parameters are invalid.
    """
    # Create synthesizer
    synth = KarplusStrong(
        frequency=frequency,
        sample_rate=sample_rate,
        duration=duration,
        pluck_intensity=pluck_intensity,
        pluck_position=pluck_position,
        decay_factor=decay_factor,
        stretch_factor=stretch_factor,
        rng_seed=rng_seed
    )
    
    # Synthesize
    audio = synth.synthesize()
    
    # Play
    player = AudioPlayer(sample_rate=sample_rate)
    player.play(audio)


def save_note(filename, frequency, duration=1.0, sample_rate=44100,
              pluck_intensity=0.5, pluck_position=0.5, decay_factor=0.99,
              stretch_factor=1.0, rng_seed=None, normalize=True):
    """
    Synthesize and save a single note to a WAV file.
    
    Convenience function that creates a KarplusStrong synthesizer, synthesizes
    the note, and exports it to a WAV file.
    
    Args:
        filename (str): Output WAV file path.
        frequency (float): Note frequency in Hz.
        duration (float): Note duration in seconds. Default 1.0.
        sample_rate (float): Audio sample rate in Hz. Default 44100.
        pluck_intensity (float): Pluck hardness in [0, 1]. Default 0.5.
        pluck_position (float): Pluck position in [0, 1]. Default 0.5.
        decay_factor (float): Sustain decay in [0, 1). Default 0.99.
        stretch_factor (float): Decay stretching factor. Default 1.0.
        rng_seed (int): Random seed for reproducibility. Default None.
        normalize (bool): Normalize to -0.1 dB peak. Default True.
    
    Raises:
        ValueError: If parameters are invalid.
        IOError: If file cannot be written.
    """
    # Create synthesizer
    synth = KarplusStrong(
        frequency=frequency,
        sample_rate=sample_rate,
        duration=duration,
        pluck_intensity=pluck_intensity,
        pluck_position=pluck_position,
        decay_factor=decay_factor,
        stretch_factor=stretch_factor,
        rng_seed=rng_seed
    )
    
    # Synthesize
    audio = synth.synthesize()
    
    # Export
    exporter = AudioExporter(sample_rate=sample_rate)
    exporter.save(filename, audio, normalize=normalize)
