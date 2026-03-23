"""
KarplusStrong synthesizer module.

Main implementation of the Karplus-Strong algorithm orchestrating all components:
excitation, delay line, and filters.

References:
- Karplus & Strong (1983): "Digital Synthesis of Plucked-String and Drum Timbres"
- Jaffe & Smith (1983): "Extensions of the Karplus-Strong Plucked-String Algorithm"
"""

import numpy as np
from core.excitation import NoiseExcitation
from core.delay_line import DelayLine
from core.filters import AllpassFilter, LowpassFilter, CombFilter, DecayStretcher


class KarplusStrong:
    """
    Karplus-Strong algorithm synthesizer for plucked string synthesis.
    
    Implements physical modelling synthesis with support for fractional delay tuning,
    dynamic bandwidth filtering, pluck position effects, and decay stretching.
    
    Attributes:
        sample_rate (float): Audio sample rate in Hz.
        frequency (float): Target frequency of the synthesized note in Hz.
        duration (float): Total duration of synthesis in seconds.
        pluck_intensity (float): Initial amplitude and decay rate (0-1).
        pluck_position (float): Pick position along string (0=bridge, 1=nut).
        decay_factor (float): Base feedback coefficient controlling sustain.
        stretch_factor (float): Decay stretching for independent sustain control.
    """
    
    def __init__(self, frequency, sample_rate=44100, duration=1.0, 
                 pluck_intensity=0.5, pluck_position=0.5, decay_factor=0.99,
                 stretch_factor=1.0, rng_seed=None):
        """
        Initialize Karplus-Strong synthesizer.
        
        Args:
            frequency (float): Target pitch in Hz. Must be > 0.
            sample_rate (float): Audio sample rate in Hz. Default 44100.
            duration (float): Duration of synthesis in seconds. Default 1.0.
            pluck_intensity (float): Amplitude and hardness of pluck in [0, 1].
                Lower values = softer/duller plucks. Default 0.5.
            pluck_position (float): Pick position in [0, 1].
                0 = near bridge (bright), 1 = near nut (dark). Default 0.5.
            decay_factor (float): Feedback coefficient in [0, 1). Typical 0.99.
                Controls overall sustain length. Default 0.99.
            stretch_factor (float): Decay stretching factor > 0. Default 1.0.
                S > 1 = longer sustain; S < 1 = shorter sustain.
            rng_seed (int): Random seed for reproducible excitation. Default None.
        
        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        # Validate frequency
        if frequency <= 0:
            raise ValueError(f"Frequency must be > 0, got {frequency}")
        
        # Validate sample rate
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be > 0, got {sample_rate}")
        
        # Validate duration
        if duration <= 0:
            raise ValueError(f"Duration must be > 0, got {duration}")
        
        # Validate dynamic parameters
        if not (0.0 <= pluck_intensity <= 1.0):
            raise ValueError(f"Pluck intensity must be in [0, 1], got {pluck_intensity}")
        
        if not (0.0 <= pluck_position <= 1.0):
            raise ValueError(f"Pluck position must be in [0, 1], got {pluck_position}")
        
        if not (0.0 <= decay_factor < 1.0):
            raise ValueError(f"Decay factor must be in [0, 1), got {decay_factor}")
        
        if stretch_factor <= 0:
            raise ValueError(f"Stretch factor must be > 0, got {stretch_factor}")
        
        # Store synthesis parameters
        self.frequency = float(frequency)
        self.sample_rate = float(sample_rate)
        self.duration = float(duration)
        self.pluck_intensity = float(pluck_intensity)
        self.pluck_position = float(pluck_position)
        self.decay_factor = float(decay_factor)
        self.stretch_factor = float(stretch_factor)
        
        # Compute derived parameters
        self.num_samples = int(np.round(self.sample_rate * self.duration))
        
        # Delay line length determines pitch
        self.delay_length = int(np.round(self.sample_rate / self.frequency))
        if self.delay_length < 1:
            raise ValueError(
                f"Delay length would be {self.delay_length} (freq {frequency} Hz "
                f"at {sample_rate} sample rate). Choose lower frequency or higher sample rate."
            )
        
        # Initialize components
        self.excitation = NoiseExcitation(amplitude=self.pluck_intensity, rng_seed=rng_seed)
        self.delay_line = DelayLine(self.delay_length)
        
        # Allpass filter for fractional delay (to tune between integer sample delays)
        self.allpass = AllpassFilter(coefficient=self._compute_allpass_coefficient())
        
        # Lowpass filter for dynamic bandwidth (pluck intensity affects brightness)
        self.lowpass = LowpassFilter(alpha=self._compute_lowpass_alpha())
        
        # Comb filter for pick position effects
        comb_delay = max(1, int(np.round(self.delay_length * self.pluck_position)))
        self.comb = CombFilter(delay_length=comb_delay, feedback=0.5)
        
        # Decay stretcher for independent sustain control
        self.stretcher = DecayStretcher(stretch_factor=self.stretch_factor)
        
        # Adjusted feedback coefficient after stretching
        self.adjusted_feedback = self.stretcher.adjust_feedback(self.decay_factor)
        
        # Safety: ensure adjusted feedback is in valid range to prevent instability
        # Feedback must be < 1.0 for stability, and we add margin for numerical safety
        if self.adjusted_feedback >= 0.999:
            self.adjusted_feedback = 0.999
        elif self.adjusted_feedback < 0.0:
            self.adjusted_feedback = 0.0
    
    def _compute_allpass_coefficient(self):
        """
        Compute allpass filter coefficient for fractional delay tuning.
        
        The delay line provides integer-sample delay. To achieve precise pitch tuning,
        a fractional delay is computed and represented by the allpass coefficient.
        
        Returns:
            float: Allpass coefficient in [-1, 1].
        """
        # Compute the fractional delay component
        ideal_delay = self.sample_rate / self.frequency
        integer_delay = float(self.delay_length)
        frac_delay = ideal_delay - integer_delay
        
        # Cap fractional delay to avoid instability (must be in range (-0.5, 0.5))
        frac_delay = np.clip(frac_delay, -0.4999, 0.4999)
        
        # Convert fractional delay to allpass coefficient
        # coeff = (1 - frac) / (1 + frac)
        if abs(frac_delay) < 1e-10:
            return 0.0
        
        coeff = (1.0 - frac_delay) / (1.0 + frac_delay)
        
        # Ensure coefficient is in valid range
        coeff = np.clip(coeff, -0.99, 0.99)
        return float(coeff)
    
    def _compute_lowpass_alpha(self):
        """
        Compute lowpass filter coefficient based on pluck intensity.
        
        Softer plucks (lower intensity) should have more damping initially.
        Harder plucks (higher intensity) should have less damping.
        
        Returns:
            float: Lowpass alpha coefficient in [0.3, 0.95].
        """
        # Map pluck_intensity [0, 1] to alpha [0.3, 0.95]
        # intensity=0 (soft) → alpha=0.3 (heavily damped)
        # intensity=1 (hard) → alpha=0.95 (minimally damped)
        alpha = 0.3 + 0.65 * self.pluck_intensity
        return float(np.clip(alpha, 0.3, 0.95))
    
    def synthesize(self):
        """
        Synthesize and return the complete audio signal.
        
        Generates the full note by:
        1. Exciting the delay line with filtered noise
        2. Running the feedback loop for the specified duration
        3. Applying allpass (fractional delay), lowpass, comb, and decay stretching
        
        Returns:
            np.ndarray: Audio samples, shape (num_samples,), dtype float64.
                Values are in roughly [-1, 1] but may exceed these bounds slightly.
        """
        # Reset all internal state before synthesis
        self.reset()
        
        output = np.zeros(self.num_samples, dtype=np.float64)
        
        # Get excitation signal (noise burst)
        excitation_duration = min(0.01, self.duration)  # 10ms or less
        excitation_signal = self.excitation.generate(self.sample_rate, duration=excitation_duration)
        excitation_idx = 0
        
        # Synthesis loop
        for n in range(self.num_samples):
            # Get excitation sample (use it for the first chunk, then zeros)
            if excitation_idx < len(excitation_signal):
                exc_sample = excitation_signal[excitation_idx]
                excitation_idx += 1
            else:
                exc_sample = 0.0
            
            # Read from delay line (this sample was written last iteration)
            delayed_sample = self.delay_line.read()
            
            # Apply filters in order: allpass -> lowpass -> comb
            filtered = self.allpass.process(delayed_sample)
            filtered = self.lowpass.process(filtered)
            filtered = self.comb.process(filtered)
            
            # Apply feedback with decay stretching
            feedback_sample = filtered * self.adjusted_feedback
            
            # Combine excitation with feedback and write back to delay line
            input_sample = exc_sample + feedback_sample
            self.delay_line.write(input_sample)
            
            # Output the current filtered sample
            output[n] = filtered
        
        # Hard clipping to prevent extreme values (should rarely happen with proper parameters)
        max_allowed = 100.0
        output = np.clip(output, -max_allowed, max_allowed)
        
        return output
    
    def reset(self):
        """Reset all internal state to prepare for a new synthesis run."""
        self.delay_line.reset()
        self.allpass.reset()
        self.lowpass.reset()
        self.comb.reset()
