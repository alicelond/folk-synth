"""
Filter module for Karplus-Strong synthesis.

Implements various filters used in the Karplus-Strong algorithm including:
- AllpassFilter: Fractional delay tuning per Jaffe & Smith (1983)
- LowpassFilter: Dynamic bandwidth control for pluck intensity variation
- CombFilter: Pick position simulation via comb filtering
- DecayStretcher: Independent sustain control via stretch factor
"""

import numpy as np


class AllpassFilter:
    """
    First-order allpass filter for fractional delay line tuning.
    
    Based on Jaffe & Smith (1983). Allows precise pitch control beyond the
    integer-sample quantization of the delay line by using a first-order
    allpass filter to add fractional delay.
    
    The filter has unity gain across all frequencies but introduces a frequency-
    dependent phase shift that effectively adds fractional delay.
    
    Attributes:
        coefficient (float): Allpass coefficient determining fractional delay.
            Typical range: [-1, 1]. formula: coefficient = (1 - frac_delay) / (1 + frac_delay)
            where frac_delay is the desired fractional delay in samples (0 < frac_delay < 1).
        state (float): Internal filter state for maintaining continuity.
    """
    
    def __init__(self, coefficient=0.0):
        """
        Initialize allpass filter with a specified coefficient.
        
        Args:
            coefficient (float): Allpass coefficient. Default 0.0 (no fractional delay).
                Typical range: [-0.5, 0.5] for stable fractional delays of 0.25-0.75 samples.
        
        Raises:
            ValueError: If |coefficient| >= 1 (filter becomes unstable).
        """
        if abs(coefficient) >= 1.0:
            raise ValueError(f"Allpass coefficient must have |coeff| < 1, got {coefficient}")
        
        self.coefficient = float(coefficient)
        self.state = 0.0
    
    def process(self, sample):
        """
        Process a single sample through the allpass filter.
        
        Args:
            sample (float): Input sample.
        
        Returns:
            float: Filtered output sample.
        """
        output = -sample + self.coefficient * (sample + self.state)
        self.state = output
        return output
    
    def set_coefficient(self, coefficient):
        """
        Change the allpass coefficient (and thus fractional delay).
        
        Args:
            coefficient (float): New coefficient value, |coeff| < 1.
        
        Raises:
            ValueError: If |coefficient| >= 1.
        """
        if abs(coefficient) >= 1.0:
            raise ValueError(f"Allpass coefficient must have |coeff| < 1, got {coefficient}")
        self.coefficient = float(coefficient)
    
    def reset(self):
        """Reset internal state to zero."""
        self.state = 0.0


class LowpassFilter:
    """
    Simple first-order IIR lowpass filter for dynamic bandwidth control.
    
    Used to shape the initial excitation for pluck dynamics (soft vs. hard plucks)
    and to dampen higher frequencies during the sustain phase.
    
    Transfer function: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    Cutoff frequency (Hz) ≈ (alpha / (2π)) * sample_rate
    
    Attributes:
        alpha (float): Filter coefficient in [0, 1].
            alpha=1: no filtering (all frequencies pass)
            alpha=0: total attenuation
        state (float): Previous output sample for IIR recursion.
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize lowpass filter.
        
        Args:
            alpha (float): Filter coefficient in [0, 1]. Default 1.0 (no filtering).
        
        Raises:
            ValueError: If alpha not in [0, 1].
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        self.alpha = float(alpha)
        self.state = 0.0
    
    def process(self, sample):
        """
        Process a single sample through lowpass filter.
        
        Args:
            sample (float): Input sample.
        
        Returns:
            float: Filtered output sample.
        """
        self.state = self.alpha * sample + (1.0 - self.alpha) * self.state
        return self.state
    
    def set_alpha(self, alpha):
        """
        Change the filter coefficient.
        
        Args:
            alpha (float): New coefficient in [0, 1].
        
        Raises:
            ValueError: If alpha not in [0, 1].
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
    
    def reset(self):
        """Reset internal state to zero."""
        self.state = 0.0


class CombFilter:
    """
    Comb filter to simulate pick position effects.
    
    The frequency at which the string is plucked affects its timbral characteristics.
    A comb filter (which is essentially a delayed copy summed with the direct signal)
    creates notches in the frequency spectrum, simulating the difference in harmonic
    content based on pluck position.
    
    Delay proportional to distance from bridge: longer delay (pluck near nut) vs.
    shorter delay (pluck near bridge).
    
    Attributes:
        delay_length (int): Comb filter delay in samples.
        buffer (np.ndarray): Delay buffer for the comb filter.
        index (int): Current write position in the circular buffer.
        feedback (float): Feedback coefficient for the comb filter.
    """
    
    def __init__(self, delay_length, feedback=0.5):
        """
        Initialize comb filter.
        
        Args:
            delay_length (int): Delay in samples (typically 1/4 to 1/2 of string delay).
                Must be > 0.
            feedback (float): Feedback amount in [0, 1]. Default 0.5.
        
        Raises:
            ValueError: If delay_length <= 0 or feedback not in [0, 1].
        """
        if not isinstance(delay_length, (int, np.integer)) or delay_length <= 0:
            raise ValueError(f"Delay length must be positive integer, got {delay_length}")
        if not (0.0 <= feedback <= 1.0):
            raise ValueError(f"Feedback must be in [0, 1], got {feedback}")
        
        self.delay_length = int(delay_length)
        self.buffer = np.zeros(self.delay_length, dtype=np.float64)
        self.index = 0
        self.feedback = float(feedback)
    
    def process(self, sample):
        """
        Process a single sample through the comb filter.
        
        Args:
            sample (float): Input sample.
        
        Returns:
            float: Filtered output (direct + delayed feedback).
        """
        delayed = self.buffer[self.index]
        output = sample + self.feedback * delayed
        self.buffer[self.index] = sample
        self.index = (self.index + 1) % self.delay_length
        return output
    
    def reset(self):
        """Reset the comfilter buffer and index."""
        self.buffer.fill(0.0)
        self.index = 0


class DecayStretcher:
    """
    Energy decay stretcher for independent sustain control.
    
    The stretching factor S allows independent control of the sustain length
    from the initial brightness of the note. Higher S values stretch the decay,
    simulating different string materials or body sizes.
    
    This is typically implemented by modifying the feedback coefficient:
    adjusted_feedback = feedback^(1/S)
    
    Attributes:
        stretch_factor (float): Stretching factor S > 0. Default 1.0 (no stretching).
            S > 1: longer sustain (stretched decay)
            S < 1: shorter sustain (compressed decay)
    """
    
    def __init__(self, stretch_factor=1.0):
        """
        Initialize decay stretcher.
        
        Args:
            stretch_factor (float): Decay stretching factor S > 0. Default 1.0.
        
        Raises:
            ValueError: If stretch_factor <= 0.
        """
        if stretch_factor <= 0:
            raise ValueError(f"Stretch factor must be > 0, got {stretch_factor}")
        
        self.stretch_factor = float(stretch_factor)
    
    def adjust_feedback(self, feedback):
        """
        Compute adjusted feedback coefficient given a base feedback value.
        
        Formula: adjusted = feedback^(1 / stretch_factor)
        
        Args:
            feedback (float): Original feedback coefficient in [0, 1).
        
        Returns:
            float: Adjusted feedback coefficient that will be stretched according
                to the stretch factor.
                
                With stretch > 1: adjusted > feedback (longer sustain)
                With stretch < 1: adjusted < feedback (shorter sustain)
                With stretch = 1: adjusted = feedback (no change)
        
        Raises:
            ValueError: If feedback not in [0, 1) or results in invalid computation.
        """
        if not (0.0 <= feedback < 1.0):
            raise ValueError(f"Feedback must be in [0, 1), got {feedback}")
        
        if feedback == 0:
            return 0.0
        
        # adjusted = feedback^(1/stretch)
        # stretch=2: 0.99^0.5 = 0.995 (more feedback → longer sustain)
        # stretch=0.5: 0.99^2 = 0.9801 (less feedback → shorter sustain)
        adjusted = np.power(feedback, 1.0 / self.stretch_factor)
        return float(adjusted)
    
    def set_stretch_factor(self, stretch_factor):
        """
        Update the stretch factor.
        
        Args:
            stretch_factor (float): New stretch factor > 0.
        
        Raises:
            ValueError: If stretch_factor <= 0.
        """
        if stretch_factor <= 0:
            raise ValueError(f"Stretch factor must be > 0, got {stretch_factor}")
        self.stretch_factor = float(stretch_factor)
