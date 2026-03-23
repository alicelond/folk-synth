"""
Excitation module for Karplus-Strong synthesis.

Generates the initial noise burst that excites the string simulation.
Based on Karplus & Strong (1983) - the excitation signal is typically
a short burst of white noise that is fed into the delay line.
"""

import numpy as np


class NoiseExcitation:
    """
    Generates a noise excitation signal for the Karplus-Strong algorithm.
    
    Attributes:
        amplitude (float): Maximum amplitude of the noise signal.
        length (int): Number of samples in the noise burst.
        rng (np.random.Generator): Random number generator for reproducibility.
    """
    
    def __init__(self, amplitude=1.0, length=None, rng_seed=None):
        """
        Initialize the noise excitation generator.
        
        Args:
            amplitude (float): Maximum amplitude of the noise burst (default 1.0).
                Must be > 0.
            length (int): Number of samples to generate. If None, uses sample_rate
                parameter passed to generate(). Default None.
            rng_seed (int): Seed for random number generator for reproducibility.
                If None, uses system randomness. Default None.
        
        Raises:
            ValueError: If amplitude <= 0.
        """
        if amplitude <= 0:
            raise ValueError(f"Amplitude must be > 0, got {amplitude}")
        
        self.amplitude = amplitude
        self.length = length
        self.rng = np.random.default_rng(rng_seed)
    
    def generate(self, sample_rate, duration=None):
        """
        Generate a white noise excitation burst.
        
        Args:
            sample_rate (float): Audio sample rate in Hz (e.g., 44100).
            duration (float): Duration of the noise burst in seconds. If None,
                uses self.length samples. If self.length is also None, raises error.
                Default None.
        
        Returns:
            np.ndarray: 1D array of noise samples with shape (num_samples,),
                dtype float64, values in range [-amplitude, amplitude].
        
        Raises:
            ValueError: If both duration and self.length are None.
            ValueError: If sample_rate <= 0 or duration < 0.
        """
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be > 0, got {sample_rate}")
        
        # Determine number of samples
        if duration is not None:
            if duration < 0:
                raise ValueError(f"Duration must be >= 0, got {duration}")
            num_samples = int(np.round(sample_rate * duration))
        elif self.length is not None:
            num_samples = self.length
        else:
            raise ValueError("Either duration or self.length must be specified")
        
        # Generate uniform white noise in [-amplitude, amplitude]
        noise = self.rng.uniform(-self.amplitude, self.amplitude, size=num_samples)
        
        return noise.astype(np.float64)
