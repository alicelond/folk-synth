"""
Unit tests for the excitation module.
"""

import pytest
import numpy as np
from core.excitation import NoiseExcitation


class TestNoiseExcitation:
    """Tests for NoiseExcitation class."""
    
    def test_init_default(self):
        """Test default initialization."""
        exc = NoiseExcitation()
        assert exc.amplitude == 1.0
        assert exc.length is None
    
    def test_init_custom_amplitude(self):
        """Test initialization with custom amplitude."""
        exc = NoiseExcitation(amplitude=0.5)
        assert exc.amplitude == 0.5
    
    def test_init_invalid_amplitude(self):
        """Amplitude must be > 0."""
        with pytest.raises(ValueError):
            NoiseExcitation(amplitude=0)
        with pytest.raises(ValueError):
            NoiseExcitation(amplitude=-0.5)
    
    def test_init_with_length(self):
        """Test initialization with fixed length."""
        exc = NoiseExcitation(length=1000)
        assert exc.length == 1000
    
    def test_init_with_seed(self):
        """Test reproducibility with seed."""
        exc1 = NoiseExcitation(rng_seed=42)
        exc2 = NoiseExcitation(rng_seed=42)
        
        noise1 = exc1.generate(sample_rate=44100, duration=0.1)
        noise2 = exc2.generate(sample_rate=44100, duration=0.1)
        
        np.testing.assert_array_equal(noise1, noise2)
    
    def test_generate_with_duration(self):
        """Test noise generation with specified duration."""
        exc = NoiseExcitation(amplitude=1.0)
        sr = 44100
        duration = 0.01  # 10 ms
        
        noise = exc.generate(sample_rate=sr, duration=duration)
        
        expected_length = int(np.round(sr * duration))
        assert len(noise) == expected_length
        assert noise.dtype == np.float64
    
    def test_generate_with_length(self):
        """Test noise generation using fixed length."""
        exc = NoiseExcitation(amplitude=1.0, length=1000)
        noise = exc.generate(sample_rate=44100)
        
        assert len(noise) == 1000
        assert noise.dtype == np.float64
    
    def test_generate_requires_length_or_duration(self):
        """Error if neither length nor duration is provided."""
        exc = NoiseExcitation()  # No length
        with pytest.raises(ValueError):
            exc.generate(sample_rate=44100)  # No duration
    
    def test_generate_invalid_sample_rate(self):
        """Sample rate must be > 0."""
        exc = NoiseExcitation()
        with pytest.raises(ValueError):
            exc.generate(sample_rate=0, duration=1.0)
        with pytest.raises(ValueError):
            exc.generate(sample_rate=-44100, duration=1.0)
    
    def test_generate_invalid_duration(self):
        """Duration must be >= 0."""
        exc = NoiseExcitation()
        with pytest.raises(ValueError):
            exc.generate(sample_rate=44100, duration=-1.0)
    
    def test_noise_amplitude_range(self):
        """Noise values should be within [-amplitude, amplitude]."""
        amplitude = 0.5
        exc = NoiseExcitation(amplitude=amplitude)
        noise = exc.generate(sample_rate=44100, duration=1.0)
        
        assert np.all(noise >= -amplitude)
        assert np.all(noise <= amplitude)
    
    def test_noise_is_random(self):
        """Generated noise should not be constant."""
        exc = NoiseExcitation()
        noise = exc.generate(sample_rate=44100, duration=0.1)
        
        # Should have variance
        assert np.var(noise) > 0
        # Not all samples should be identical
        assert len(np.unique(noise)) > 1
    
    def test_noise_statistical_properties(self):
        """Test that noise is approximately uniform."""
        exc = NoiseExcitation(amplitude=1.0, rng_seed=123)
        noise = exc.generate(sample_rate=44100, duration=1.0)
        
        # For uniform distribution [-1, 1], mean ≈ 0
        mean = np.mean(noise)
        assert abs(mean) < 0.1  # Loose tolerance for randomness
        
        # For uniform [-1, 1], std ≈ sqrt((2^2)/12) ≈ 0.577
        std = np.std(noise)
        assert 0.4 < std < 0.7  # Loose tolerance
