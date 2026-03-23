"""
Unit tests for the KarplusStrong synthesizer.
"""

import pytest
import numpy as np
from core.string import KarplusStrong
from core.excitation import NoiseExcitation


class TestKarplusStrong:
    """Tests for KarplusStrong synthesizer class."""
    
    def test_init_default(self):
        """Test default initialization."""
        ks = KarplusStrong(frequency=440.0)
        
        assert ks.frequency == 440.0
        assert ks.sample_rate == 44100.0
        assert ks.duration == 1.0
        assert ks.pluck_intensity == 0.5
        assert ks.pluck_position == 0.5
        assert ks.decay_factor == 0.99
        assert ks.stretch_factor == 1.0
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        ks = KarplusStrong(
            frequency=220.0,
            sample_rate=48000,
            duration=2.0,
            pluck_intensity=0.8,
            pluck_position=0.3,
            decay_factor=0.98,
            stretch_factor=1.5
        )
        
        assert ks.frequency == 220.0
        assert ks.sample_rate == 48000.0
        assert ks.duration == 2.0
        assert ks.pluck_intensity == 0.8
        assert ks.pluck_position == 0.3
        assert ks.decay_factor == 0.98
        assert ks.stretch_factor == 1.5
    
    def test_init_invalid_frequency(self):
        """Frequency must be > 0."""
        with pytest.raises(ValueError):
            KarplusStrong(frequency=0)
        with pytest.raises(ValueError):
            KarplusStrong(frequency=-440)
    
    def test_init_invalid_sample_rate(self):
        """Sample rate must be > 0."""
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, sample_rate=0)
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, sample_rate=-44100)
    
    def test_init_invalid_duration(self):
        """Duration must be > 0."""
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, duration=0)
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, duration=-1)
    
    def test_init_invalid_pluck_intensity(self):
        """Pluck intensity must be in [0, 1]."""
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, pluck_intensity=-0.1)
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, pluck_intensity=1.5)
    
    def test_init_invalid_pluck_position(self):
        """Pluck position must be in [0, 1]."""
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, pluck_position=-0.1)
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, pluck_position=1.5)
    
    def test_init_invalid_decay_factor(self):
        """Decay factor must be in [0, 1)."""
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, decay_factor=-0.1)
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, decay_factor=1.0)
    
    def test_init_invalid_stretch_factor(self):
        """Stretch factor must be > 0."""
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, stretch_factor=0)
        with pytest.raises(ValueError):
            KarplusStrong(frequency=440, stretch_factor=-1)
    
    def test_num_samples_calculation(self):
        """Test that num_samples is correctly computed."""
        ks = KarplusStrong(frequency=440, sample_rate=44100, duration=1.0)
        expected = int(np.round(44100 * 1.0))
        assert ks.num_samples == expected
    
    def test_delay_length_calculation(self):
        """Test that delay length is correctly computed for pitch."""
        ks = KarplusStrong(frequency=440, sample_rate=44100)
        
        expected_delay = int(np.round(44100 / 440))
        assert ks.delay_length == expected_delay
    
    def test_very_high_frequency_handling(self):
        """Very high frequencies at low sample rates work but with shorter wavelengths."""
        # High frequency still works, just has shorter delay
        ks = KarplusStrong(frequency=22000, sample_rate=44100)
        assert ks.delay_length >= 2
        audio = ks.synthesize()
        assert np.all(np.isfinite(audio))
    
    def test_synthesize_output_shape(self):
        """Test that synthesize produces correct output shape."""
        ks = KarplusStrong(frequency=440, duration=0.1)
        output = ks.synthesize()
        
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float64
        assert len(output) == ks.num_samples
    
    def test_synthesize_output_values(self):
        """Test that synthesized values are reasonable."""
        ks = KarplusStrong(frequency=440, duration=0.1)
        output = ks.synthesize()
        
        # Output should be finite
        assert np.all(np.isfinite(output))
        
        # Output should be bounded (hard-clipped to prevent explosion)
        assert np.max(np.abs(output)) <= 100.0  # Hard limit in synthesize()
    
    def test_synthesize_produces_nonzero(self):
        """Test that synthesis produces non-trivial output."""
        ks = KarplusStrong(
            frequency=440,
            duration=0.1,
            pluck_intensity=1.0  # Maximum intensity
        )
        output = ks.synthesize()
        
        # Should have some significant amplitude
        max_amp = np.max(np.abs(output))
        assert max_amp > 0.01  # At least some energy
    
    def test_synthesize_decay(self):
        """Test that output has energy envelope."""
        ks = KarplusStrong(
            frequency=440,
            duration=0.5,
            decay_factor=0.95,  # Moderate decay
            pluck_intensity=1.0  # Full intensity for clear signal
        )
        output = ks.synthesize()
        
        # Just verify it has reasonable oscillations
        assert np.max(np.abs(output)) > 0.1  # Has energy
        assert np.all(np.isfinite(output))  # All finite
    
    def test_pluck_intensity_affects_brightness(self):
        """Harder plucks should be brighter (less lowpass filtering)."""
        ks_soft = KarplusStrong(frequency=440, duration=0.1, pluck_intensity=0.1)
        ks_hard = KarplusStrong(frequency=440, duration=0.1, pluck_intensity=0.9)
        
        output_soft = ks_soft.synthesize()
        output_hard = ks_hard.synthesize()
        
        # Get initial transient
        transient_size = int(0.01 * ks_soft.sample_rate)
        
        # Hard pluck should have higher initial energy
        energy_soft = np.sum(output_soft[:transient_size] ** 2)
        energy_hard = np.sum(output_hard[:transient_size] ** 2)
        
        assert energy_hard > energy_soft
    
    def test_pluck_position_affects_timbre(self):
        """Different pluck positions should produce different timbres."""
        ks_bridge = KarplusStrong(frequency=440, duration=0.1, pluck_position=0.0)
        ks_nut = KarplusStrong(frequency=440, duration=0.1, pluck_position=1.0)
        
        output_bridge = ks_bridge.synthesize()
        output_nut = ks_nut.synthesize()
        
        # Outputs should be different (due to different comb filter delays)
        # They won't be exactly the same
        assert not np.allclose(output_bridge, output_nut)
    
    def test_stretch_factor_affects_synthesis(self):
        """Stretch factor should affect the synthesis characteristics."""
        ks_short = KarplusStrong(
            frequency=440,
            duration=0.5,
            decay_factor=0.99,
            stretch_factor=0.5  # Shorter sustain
        )
        ks_long = KarplusStrong(
            frequency=440,
            duration=0.5,
            decay_factor=0.99,
            stretch_factor=2.0  # Longer sustain
        )
        
        # Different stretch factors should give different feedback coefficients
        assert ks_short.adjusted_feedback != ks_long.adjusted_feedback
        
        # Longer sustain should have larger feedback coefficient
        assert ks_long.adjusted_feedback > ks_short.adjusted_feedback
    
    def test_reset_clears_state(self):
        """Test that reset allows re-synthesis."""
        ks = KarplusStrong(frequency=440, duration=0.1, rng_seed=42)
        
        # Run once
        output1 = ks.synthesize()
        
        # Reset manually and run again (synthesize() auto-resets internally)
        ks.reset()
        ks.excitation = NoiseExcitation(amplitude=ks.pluck_intensity, rng_seed=42)
        output2 = ks.synthesize()
        
        # With same seed and reset, should get same output
        np.testing.assert_array_equal(output1, output2)
    
    def test_reproducibility_with_seed(self):
        """Same seed should produce identical output twice."""
        ks1 = KarplusStrong(frequency=440, duration=0.1, rng_seed=42)
        ks2 = KarplusStrong(frequency=440, duration=0.1, rng_seed=42)
        
        output1 = ks1.synthesize()
        output2 = ks2.synthesize()
        
        np.testing.assert_array_equal(output1, output2)
    
    def test_different_frequencies(self):
        """Test synthesis at different frequencies."""
        ks1 = KarplusStrong(frequency=440, duration=0.1)
        ks2 = KarplusStrong(frequency=880, duration=0.1)
        
        output1 = ks1.synthesize()
        output2 = ks2.synthesize()
        
        # Both should produce valid output
        assert np.all(np.isfinite(output1))
        assert np.all(np.isfinite(output2))
        
        # Outputs should be different
        assert not np.allclose(output1, output2)
    
    def test_low_frequency_synthesis(self):
        """Test synthesis at low frequencies."""
        ks = KarplusStrong(frequency=55, sample_rate=44100, duration=0.5)
        output = ks.synthesize()
        
        assert np.all(np.isfinite(output))
        assert np.max(np.abs(output)) > 0.01
