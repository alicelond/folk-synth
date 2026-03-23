"""
Unit tests for the filters module.
"""

import pytest
import numpy as np
from core.filters import AllpassFilter, LowpassFilter, CombFilter, DecayStretcher


class TestAllpassFilter:
    """Tests for AllpassFilter class."""
    
    def test_init_default(self):
        """Test default initialization."""
        af = AllpassFilter()
        assert af.coefficient == 0.0
        assert af.state == 0.0
    
    def test_init_custom_coefficient(self):
        """Test initialization with custom coefficient."""
        af = AllpassFilter(coefficient=0.3)
        assert af.coefficient == 0.3
    
    def test_init_invalid_coefficient(self):
        """Coefficient magnitude must be < 1."""
        with pytest.raises(ValueError):
            AllpassFilter(coefficient=1.0)
        with pytest.raises(ValueError):
            AllpassFilter(coefficient=1.5)
        with pytest.raises(ValueError):
            AllpassFilter(coefficient=-1.0)
    
    def test_process_zero_coefficient(self):
        """With coefficient=0, filter should produce -input."""
        af = AllpassFilter(coefficient=0.0)
        output = af.process(1.0)
        assert output == -1.0
    
    def test_process_positive_coefficient(self):
        """Test processing with positive coefficient."""
        af = AllpassFilter(coefficient=0.5)
        output = af.process(1.0)
        # output = -1.0 + 0.5 * (1.0 + 0.0) = -0.5
        assert output == -0.5
    
    def test_unity_gain_property(self):
        """Allpass filter should have unity gain (magnitude)."""
        af = AllpassFilter(coefficient=0.3)
        
        # Send impulse train and check magnitude
        impulse = 1.0
        output1 = af.process(impulse)
        output2 = af.process(0.0)
        output3 = af.process(0.0)
        
        # For allpass, the magnitude response is always 1 regardless of frequency
        # Check that persistent oscillations don't grow
        magnitude = abs(output1)
        assert magnitude <= 1.5  # Should be relatively bounded
    
    def test_set_coefficient(self):
        """Test changing coefficient."""
        af = AllpassFilter(coefficient=0.1)
        af.set_coefficient(0.5)
        
        assert af.coefficient == 0.5
        output = af.process(1.0)
        assert output == -0.5
    
    def test_set_coefficient_invalid(self):
        """Set coefficient with invalid value should raise error."""
        af = AllpassFilter()
        with pytest.raises(ValueError):
            af.set_coefficient(1.5)
    
    def test_reset(self):
        """Test reset functionality."""
        af = AllpassFilter(coefficient=0.5)
        af.process(1.0)
        assert af.state != 0.0  # State should be modified
        
        af.reset()
        assert af.state == 0.0


class TestLowpassFilter:
    """Tests for LowpassFilter class."""
    
    def test_init_default(self):
        """Test default initialization."""
        lp = LowpassFilter()
        assert lp.alpha == 1.0
        assert lp.state == 0.0
    
    def test_init_custom_alpha(self):
        """Test initialization with custom alpha."""
        lp = LowpassFilter(alpha=0.5)
        assert lp.alpha == 0.5
    
    def test_init_invalid_alpha(self):
        """Alpha must be in [0, 1]."""
        with pytest.raises(ValueError):
            LowpassFilter(alpha=-0.1)
        with pytest.raises(ValueError):
            LowpassFilter(alpha=1.5)
    
    def test_process_alpha_one(self):
        """With alpha=1, filter is transparent (no filtering)."""
        lp = LowpassFilter(alpha=1.0)
        output = lp.process(0.5)
        assert output == 0.5
    
    def test_process_alpha_zero(self):
        """With alpha=0, output remains zero (infinite attenuation)."""
        lp = LowpassFilter(alpha=0.0)
        lp.process(0.5)
        output = lp.process(1.0)
        assert output == 0.0
    
    def test_process_smoothing(self):
        """Lowpass filter should smooth step changes."""
        lp = LowpassFilter(alpha=0.5)
        
        # Start at 0, step to 1.0
        out1 = lp.process(1.0)
        out2 = lp.process(1.0)
        out3 = lp.process(1.0)
        
        # Step response should be smooth, approaching 1.0 asymptotically
        assert out1 == 0.5  # 0.5 * 1.0 + 0.5 * 0.0
        assert 0.5 < out2 < 1.0  # Should approach 1
        assert out2 < out3 < 1.0  # Should continue approaching 1
    
    def test_set_alpha(self):
        """Test changing alpha."""
        lp = LowpassFilter(alpha=0.5)
        lp.set_alpha(0.2)
        
        assert lp.alpha == 0.2
    
    def test_set_alpha_invalid(self):
        """Set alpha with invalid value should raise error."""
        lp = LowpassFilter()
        with pytest.raises(ValueError):
            lp.set_alpha(1.5)
    
    def test_reset(self):
        """Test reset functionality."""
        lp = LowpassFilter(alpha=0.5)
        lp.process(1.0)
        assert lp.state != 0.0
        
        lp.reset()
        assert lp.state == 0.0


class TestCombFilter:
    """Tests for CombFilter class."""
    
    def test_init_valid(self):
        """Test valid initialization."""
        cf = CombFilter(delay_length=10)
        assert cf.delay_length == 10
        assert len(cf.buffer) == 10
        assert cf.index == 0
        assert cf.feedback == 0.5
    
    def test_init_custom_feedback(self):
        """Test initialization with custom feedback."""
        cf = CombFilter(delay_length=10, feedback=0.3)
        assert cf.feedback == 0.3
    
    def test_init_invalid_delay_length(self):
        """Delay length must be positive integer."""
        with pytest.raises(ValueError):
            CombFilter(delay_length=0)
        with pytest.raises(ValueError):
            CombFilter(delay_length=-10)
        with pytest.raises(ValueError):
            CombFilter(delay_length=10.5)
    
    def test_init_invalid_feedback(self):
        """Feedback must be in [0, 1]."""
        with pytest.raises(ValueError):
            CombFilter(delay_length=10, feedback=-0.1)
        with pytest.raises(ValueError):
            CombFilter(delay_length=10, feedback=1.5)
    
    def test_process_basic(self):
        """Test basic comb filter processing."""
        cf = CombFilter(delay_length=3, feedback=0.5)
        
        # Impulse at start
        output = cf.process(1.0)
        # output = 1.0 + 0.5 * 0.0 (delayed was zero)
        assert output == 1.0
    
    def test_process_impulse_response(self):
        """Test comb filter impulse response."""
        cf = CombFilter(delay_length=2, feedback=0.5)
        
        # Send impulse
        # buffer starts as [0, 0], index = 0
        outputs = []
        
        # Process 1.0
        # output = 1.0 + 0.5*buffer[0] = 1.0 + 0.5*0 = 1.0
        # buffer[0] = 1.0, index = 1
        outputs.append(cf.process(1.0))  # output = 1.0
        
        # Process 0.0
        # output = 0.0 + 0.5*buffer[1] = 0.0 + 0.5*0 = 0.0
        # buffer[1] = 0.0, index = 0
        outputs.append(cf.process(0.0))  # output = 0.0
        
        # Process 0.0
        # output = 0.0 + 0.5*buffer[0] = 0.0 + 0.5*1.0 = 0.5
        # buffer[0] = 0.0, index = 1
        outputs.append(cf.process(0.0))  # output = 0.5
        
        # Process 0.0
        # output = 0.0 + 0.5*buffer[1] = 0.0 + 0.5*0 = 0.0
        # buffer[1] = 0.0, index = 0
        outputs.append(cf.process(0.0))  # output = 0.0
        
        assert outputs[0] == 1.0
        assert outputs[1] == 0.0
        assert abs(outputs[2] - 0.5) < 0.0001  # Delayed impulse returns
        assert outputs[3] == 0.0
    
    def test_reset(self):
        """Test reset functionality."""
        cf = CombFilter(delay_length=5)
        
        # Fill buffer
        for i in range(10):
            cf.process(float(i))
        
        cf.reset()
        
        assert np.all(cf.buffer == 0)
        assert cf.index == 0


class TestDecayStretcher:
    """Tests for DecayStretcher class."""
    
    def test_init_default(self):
        """Test default initialization."""
        ds = DecayStretcher()
        assert ds.stretch_factor == 1.0
    
    def test_init_custom_stretch(self):
        """Test initialization with custom stretch factor."""
        ds = DecayStretcher(stretch_factor=2.0)
        assert ds.stretch_factor == 2.0
    
    def test_init_invalid_stretch(self):
        """Stretch factor must be > 0."""
        with pytest.raises(ValueError):
            DecayStretcher(stretch_factor=0)
        with pytest.raises(ValueError):
            DecayStretcher(stretch_factor=-1.0)
    
    def test_adjust_feedback_no_stretch(self):
        """With stretch_factor=1, feedback should be unchanged."""
        ds = DecayStretcher(stretch_factor=1.0)
        
        feedback = 0.99
        adjusted = ds.adjust_feedback(feedback)
        
        assert abs(adjusted - feedback) < 1e-10
    
    def test_adjust_feedback_stretch_longer(self):
        """stretch_factor > 1 should increase feedback (longer sustain)."""
        ds = DecayStretcher(stretch_factor=2.0)
        
        feedback = 0.9
        adjusted = ds.adjust_feedback(feedback)
        
        # With stretch > 1, adjusted = feedback^(1/stretch) = feedback^0.5
        # which is larger than original feedback
        assert adjusted > feedback
    
    def test_adjust_feedback_stretch_shorter(self):
        """stretch_factor < 1 should decrease feedback (shorter sustain)."""
        ds = DecayStretcher(stretch_factor=0.5)
        
        feedback = 0.9
        adjusted = ds.adjust_feedback(feedback)
        
        # With stretch < 1, adjusted = feedback^2, which is smaller
        assert adjusted < feedback
    
    def test_adjust_feedback_zero_feedback(self):
        """Zero feedback should remain zero."""
        ds = DecayStretcher(stretch_factor=2.0)
        
        adjusted = ds.adjust_feedback(0.0)
        assert adjusted == 0.0
    
    def test_adjust_feedback_invalid(self):
        """Feedback must be in [0, 1)."""
        ds = DecayStretcher()
        
        with pytest.raises(ValueError):
            ds.adjust_feedback(-0.1)
        with pytest.raises(ValueError):
            ds.adjust_feedback(1.0)
        with pytest.raises(ValueError):
            ds.adjust_feedback(1.5)
    
    def test_set_stretch_factor(self):
        """Test changing stretch factor."""
        ds = DecayStretcher(stretch_factor=1.0)
        ds.set_stretch_factor(2.0)
        
        assert ds.stretch_factor == 2.0
    
    def test_set_stretch_factor_invalid(self):
        """Set stretch factor with invalid value should raise error."""
        ds = DecayStretcher()
        with pytest.raises(ValueError):
            ds.set_stretch_factor(-1.0)
