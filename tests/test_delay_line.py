"""
Unit tests for the delay_line module.
"""

import pytest
import numpy as np
from core.delay_line import DelayLine


class TestDelayLine:
    """Tests for DelayLine class."""
    
    def test_init_valid(self):
        """Test valid initialization."""
        dl = DelayLine(1000)
        assert dl.delay_length == 1000
        assert len(dl.buffer) == 1000
        assert np.all(dl.buffer == 0)
        assert dl.write_index == 0
        assert dl.read_index == 0
    
    def test_init_invalid_length(self):
        """Delay length must be positive integer."""
        with pytest.raises(ValueError):
            DelayLine(0)
        with pytest.raises(ValueError):
            DelayLine(-100)
        with pytest.raises(ValueError):
            DelayLine(50.5)  # Float
    
    def test_write_single_sample(self):
        """Test writing a single sample."""
        dl = DelayLine(10)
        dl.write(0.5)
        
        assert dl.buffer[0] == 0.5
        assert dl.write_index == 1
    
    def test_write_wraps_around(self):
        """Test that write index wraps around."""
        dl = DelayLine(3)
        dl.write(1.0)
        dl.write(2.0)
        dl.write(3.0)
        dl.write(4.0)  # Should wrap to index 0
        
        assert dl.write_index == 1
        assert dl.buffer[0] == 4.0
    
    def test_read_single_sample(self):
        """Test reading from delay line."""
        dl = DelayLine(10)
        dl.buffer[0] = 0.5
        
        value = dl.read()
        assert value == 0.5
        assert dl.read_index == 1
    
    def test_read_wraps_around(self):
        """Test that read index wraps around."""
        dl = DelayLine(3)
        dl.buffer[0] = 1.0
        dl.buffer[1] = 2.0
        dl.buffer[2] = 3.0
        
        val1 = dl.read()
        val2 = dl.read()
        val3 = dl.read()
        val4 = dl.read()  # Should wrap to index 0
        
        assert val1 == 1.0
        assert val2 == 2.0
        assert val3 == 3.0
        assert val4 == 1.0
        assert dl.read_index == 1
    
    def test_write_and_read_basic(self):
        """Test atomic write_and_read operation."""
        dl = DelayLine(3)
        # Prime the buffer
        dl.buffer[0] = 10.0
        
        # write_and_read should read position 0, write at position 0
        old_value = dl.write_and_read(5.0)
        
        assert old_value == 10.0
        assert dl.buffer[0] == 5.0
        assert dl.write_index == 1
        assert dl.read_index == 1
    
    def test_write_and_read_sequence(self):
        """Test write_and_read in a sequence (simulating feedback loop)."""
        dl = DelayLine(2)
        dl.buffer[0] = 100.0
        dl.buffer[1] = 200.0
        
        # First call: read from index 0, write at index 0
        val1 = dl.write_and_read(1.0)
        assert val1 == 100.0
        assert dl.buffer[0] == 1.0
        
        # Second call: read from index 1, write at index 1
        val2 = dl.write_and_read(2.0)
        assert val2 == 200.0
        assert dl.buffer[1] == 2.0
        
        # Third call: wraps back to index 0
        val3 = dl.write_and_read(3.0)
        assert val3 == 1.0
        assert dl.buffer[0] == 3.0
    
    def test_reset(self):
        """Test reset functionality."""
        dl = DelayLine(5)
        
        # Fill with data
        for i in range(10):
            dl.write(float(i))
        
        # Reset
        dl.reset()
        
        assert np.all(dl.buffer == 0)
        assert dl.write_index == 0
        assert dl.read_index == 0
    
    def test_get_buffer_snapshot(self):
        """Test getting a snapshot of buffer without modifying indices."""
        dl = DelayLine(3)
        dl.buffer[0] = 1.0
        dl.buffer[1] = 2.0
        dl.buffer[2] = 3.0
        
        snapshot1 = dl.get_buffer_snapshot()
        snap_value = snapshot1[0]  # Simulate reading from snapshot
        snapshot2 = dl.get_buffer_snapshot()
        
        # Indices should not change
        assert dl.read_index == 0
        assert dl.write_index == 0
        
        # Snapshots should be copies, not references
        snapshot1[0] = 999.0
        assert dl.buffer[0] == 1.0  # Original unchanged
        numpy.testing.assert_array_equal(snapshot2, np.array([1.0, 2.0, 3.0]))
    
    def test_delay_line_property(self):
        """Test that delay line acts as a true delay."""
        dl = DelayLine(5)
        
        # The delay line starts with read_index=0 and write_index=0
        # Write zeros, then an impulse
        dl.write(0.0)  # index 0
        dl.write(0.0)  # index 1
        dl.write(0.0)  # index 2
        dl.write(0.0)  # index 3
        dl.write(0.0)  # index 4
        dl.write(1.0)  # wraps to index 0 (overwrites first zero)
        
        # Now buffer is [1.0, 0, 0, 0, 0] and read_index=0, write_index=1
        # Reading should give us: 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 (wraps back)
        output = []
        for _ in range(6):
            output.append(dl.read())
        
        assert output[0] == 1.0  # Read from position that was just written
        assert output[1] == 0.0
        assert output[2] == 0.0
        assert output[3] == 0.0
        assert output[4] == 0.0
        # Impulse repeats after delay_length
    
    def test_buffer_data_type(self):
        """Test that buffer maintains float64 dtype."""
        dl = DelayLine(10)
        assert dl.buffer.dtype == np.float64
        
        dl.write(10)  # Int input
        assert dl.buffer.dtype == np.float64


# Import for snapshot test
import numpy
