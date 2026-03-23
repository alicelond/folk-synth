"""
Delay line module for Karplus-Strong synthesis.

Implements a circular buffer delay line that forms the core of the Karplus-Strong
algorithm. The delay line stores the resonating signal and applies feedback to
sustain the oscillation.
"""

import numpy as np


class DelayLine:
    """
    Circular buffer delay line for the Karplus-Strong algorithm.
    
    Maintains a fixed-length buffer representing the resonant frequency of the
    string. The delay length directly determines the pitch: longer delay = lower pitch.
    
    Attributes:
        buffer (np.ndarray): Circular buffer storing samples.
        delay_length (int): Total length of the delay line in samples.
        write_index (int): Current write position in the circular buffer.
        read_index (int): Current read position in the circular buffer.
    """
    
    def __init__(self, delay_length):
        """
        Initialize the delay line with a specified length.
        
        Args:
            delay_length (int): Number of samples in the delay line.
                Must be > 0. Typical value: sample_rate / frequency.
        
        Raises:
            ValueError: If delay_length <= 0 or not an integer.
        """
        if not isinstance(delay_length, (int, np.integer)) or delay_length <= 0:
            raise ValueError(f"Delay length must be a positive integer, got {delay_length}")
        
        self.delay_length = int(delay_length)
        self.buffer = np.zeros(self.delay_length, dtype=np.float64)
        self.write_index = 0
        self.read_index = 0
    
    def write(self, sample):
        """
        Write a sample to the delay line at the current write position.
        
        Args:
            sample (float): Single sample value to write.
        """
        self.buffer[self.write_index] = float(sample)
        self.write_index = (self.write_index + 1) % self.delay_length
    
    def read(self):
        """
        Read a sample from the delay line at the current read position.
        
        Returns:
            float: Sample value at the current read position.
        """
        value = self.buffer[self.read_index]
        self.read_index = (self.read_index + 1) % self.delay_length
        return value
    
    def write_and_read(self, sample):
        """
        Atomically write a sample and read the previous delayed sample.
        
        This is the standard operation in the Karplus-Strong feedback loop:
        the new excitation (or filtered feedback) is written, and the old
        delayed sample is read out for further processing.
        
        Args:
            sample (float): Sample to write.
        
        Returns:
            float: Sample that was at the read position (now being displaced).
        """
        delayed_sample = self.buffer[self.read_index]
        self.buffer[self.write_index] = float(sample)
        
        self.read_index = (self.read_index + 1) % self.delay_length
        self.write_index = (self.write_index + 1) % self.delay_length
        
        return delayed_sample
    
    def reset(self):
        """
        Clear the buffer and reset read/write indices to the start.
        
        Useful for stopping synthesis or restarting with fresh initial conditions.
        """
        self.buffer.fill(0.0)
        self.read_index = 0
        self.write_index = 0
    
    def get_buffer_snapshot(self):
        """
        Get a copy of the entire buffer content without modifying indices.
        
        Returns:
            np.ndarray: Copy of the internal buffer array.
        """
        return self.buffer.copy()
