"""
Tests for Extended Configuration Space.

Author: H M Shujaat Zaheer
"""

import numpy as np
import pytest

from morphollm.core.extended_config_space import (
    ExtendedConfigSpace,
    MorphologySpace,
    JointSpace,
    create_manipulator_space
)


class TestMorphologySpace:
    """Tests for MorphologySpace."""
    
    def test_initialization(self):
        """Test basic initialization."""
        space = MorphologySpace(dim=6)
        assert space.dim == 6
        assert len(space.parameter_names) == 6
        assert space.bounds[0].shape == (6,)
        assert space.bounds[1].shape == (6,)
    
    def test_sample(self):
        """Test random sampling."""
        space = MorphologySpace(dim=4)
        samples = space.sample(n=10)
        assert samples.shape == (10, 4)
        assert np.all(samples >= space.bounds[0])
        assert np.all(samples <= space.bounds[1])
    
    def test_clip(self):
        """Test clipping to bounds."""
        space = MorphologySpace(dim=3, bounds=(np.zeros(3), np.ones(3)))
        m = np.array([-0.5, 0.5, 1.5])
        clipped = space.clip(m)
        np.testing.assert_array_equal(clipped, [0.0, 0.5, 1.0])
    
    def test_rate_limits(self):
        """Test rate limit clipping."""
        space = MorphologySpace(dim=3, rate_limits=np.array([0.1, 0.2, 0.3]))
        m_dot = np.array([0.5, -0.5, 0.1])
        clipped = space.clip_rate(m_dot)
        np.testing.assert_array_almost_equal(clipped, [0.1, -0.2, 0.1])


class TestJointSpace:
    """Tests for JointSpace."""
    
    def test_initialization(self):
        """Test basic initialization."""
        space = JointSpace(dim=7)
        assert space.dim == 7
        assert len(space.joint_names) == 7
    
    def test_clip(self):
        """Test joint limit clipping."""
        space = JointSpace(dim=2, joint_limits=(np.array([-1, -1]), np.array([1, 1])))
        q = np.array([2.0, -2.0])
        clipped = space.clip(q)
        np.testing.assert_array_equal(clipped, [1.0, -1.0])


class TestExtendedConfigSpace:
    """Tests for ExtendedConfigSpace."""
    
    def test_initialization(self):
        """Test basic initialization."""
        space = ExtendedConfigSpace(joint_dim=7, morphology_dim=12)
        assert space.joint_dim == 7
        assert space.morphology_dim == 12
        assert space.total_dim == 19
    
    def test_compose_decompose(self):
        """Test state composition and decomposition."""
        space = ExtendedConfigSpace(joint_dim=3, morphology_dim=2)
        
        q = np.array([1.0, 2.0, 3.0])
        m = np.array([0.5, 0.6])
        
        x_ext = space.compose(q, m)
        assert x_ext.shape == (5,)
        np.testing.assert_array_equal(x_ext[:3], q)
        np.testing.assert_array_equal(x_ext[3:], m)
        
        q_out, m_out = space.decompose(x_ext)
        np.testing.assert_array_equal(q_out, q)
        np.testing.assert_array_equal(m_out, m)
    
    def test_compose_decompose_control(self):
        """Test control composition and decomposition."""
        space = ExtendedConfigSpace(joint_dim=3, morphology_dim=2)
        
        q_dot = np.array([0.1, 0.2, 0.3])
        m_dot = np.array([0.05, 0.06])
        
        u_ext = space.compose_control(q_dot, m_dot)
        assert u_ext.shape == (5,)
        
        q_dot_out, m_dot_out = space.decompose_control(u_ext)
        np.testing.assert_array_equal(q_dot_out, q_dot)
        np.testing.assert_array_equal(m_dot_out, m_dot)
    
    def test_interpolate(self):
        """Test linear interpolation."""
        space = ExtendedConfigSpace(joint_dim=2, morphology_dim=2)
        
        x1 = np.array([0.0, 0.0, 0.0, 0.0])
        x2 = np.array([1.0, 1.0, 1.0, 1.0])
        
        x_mid = space.interpolate(x1, x2, 0.5)
        np.testing.assert_array_almost_equal(x_mid, [0.5, 0.5, 0.5, 0.5])
    
    def test_distance(self):
        """Test distance computation."""
        space = ExtendedConfigSpace(joint_dim=2, morphology_dim=2)
        
        x1 = np.array([0.0, 0.0, 0.0, 0.0])
        x2 = np.array([1.0, 0.0, 0.0, 0.0])
        
        dist = space.distance(x1, x2)
        assert dist == 1.0


class TestCreateManipulatorSpace:
    """Tests for convenience function."""
    
    def test_create_default(self):
        """Test default manipulator space creation."""
        space = create_manipulator_space()
        assert space.joint_dim == 7
        assert space.morphology_dim == 12  # 6 + 3 + 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
