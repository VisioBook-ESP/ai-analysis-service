import pytest
from unittest.mock import MagicMock, patch


# ==============================================================================
# Tests pour is_gpu_available()
# ==============================================================================


@pytest.mark.unit
def test_is_gpu_available_with_gpu():
    """Test is_gpu_available() quand GPU est disponible."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True

    with patch('torch.cuda', mock_cuda):
        from src.utils.gpu_utils import is_gpu_available

        result = is_gpu_available()

        assert result is True
        mock_cuda.is_available.assert_called_once()


@pytest.mark.unit
def test_is_gpu_available_without_gpu():
    """Test is_gpu_available() quand GPU n'est pas disponible."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = False

    with patch('torch.cuda', mock_cuda):
        from src.utils.gpu_utils import is_gpu_available

        result = is_gpu_available()

        assert result is False
        mock_cuda.is_available.assert_called_once()


# ==============================================================================
# Tests pour get_gpu_info()
# ==============================================================================


@pytest.mark.unit
def test_get_gpu_info_without_gpu():
    """Test get_gpu_info() retourne None si pas de GPU."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = False

    with patch('torch.cuda', mock_cuda):
        from src.utils.gpu_utils import get_gpu_info

        result = get_gpu_info()

        assert result is None


@pytest.mark.unit
def test_get_gpu_info_with_single_gpu():
    """Test get_gpu_info() avec un seul GPU."""
    # Mock torch.cuda
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mock_cuda.current_device.return_value = 0

    # Mock device properties
    mock_props = MagicMock()
    mock_props.name = "NVIDIA GeForce RTX 3080"
    mock_props.major = 8
    mock_props.minor = 6
    mock_props.total_memory = 10 * 1024**3  # 10 GB

    mock_cuda.get_device_properties.return_value = mock_props
    mock_cuda.memory_allocated.return_value = 2 * 1024**3  # 2 GB allocated
    mock_cuda.memory_reserved.return_value = 3 * 1024**3   # 3 GB reserved

    # Mock torch.version
    mock_version = MagicMock()
    mock_version.cuda = "12.1"

    with patch('torch.cuda', mock_cuda), patch('torch.version', mock_version):
        from src.utils.gpu_utils import get_gpu_info

        result = get_gpu_info()

        assert result is not None
        assert result["available"] is True
        assert result["device_count"] == 1
        assert result["current_device"] == 0
        assert result["cuda_version"] == "12.1"
        assert len(result["devices"]) == 1

        device = result["devices"][0]
        assert device["id"] == 0
        assert device["name"] == "NVIDIA GeForce RTX 3080"
        assert device["compute_capability"] == "8.6"
        assert device["memory_total_gb"] == 10.0
        assert device["memory_allocated_gb"] == 2.0
        assert device["memory_reserved_gb"] == 3.0
        assert device["memory_free_gb"] == 7.0
        assert device["utilization_percent"] == 20.0


@pytest.mark.unit
def test_get_gpu_info_with_multiple_gpus():
    """Test get_gpu_info() avec plusieurs GPUs."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2
    mock_cuda.current_device.return_value = 0

    # Mock pour GPU 0
    mock_props_0 = MagicMock()
    mock_props_0.name = "GPU 0"
    mock_props_0.major = 8
    mock_props_0.minor = 0
    mock_props_0.total_memory = 8 * 1024**3

    # Mock pour GPU 1
    mock_props_1 = MagicMock()
    mock_props_1.name = "GPU 1"
    mock_props_1.major = 8
    mock_props_1.minor = 6
    mock_props_1.total_memory = 16 * 1024**3

    def get_props(device_id):
        return mock_props_0 if device_id == 0 else mock_props_1

    mock_cuda.get_device_properties.side_effect = get_props
    mock_cuda.memory_allocated.return_value = 1 * 1024**3
    mock_cuda.memory_reserved.return_value = 2 * 1024**3

    mock_version = MagicMock()
    mock_version.cuda = "11.8"

    with patch('torch.cuda', mock_cuda), patch('torch.version', mock_version):
        from src.utils.gpu_utils import get_gpu_info

        result = get_gpu_info()

        assert result["device_count"] == 2
        assert len(result["devices"]) == 2
        assert result["devices"][0]["name"] == "GPU 0"
        assert result["devices"][1]["name"] == "GPU 1"


@pytest.mark.unit
def test_get_gpu_info_with_zero_memory():
    """Test get_gpu_info() avec memoire totale a 0 (edge case)."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mock_cuda.current_device.return_value = 0

    mock_props = MagicMock()
    mock_props.name = "Test GPU"
    mock_props.major = 7
    mock_props.minor = 5
    mock_props.total_memory = 0  # Edge case

    mock_cuda.get_device_properties.return_value = mock_props
    mock_cuda.memory_allocated.return_value = 0
    mock_cuda.memory_reserved.return_value = 0

    mock_version = MagicMock()
    mock_version.cuda = "11.0"

    with patch('torch.cuda', mock_cuda), patch('torch.version', mock_version):
        from src.utils.gpu_utils import get_gpu_info

        result = get_gpu_info()

        device = result["devices"][0]
        assert device["utilization_percent"] == 0  # Doit gerer la division par 0


@pytest.mark.unit
def test_get_gpu_info_with_exception():
    """Test get_gpu_info() gere les exceptions gracieusement."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.side_effect = RuntimeError("CUDA error")

    with patch('torch.cuda', mock_cuda):
        from src.utils.gpu_utils import get_gpu_info

        result = get_gpu_info()

        assert result is not None
        assert result["available"] is True
        assert "error" in result
        assert "CUDA error" in result["error"]


# ==============================================================================
# Tests pour get_device_name()
# ==============================================================================


@pytest.mark.unit
def test_get_device_name_with_gpu():
    """Test get_device_name() retourne 'cuda' si GPU disponible."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True

    with patch('torch.cuda', mock_cuda):
        from src.utils.gpu_utils import get_device_name

        result = get_device_name()

        assert result == "cuda"


@pytest.mark.unit
def test_get_device_name_without_gpu():
    """Test get_device_name() retourne 'cpu' si pas de GPU."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = False

    with patch('torch.cuda', mock_cuda):
        from src.utils.gpu_utils import get_device_name

        result = get_device_name()

        assert result == "cpu"


# ==============================================================================
# Tests pour get_device_index()
# ==============================================================================


@pytest.mark.unit
def test_get_device_index_with_gpu():
    """Test get_device_index() retourne 0 si GPU disponible."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True

    with patch('torch.cuda', mock_cuda):
        from src.utils.gpu_utils import get_device_index

        result = get_device_index()

        assert result == 0


@pytest.mark.unit
def test_get_device_index_without_gpu():
    """Test get_device_index() retourne -1 si pas de GPU."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = False

    with patch('torch.cuda', mock_cuda):
        from src.utils.gpu_utils import get_device_index

        result = get_device_index()

        assert result == -1
