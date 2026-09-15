import torch

from probflow.parameters import Parameter


def test_Parameter_slicing_pytorch():
    """Tests a slicing Parameters."""
    # Create 1D parameter
    param = Parameter(shape=[2, 3, 4, 5])

    sl = param[torch.tensor([0]), :, ::2, :].detach().numpy()
    assert sl.ndim == 4
    assert sl.shape[0] == 1
    assert sl.shape[1] == 3
    assert sl.shape[2] == 2
    assert sl.shape[3] == 5
