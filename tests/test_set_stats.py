import pytest
import torch

from heterowave.models.set_stats import MaskedSetAggregator, masked_set_statistics


def test_masked_statistics_use_population_variance():
    features = torch.tensor([1.0, 3.0, 100.0]).view(1, 3, 1, 1, 1)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    mean, variance, count = masked_set_statistics(features, mask)
    torch.testing.assert_close(mean, torch.tensor([[[[2.0]]]]))
    torch.testing.assert_close(variance, torch.tensor([[[[1.0]]]]))
    torch.testing.assert_close(count, torch.tensor([[[[2 / 3]]]]))


@pytest.mark.parametrize("mode", ["mean", "mean_var", "mean_var_count"])
def test_aggregator_is_permutation_invariant(mode):
    torch.manual_seed(3)
    features = torch.randn(2, 5, 4, 6, 6)
    mask = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.bool)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    aggregator = MaskedSetAggregator(4, mode)
    expected = aggregator(features, mask)
    actual = aggregator(features[:, permutation], mask[:, permutation])
    torch.testing.assert_close(actual, expected)
