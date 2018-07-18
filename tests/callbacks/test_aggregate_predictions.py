from unittest import TestCase
import torch
from torchbearer.callbacks import AggregatePredictions
import torchbearer as tb


class TestAggregatePredictions(TestCase):

    def test_aggreate_predictions(self):
        aggregator = AggregatePredictions()

        y_pred_1 = torch.Tensor([1,2,3])
        y_pred_2 = torch.Tensor([3,4,5])

        state_1 = {tb.Y_PRED: y_pred_1}
        state_2 = {tb.Y_PRED: y_pred_2}
        final_state = {}

        aggregator.on_step_validation(state_1)
        self.assertTrue(list(aggregator.predictions_list[0].numpy()) == list(y_pred_1.numpy()))

        aggregator.on_step_validation(state_2)
        self.assertTrue(list(aggregator.predictions_list[1].numpy()) == list(y_pred_2.numpy()))

        aggregate = torch.cat([y_pred_1, y_pred_2])
        aggregator.on_end_validation(final_state)
        self.assertTrue(list(final_state[tb.FINAL_PREDICTIONS].numpy()) == list(aggregate.numpy()))


