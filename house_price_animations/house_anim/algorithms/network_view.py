"""The MLP regressor as a live network diagram: animkit's NetworkView with a price output unit.

animkit draws the layers, the weights as edges and, at inference, the activations firing through
them. This subclass only says what the single output unit means here: a price, labelled in $k and
coloured on the price ramp.
"""

import numpy as np

from animkit.algorithms.network_view import NetworkView as BaseNetworkView
from house_anim.algorithms.common import price_k, price_rgba


class NetworkView(BaseNetworkView):
    """animkit's network diagram, with the output unit read as a house price.

    Attributes:
        output_label: What the last layer's single unit means.
        subject: What one prediction is about, for the title.
    """

    output_label = "price"  # what the last layer's single unit means
    subject = "House"  # what one prediction is about, for the title

    def _output(self, z: float) -> tuple[str, object]:
        """Text and colour of the output node for the network's raw output ``z``."""
        return price_k(z), price_rgba(z)

    def show_prediction(self, model, x: np.ndarray, house_number: int) -> None:
        """Draws animkit's forward pass for the house; only the argument's name is this project's.

        Args:
            model: The fitted ``MLPRegressor``.
            x: The house's scaled features.
            house_number: The house's position among the test houses, from 1, for the title.
        """
        super().show_prediction(model, x, house_number)
