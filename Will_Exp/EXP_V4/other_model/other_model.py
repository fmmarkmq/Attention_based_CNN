import warnings
from typing import Any, Iterable, Literal, Optional

import pandas as pd
import pts.trainer
from gluonts.dataset.common import ListDataset
from gluonts.model.estimator import Estimator
from tqdm.auto import tqdm

__all__ = ["df_to_ds", "PTSForecaster", "make_default_model"]


def df_to_ds(df: pd.DataFrame) -> ListDataset:
    # df = df.set_axis(df.index.to_timestamp())  # type: ignore [attr-defined]
    df = df.set_axis(df.index.to_pydatetime())
    return ListDataset(
        [
            {
                "item_id": str(name),
                "start": str(df.index[0]),
                "target": vals.values.tolist(),
            }
            for name, vals in df.iteritems()
        ],
        freq="1W-MON",
    )


def tqdm_disabled(iterable: Iterable, **kwargs: Any) -> tqdm:
    return tqdm(iterable, disable=True, **kwargs)


class PTSForecaster:
    """An adapter for ``pytorch-ts`` models to support the ``atd2022`` API.
    Parameters
    ----------
    model: gluonts.model.estimator.Estimator
        A model that supports the gluonts Estimator protocol.
    verbose: bool
        If ``True``, show ``tqdm`` progress bar for each fit epoch or call to predict.
        Otherwise, monkey-patch ``pytorch-ts`` to disable ``tqdm`` progress bars.
        Note: This will disable progress bars in ``pytorch-ts`` interpreter-wide,
        because ``pytorch-ts`` itself does not provide a convenient ``verbose``
        argument.
    """

    def __init__(self, model: Estimator, verbose: bool = True) -> None:
        self.model = model
        # Since pts doesn't provide a verbose option, monkey-patch tqdm.
        pts.trainer.tqdm = tqdm if verbose else tqdm_disabled

    def fit(
        self, y: pd.DataFrame, past_covariates: Optional[pd.DataFrame] = None
    ) -> "PTSForecaster":
        """Fit the model.
        This method will:
        - Adapt the input dataframe into a format suitable for ``gluon-ts``.
        - Fit a ``Predictor`` object and store it in ``self``.
        - Save the last several observations from the training dataset so that
          we can provide sufficient "context" to the model at prediction time.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.predictor = self.model.train(training_data=df_to_ds(y), num_workers=0)
        self.context = df_to_ds(y.iloc[-self.model.context_length :])
        return self

    def predict(self, x: pd.Index) -> pd.DataFrame:
        """Predict the target values in the future.
        This method will:
        - Ensure that we are not requesting more than trained model's prediction
          length.
        - Use the trained predictor to make predictions.
        - Format those predictions into the ``atd2022`` format.
        """
        assert len(x) <= self.predictor.prediction_length
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return pd.DataFrame(
                {
                    eval(col.item_id): col.mean[: len(x)]
                    for col in self.predictor.predict(self.context)
                },
                index=x,
            )

    def __repr__(self) -> str:
        return f"PTSForecaster(model={self.model.__class__.__name__})"


def make_default_model(model_name: Literal["deepar", "nbeats", "tft"]) -> Estimator:
    """Generate preconfigured models with default settings.
    Parameters
    ----------
    model_name: Literal["deepar", "nbeats", "tft"]
        A string mapping to one of the three preconfigured models.
    Returns
    -------
    Estimator
        A ``gluon-ts`` estimator that is compatible with the ``PTSEForecaster`` class.
    """
    import torch
    from pts import Trainer
    from pts.model.deepar import DeepAREstimator
    from pts.model.n_beats import NBEATSEstimator
    from pts.model.tft import TemporalFusionTransformerEstimator
    from pts.modules.distribution_output import ImplicitQuantileOutput

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_length = 4
    trainer = Trainer(
        device=device,
        epochs=150,
        learning_rate=1e-4,
        num_batches_per_epoch=16,
        batch_size=64,
        gradient_clip_val=0.1,
    )

    if model_name == "deepar":
        estimator = DeepAREstimator(
            distr_output=ImplicitQuantileOutput(output_domain="Positive"),
            cell_type="GRU",
            input_size=21,
            dropout_rate=0.01,
            prediction_length=prediction_length,
            context_length=25,
            # freq="1W-MON", #atd
            freq = "D", #wiki_traffic
            num_cells=20,
            num_layers=2,
            trainer=trainer,
        )
    elif model_name == "nbeats":
        estimator = NBEATSEstimator(
            # freq="1W-MON", #atd
            freq = "D", #wiki_traffic
            prediction_length=prediction_length,
            trainer=trainer,
            context_length=25,
            num_stacks=15,
        )
    elif model_name == "tft":
        estimator = TemporalFusionTransformerEstimator(
            # freq="1W-MON", #atd
            freq = "D", #wiki_traffic
            embed_dim=16,
            num_heads=4,
            prediction_length=4,
            context_length=25,
            trainer=trainer,
        )
    else:
        raise ValueError(f"Default model not available for '{model_name}'")
    return estimator