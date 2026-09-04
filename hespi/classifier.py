import sys
import types
from pathlib import Path
from typing import List, Union

import pandas as pd


def _install_torchapp_unpickling_shim() -> None:
    """
    Registers a stand-in for `torchapp.examples.image_classifier` in `sys.modules`.

    Pretrained classifier weights were exported with `fastai.Learner.export`, which pickles
    the full data pipeline, including the `PathColReader` transform that used to live in
    torchapp. Unpickling needs that exact class importable at that exact module path, even
    though it's never called during inference (predictions are made directly from image
    paths, bypassing the CSV column readers). Since torchapp itself isn't installable on
    Python 3.12+, this shim supplies just that one class so `load_learner` can succeed
    without depending on torchapp at all.
    """
    if "torchapp.examples.image_classifier" in sys.modules:
        return

    from fastcore.transform import DisplayedTransform

    class PathColReader(DisplayedTransform):
        def __init__(self, column_name: str = None, base_dir: Path = None):
            self.column_name = column_name
            self.base_dir = base_dir

        def __call__(self, row, **kwargs):
            path = Path(row[self.column_name])
            if not path.is_absolute():
                path = self.base_dir / path
            return path

    torchapp_module = types.ModuleType("torchapp")
    examples_module = types.ModuleType("torchapp.examples")
    image_classifier_module = types.ModuleType("torchapp.examples.image_classifier")
    image_classifier_module.PathColReader = PathColReader
    examples_module.image_classifier = image_classifier_module
    torchapp_module.examples = examples_module

    sys.modules["torchapp"] = torchapp_module
    sys.modules["torchapp.examples"] = examples_module
    sys.modules["torchapp.examples.image_classifier"] = image_classifier_module


class PrimaryLabelClassifier:
    """
    Loads a fastai-exported ``Learner`` pickle and classifies images with it.

    This replaces torchapp's ``examples.image_classifier.ImageClassifier``, which hespi
    previously used only for inference. torchapp's fastai-based releases cap Python at <3.12,
    and its newer releases dropped fastai (and the ability to load these pretrained weights)
    in favour of torchvision/Lightning, so hespi now talks to fastai directly instead.
    """

    def __init__(self):
        self.pretrained = None

    def __call__(
        self,
        items: Union[Path, str, List[Union[Path, str]]],
        pretrained: Union[Path, str] = None,
        gpu: bool = False,
        output_csv: Path = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        from fastai.learner import load_learner

        _install_torchapp_unpickling_shim()

        pretrained = pretrained or self.pretrained
        learner = load_learner(pretrained, cpu=not gpu)

        if isinstance(items, (str, Path)):
            items = [items]
        items = [Path(item) for item in items]

        dataloader = learner.dls.test_dl(items)
        probabilities, _ = learner.get_preds(dl=dataloader, reorder=False, with_decoded=False)
        vocab = learner.dls.vocab

        data = []
        for item, scores in zip(items, probabilities):
            prediction = vocab[int(scores.argmax())]
            if verbose:
                print(f"'{item}': '{prediction}'")
            data.append([item, prediction] + scores.tolist())

        df = pd.DataFrame(data, columns=["path", "prediction"] + list(vocab))
        if output_csv:
            df.to_csv(output_csv)

        if verbose:
            print(df)

        return df
