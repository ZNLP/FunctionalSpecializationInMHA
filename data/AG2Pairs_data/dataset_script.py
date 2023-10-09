import datasets
import pandas as pd
import six

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class AG2PairsDataset(datasets.GeneratorBasedBuilder):
    """A dataset script for loading AGNews(2pairs). Version 1.0.0"""
    VERSION = datasets.Version("1.0.0")

    _URL = "./"
    _URLS = {
        "train": _URL + "train.tsv",
        # "train_aug": _URL + "train_aug.tsv",
        "test": _URL + "test.tsv",
    }
    _NAMES = [
        "same",
        "different"
    ]

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = datasets.Features(
            {
                "news1": datasets.Value("string"),
                "news2": datasets.Value("string"),
                "label": datasets.ClassLabel(names=self._NAMES)
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="Pair of AGNews dataset, generated by AGNews",
            # This defines the different columns of the dataset and their types
            features=features
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        urls_to_download = self._URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name="train", gen_kwargs={"filepath": downloaded_files["train"], "split": "train"}),
            # datasets.SplitGenerator(name="train_aug", gen_kwargs={"filepath": downloaded_files["train_aug"], "split": "train_aug"}),
            datasets.SplitGenerator(name="validation", gen_kwargs={"filepath": downloaded_files["test"], "split": "test"}),
        ]
    
    def _generate_examples(self, filepath, split):

        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        
        with open(filepath, encoding="utf-8") as f:
            for key,line in enumerate(f):
                if key == 0:
                    continue
                line = line.strip().split("\t")

                yield key, {
                        "news1": convert_to_unicode(line[1]+" - "+line[2]),
                        "news2": convert_to_unicode(line[3]+" - "+line[4]),
                        "label": convert_to_unicode(line[0])
                    }