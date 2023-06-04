import xml.etree.ElementTree as ET
from xml.etree import cElementTree as ElementTree
import pandas as pd
import numpy as np


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    """
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    """

    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


class MC15TuneV1(object):
    def __init__(self, config_path, particle_name):
        """
        :param config_path - path to .xml with weights
        """
        tree = ET.parse(config_path)
        root = tree.getroot()
        xmldict = XmlDictConfig(root)
        self.features = [feature.attrib for feature in root[2]]
        self.feature_names = [f["Title"] for f in self.features]
        self.cuts = {}
        if particle_name.lower() == "muon":
            #         https://gitlab.cern.ch/lhcb/Rec/-/blob/master/Rec/ChargedProtoANNPID/data/MCUpTuneV1/IsMuonTrackPreselection.txt
            self.cuts = {"MuonIsMuon": lambda x: x > 0.5}

        weights = xmldict["Weights"]["Layout"]["Layer"][:-1]
        self.weights = [
            np.array(
                [[float(v) for v in line.split(" ")] for line in matrix],
                dtype="float64",
            )
            for matrix in weights
        ]
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.out_activation = sigmoid
        if particle_name.lower() == "ghost":
            self.in_activation = sigmoid
        else:
            self.in_activation = np.tanh

    def predict(self, df):
        """
        :param values - input values (pd.DataFrame)
        """
        values = self.preprocess(df[self.feature_names].values)
        batch_size = values.shape[0]
        activations = [self.in_activation for _ in range(len(self.weights) - 1)] + [
            self.out_activation
        ]
        for act, layer in zip(activations, self.weights):
            values = np.hstack([values, np.ones((batch_size, 1))])
            values = values.dot(layer)
            values = act(values)
        preds = values.squeeze(-1)
        if len(self.cuts) == 0:
            return preds
        else:
            return np.where(self._cut(df), preds, -2 * np.ones_like(preds))

    def _cut(self, df):
        res = np.ones(df.shape[0])
        for feature_name, feature_cut in self.cuts.items():
            res = np.logical_and(res, feature_cut(df[feature_name]))
        return res

    def preprocess(self, values):
        min_values = np.array([float(f["Min"]) for f in self.features])
        max_values = np.array([float(f["Max"]) for f in self.features])
        return 2 * (values - min_values) / (max_values - min_values) - 1
