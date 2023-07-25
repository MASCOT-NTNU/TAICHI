"""
Unit test module for the GRF class.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-07-05
"""

from unittest import TestCase
from EDA import EDA
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from pykdtree.kdtree import KDTree
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
from matplotlib.gridspec import GridSpec


class TestEDA(TestCase):

    def setUp(self) -> None:
        self.eda = EDA()
        self.eda.load_data()

    def test_rerun_eda_using_grf(self) -> None:
        self.eda.rerun_mission_using_grf_kernel()
