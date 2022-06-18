"""
This class will get the operational area
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-17
"""

import numpy as np
import pandas as pd

from TAICHI.Square2D.Config.Config import XLIM, YLIM, FILEPATH


class OpArea:

    def __init__(self):
        pass

    def get_operational_area(self):
        self.polygon = np.array([[XLIM[0], YLIM[0]],
                                 [XLIM[1], YLIM[0]],
                                 [XLIM[1], YLIM[1]],
                                 [XLIM[0], YLIM[1]]])
        df = pd.DataFrame(self.polygon, columns=['x', 'y'])
        df.to_csv(FILEPATH + "Config/polygon_border.csv", index=False)
        print("Operational area is saved successfully!")


if __name__ == "__main__":
    op = OpArea()
    op.get_operational_area()

