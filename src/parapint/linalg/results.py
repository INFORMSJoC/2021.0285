#  ___________________________________________________________________________
#
#  Parapint
#  Copyright (c) 2020
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum


class LinearSolverStatus(enum.Enum):
    successful = 0
    not_enough_memory = 1
    singular = 2
    error = 3
    warning = 4


class LinearSolverResults(object):
    def __init__(self):
        self.status = None
