# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Datatypes used for configuring diffusion.
"""

from __future__ import annotations

import enum


class DiffusionType(int, enum.Enum):
    """
    Order of nabla operator for diffusion.

    Note: Called `hdiff_order` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 5.
    """

    NO_DIFFUSION = -1  #: no diffusion
    LINEAR_2ND_ORDER = 2  #: 2nd order linear diffusion on all vertical levels
    SMAGORINSKY_NO_BACKGROUND = 3  #: Smagorinsky diffusion without background diffusion
    LINEAR_4TH_ORDER = 4  #: 4th order linear diffusion on all vertical levels
    SMAGORINSKY_4TH_ORDER = 5  #: Smagorinsky diffusion with fourth-order background diffusion


class SmagorinskyStencilType(int, enum.Enum):
    """
    Type of the reconstruction stencil for the Smagorinsky diffusion of normal wind (vn).

    Note: Called `itype_vn_diffu` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 1 in combination with lsmag_3d=False.
    """

    DIAMOND_VERTICES = (
        1  #: Smagorinsky diffusion of vn with diamond stencil on vertices (only for vn)
    )
    CELLS_AND_VERTICES = 2  #: Smagorinsky diffusion of vn with stencil on neighboring vertices (E2V) and cell centers (E2C)


class TemperatureDiscretizationType(int, enum.Enum):
    """
    Type of the discretization of the Smagorinsky diffusion of temperature.

    Note: Called `itype_t_diffu` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 2.
    """

    HOMOGENEOUS = 1  #: K Lap(T)
    HETEROGENEOUS = 2  #: Div (K Grad(T))


class TurbulenceShearForcingType(int, enum.Enum):
    """
    Type of shear forcing used in turbulence.

    Note: called `itype_sher` in `mo_turbdiff_nml.f90`
    """

    VERTICAL_OF_HORIZONTAL_WIND = 0  #: only vertical shear of horizontal wind
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND = (
        1  #: as `VERTICAL_ONLY` plus horizontal shear correction
    )
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND = (
        2  #: as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` plus shear form vertical velocity
    )
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND_LTHESH = 3  #: same as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` but scaling of coarse-grid horizontal shear production term with 1/sqrt(Ri) (if LTKESH = TRUE)
