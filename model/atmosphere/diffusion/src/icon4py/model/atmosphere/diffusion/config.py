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

import dataclasses
import enum
import functools
import math
import typing
from typing import Any

from icon4py.model.common import constants
from icon4py.model.common.utils import fortran_config


_T = typing.TypeVar("_T")


def _alias_property(attr_name: str, prop_name: str, attr_type: type[_T]) -> property:
    def getter(self) -> _T:
        return getattr(self, attr_name)

    def setter(self, value: _T) -> None:
        return setattr(self, attr_name, value)

    return property(fget=getter, fset=setter, doc=f"Alias to attribute '{attr_name}'")


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


class ForcingType(int, enum.Enum):
    """
    Type of physics forcing applied to the model.

    Note: called `iforcing` in `mo_run_nml.f90`
    """

    NO_FORCING = 0  #: no physics forcing (diagnostic / idealized runs)
    AES = 2  #: Atmospheric Earth System / ECHAM forcing (iaes)
    NWP = 3  #: Numerical Weather Prediction forcing (inwp)


@dataclasses.dataclass(kw_only=True)
class DiffusionConfig:
    """
    Contains necessary parameter to configure a diffusion run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    # TODO(halungge): to be read from config
    # TODO(halungge):  handle dependencies on other namelists (see below...)

    # parameters from namelist diffusion_nml

    diffusion_type: DiffusionType = DiffusionType.SMAGORINSKY_4TH_ORDER

    #: If True, apply diffusion on the vertical wind field
    #: Called 'lhdiff_w' in mo_diffusion_nml.f90
    hdiff_w: bool = True

    apply_to_vertical_wind: typing.ClassVar[property] = _alias_property(
        "hdiff_w", "apply_to_vertical_wind", bool
    )

    #: True apply diffusion on the horizontal wind field, is ONLY used in mo_nh_stepping.f90
    #: Called 'lhdiff_vn' in mo_diffusion_nml.f90
    hdiff_vn: bool = True

    apply_to_horizontal_wind: typing.ClassVar[property] = _alias_property(
        "hdiff_vn", "apply_to_horizontal_wind", bool
    )

    #:  If True, apply horizontal diffusion to temperature field
    #: Called 'lhdiff_temp' in mo_diffusion_nml.f90
    hdiff_temp: bool = True

    apply_to_temperature: typing.ClassVar[property] = _alias_property(
        "hdiff_temp", "apply_to_temperature", bool
    )

    #: If True, compute Smagorinsky diffusion to vertical wind field
    #: Called 'lhdiff_smag_w' in mo_diffusion_nml.f90
    hdiff_smag_w: bool = False

    apply_smag_diff_to_vertical_wind: typing.ClassVar[property] = _alias_property(
        "hdiff_smag_w", "apply_smag_diff_to_vertical_wind", bool
    )

    #: If True, compute 3D Smagorinsky diffusion coefficient
    #: Called 'lsmag_3d' in mo_diffusion_nml.f90
    smag_3d: bool = False

    compute_3d_smag_coeff: typing.ClassVar[property] = _alias_property(
        "smag_3d", "compute_3d_smag_coeff", bool
    )

    #: Options for discretizing the Smagorinsky momentum diffusion
    #: Called 'itype_vn_diffu' in mo_diffusion_nml.f90
    type_vn_diffu: SmagorinskyStencilType = SmagorinskyStencilType.DIAMOND_VERTICES

    #: Options for discretizing the Smagorinsky temperature diffusion
    #: Called 'itype_t_diffu' in mo_diffusion_nml.f90
    type_t_diffu: TemperatureDiscretizationType = TemperatureDiscretizationType.HETEROGENEOUS

    #: Ratio of e-folding time to (2*)time step
    #: Called 'hdiff_efdt_ratio' in mo_diffusion_nml.f90
    hdiff_efdt_ratio: float = 36.0

    #: Ratio of e-folding time to time step for w diffusion (NH only)
    #: Called 'hdiff_w_efdt_ratio' in mo_diffusion_nml.f90.
    hdiff_w_efdt_ratio: float = 15.0

    # TODO(muellch): The four smagorinsky factors and heights should be in one or two dataclasses.
    #: Smagorinsky factor for z <= smagorinski_scaling_height (constant base value)
    #: Called 'hdiff_smag_fac' in mo_diffusion_nml.f90
    smagorinski_scaling_factor: float = 0.015

    #: Smagorinsky factor at z = smagorinski_scaling_height2: end of the linear segment and
    #: start of the quadratic segment. The linear slope is (factor2-factor1)/(height2-height1).
    #: Called 'hdiff_smag_fac2' in mo_diffusion_nml.f90
    smagorinski_scaling_factor2: float = 2e-6 * (
        1600.0 + 25000.0 + math.sqrt(1600.0 * (1600 + 50000.0))
    )

    #: Smagorinsky factor at z = smagorinski_scaling_height3: interior control point of the
    #: quadratic segment (height2 <= height3 <= height4), used to fit the quadratic coefficients.
    #: Called 'hdiff_smag_fac3' in mo_diffusion_nml.f90
    smagorinski_scaling_factor3: float = 0.0

    #: Smagorinsky factor for z >= smagorinski_scaling_height4 (constant asymptotic value).
    #: Also the third control point that defines the quadratic segment together with factor2 and factor3.
    #: Called 'hdiff_smag_fac4' in mo_diffusion_nml.f90
    smagorinski_scaling_factor4: float = 1.0

    #: Lower boundary of the linear segment: factor is constant at smagorinski_scaling_factor below this height.
    #: Called 'hdiff_smag_z' in mo_diffusion_nml.f90
    smagorinski_scaling_height: float = 32500.0

    #: Transition height between linear and quadratic segments.
    #: Called 'hdiff_smag_z2' in mo_diffusion_nml.f90
    smagorinski_scaling_height2: float = 1600.0 + 50000.0 + math.sqrt(1600.0 * (1600 + 50000.0))

    #: Interior control point height within the quadratic segment (height2 <= height3 <= height4).
    #: Called 'hdiff_smag_z3' in mo_diffusion_nml.f90
    smagorinski_scaling_height3: float = 50000.0

    #: Upper boundary of the quadratic segment: factor is constant at smagorinski_scaling_factor4 above this height.
    #: Called 'hdiff_smag_z4' in mo_diffusion_nml.f90
    smagorinski_scaling_height4: float = 90000.0

    #: If True, apply truly horizontal temperature diffusion over steep slopes
    #: Called 'l_zdiffu_t' in mo_nonhydrostatic_nml.f90
    zdiffu_t: bool = True

    apply_zdiffusion_t: typing.ClassVar[property] = _alias_property(
        "zdiffu_t", "apply_zdiffusion_t", bool
    )

    # from other namelists:
    # from parent namelist mo_nonhydrostatic_nml

    #: Number of dynamics substeps per fast-physics step
    #: Called 'ndyn_substeps' in mo_nonhydrostatic_nml.f90
    n_substeps: int = 5

    ndyn_substeps: typing.ClassVar[property] = _alias_property("n_substeps", "ndyn_substeps", int)

    # namelist mo_gridref_nml.f90

    #: Denominator for temperature boundary diffusion
    #: Called 'denom_diffu_t' in mo_gridref_nml.f90
    temperature_boundary_diffusion_denom: float = 135.0

    temperature_boundary_diffusion_denominator: typing.ClassVar[property] = _alias_property(
        "temperature_boundary_diffusion_denom", "temperature_boundary_diffusion_denominator", float
    )

    #: Denominator for velocity boundary diffusion
    #: Called 'denom_diffu_v' in mo_gridref_nml.f90
    velocity_boundary_diffusion_denom: float = 200.0

    velocity_boundary_diffusion_denominator: typing.ClassVar[property] = _alias_property(
        "velocity_boundary_diffusion_denom", "velocity_boundary_diffusion_denominator", float
    )

    # parameters from namelist: mo_interpol_nml.f90

    #: Parameter describing the lateral boundary nudging in limited area mode.
    #:
    #: Maximal value of the nudging coefficients used cell row bordering the boundary interpolation zone,
    #: from there nudging coefficients decay exponentially with `nudge_efold_width` in units of cell rows.
    #: Called 'nudge_max_coeff' in mo_interpol_nml.f90.
    max_nudging_coefficient: float = constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * 0.02

    #: Type of shear forcing used in turbulence
    #: Called 'itype_sher' in mo_turbdiff_nml.f90
    shear_type: TurbulenceShearForcingType = TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND

    #: Type of physics forcing
    #: Called 'iforcing' in mo_run_nml.f90
    iforcing: ForcingType = ForcingType.NO_FORCING

    #: Scaling factor for horizontal shear production term
    #: Called 'a_hshr' in mo_turbdiff_nml.f90
    a_hshr: float = 1.0

    #: Output flag for horizontal shear
    #: Called 'loutshs' in mo_turbdiff_nml.f90
    #: not a namelist parameter: its default is FALSE and only set to true in fortran `IF (.NOT. ldynamics)`
    loutshs: bool = False

    def __post_init__(self):
        self._validate()

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> DiffusionConfig:
        diffusion_nml = atmo_dict["diffusion_nml"]
        nonhydrostatic_nml = atmo_dict["nonhydrostatic_nml"]
        gridref_nml = atmo_dict["gridref_nml"]
        turbdiff_nml = atmo_dict["turbdiff_nml"]
        run_nml = atmo_dict["run_nml"]
        return cls(
            diffusion_type=DiffusionType(diffusion_nml["hdiff_order"]),
            hdiff_w=diffusion_nml["lhdiff_w"],
            hdiff_vn=diffusion_nml["lhdiff_vn"],
            hdiff_temp=diffusion_nml["lhdiff_temp"],
            hdiff_smag_w=fortran_config.list_to_value(diffusion_nml["lhdiff_smag_w"]),
            type_vn_diffu=SmagorinskyStencilType(diffusion_nml["itype_vn_diffu"]),
            smag_3d=fortran_config.list_to_value(diffusion_nml["lsmag_3d"]),
            type_t_diffu=TemperatureDiscretizationType(diffusion_nml["itype_t_diffu"]),
            hdiff_efdt_ratio=diffusion_nml["hdiff_efdt_ratio"],
            hdiff_w_efdt_ratio=diffusion_nml["hdiff_w_efdt_ratio"],
            smagorinski_scaling_factor=diffusion_nml["hdiff_smag_fac"],
            smagorinski_scaling_factor2=diffusion_nml["hdiff_smag_fac2"],
            smagorinski_scaling_factor3=diffusion_nml["hdiff_smag_fac3"],
            smagorinski_scaling_factor4=diffusion_nml["hdiff_smag_fac4"],
            smagorinski_scaling_height=diffusion_nml["hdiff_smag_z"],
            smagorinski_scaling_height2=diffusion_nml["hdiff_smag_z2"],
            smagorinski_scaling_height3=diffusion_nml["hdiff_smag_z3"],
            smagorinski_scaling_height4=diffusion_nml["hdiff_smag_z4"],
            n_substeps=nonhydrostatic_nml["ndyn_substeps"],
            zdiffu_t=nonhydrostatic_nml["l_zdiffu_t"],
            velocity_boundary_diffusion_denom=gridref_nml["denom_diffu_v"],
            temperature_boundary_diffusion_denom=gridref_nml["denom_diffu_t"],
            shear_type=TurbulenceShearForcingType(turbdiff_nml["itype_sher"]),
            iforcing=ForcingType(run_nml["iforcing"]),
            a_hshr=turbdiff_nml["a_hshr"],
            **overrides,
        )

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        if self.diffusion_type != DiffusionType.SMAGORINSKY_4TH_ORDER:
            raise NotImplementedError(
                "Only diffusion type 5 = `Smagorinsky diffusion with fourth-order background "
                "diffusion` is implemented"
            )

        if self.type_vn_diffu != SmagorinskyStencilType.DIAMOND_VERTICES:
            raise NotImplementedError(
                "Only type_vn_diffu 1 = `Smagorinsky diffusion with diamond stencil on vertices` is implemented"
            )

        if self.type_t_diffu != TemperatureDiscretizationType.HETEROGENEOUS:
            raise NotImplementedError(
                "Only type_t_diffu 2 = `Smagorinsky diffusion with heterogeneous discretization` is implemented"
            )

        if self.apply_smag_diff_to_vertical_wind:
            raise NotImplementedError("Smagorinsky diffusion for vertical wind is not implemented")

        if self.compute_3d_smag_coeff:
            raise NotImplementedError("3D Smagorinsky diffusion computation is not implemented")

        if self.shear_type not in (
            TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND,
            TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND,
            TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
        ):
            raise NotImplementedError(
                f"Turbulence Shear only {TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND} "
                f"and {TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND} "
                f"and {TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND} "
                f"implemented"
            )

    @functools.cached_property
    def substep_as_float(self):
        return float(self.ndyn_substeps)
