"""sed.calibrator.momentum module. Code for momentum calibration and distortion
correction. Mostly ported from https://github.com/mpes-kit/mpes.
"""
# pylint: disable=too-many-lines
import itertools as it
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import bokeh.palettes as bp
import bokeh.plotting as pbk
import dask.dataframe
import ipywidgets as ipw
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import xarray as xr
from bokeh.colors import RGB
from bokeh.io import output_notebook
from bokeh.palettes import Category10 as ColorCycle
from IPython.display import display
from matplotlib import cm
from numpy.linalg import norm
from scipy.interpolate import griddata
from symmetrize import pointops as po
from symmetrize import sym
from symmetrize import tps


class MomentumCorrector:  # pylint: disable=too-many-instance-attributes
    """
    Momentum distortion correction and momentum calibration workflow.
    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        data: Union[xr.DataArray, np.ndarray] = None,
        bin_ranges: List[Tuple] = None,
        rotsym: int = 6,
        config: dict = {},
    ):
        """
        Parameters:
            image: 3d array
                Volumetric band structure data.
            rotsym: int | 6
                Order of rotational symmetry.
        """
        self.image: np.ndarray = None
        self.img_ndim: int = None
        self.slice: np.ndarray = None
        self.slice_corrected: np.ndarray = None
        self.slice_transformed: np.ndarray = None
        self.bin_ranges: List[Tuple] = []

        if data is not None:
            self.load_data(data=data, bin_ranges=bin_ranges, rotsym=rotsym)

        self._config = config

        self.detector_ranges = self._config.get("momentum", {}).get(
            "detector_ranges",
            [[0, 2048], [0, 2048]],
        )

        self.rotsym = int(rotsym)
        self.rotsym_angle = int(360 / self.rotsym)
        self.arot = np.array([0] + [self.rotsym_angle] * (self.rotsym - 1))
        self.ascale = np.array([1.0] * self.rotsym)
        self.adjust_params: Dict[Any, Any] = {}
        self.peaks: np.ndarray = None
        self.pouter: np.ndarray = None
        self.pcent: Tuple[float, ...] = None
        self.pouter_ord: np.ndarray = None
        self.prefs: np.ndarray = None
        self.ptargs: np.ndarray = None
        self.csm_original: float = np.nan
        self.mdist: float = np.nan
        self.mcvdist: float = np.nan
        self.mvvdist: float = np.nan
        self.cvdist: float = np.nan
        self.vvdist: float = np.nan
        self.rdeform_field: np.ndarray = None
        self.cdeform_field: np.ndarray = None
        self.inverse_dfield: np.ndarray = None
        self.transformations: Dict[Any, Any] = {}
        self.calibration: Dict[Any, Any] = {}

    @property
    def features(self) -> dict:
        """Dictionary of detected features for the symmetrization process.
        ``self.features`` is a derived attribute from existing ones.
        """

        feature_dict = {
            "verts": np.asarray(self.__dict__.get("pouter_ord", [])),
            "center": np.asarray(self.__dict__.get("pcent", [])),
        }

        return feature_dict

    @property
    def symscores(self) -> dict:
        """Dictionary of symmetry-related scores."""

        sym_dict = {
            "csm_original": self.__dict__.get("csm_original", ""),
            "csm_current": self.__dict__.get("csm_current", ""),
            "arm_original": self.__dict__.get("arm_original", ""),
            "arm_current": self.__dict__.get("arm_current", ""),
        }

        return sym_dict

    def load_data(
        self,
        data: Union[xr.DataArray, np.ndarray],
        bin_ranges: List[Tuple] = None,
        rotsym: int = 6,
    ):
        """Load binned data into the momentum calibrator class

        Args:
            data (Union[xr.DataArray, np.ndarray]):
                2D or 3D data array, either as np.ndarray or xr.DataArray.
            bin_ranges (List[Tuple], optional):
                Binning ranges. Needs to be provided in case the data are given
                as np.ndarray. Otherwise, they are determined from the coords of
                the xr.DataArray Defaults to None.
            rotsym (int, optional): Rotational symmetry of the data. Defaults to 6.

        Raises:
            ValueError: Raised if the dimensions of the input data do not fit.
        """
        if isinstance(data, xr.DataArray):
            self.image = np.squeeze(data.data)
            self.bin_ranges = []
            for axis in data.coords:
                self.bin_ranges.append(
                    (
                        data.coords[axis][0].values,
                        data.coords[axis][-1].values,
                    ),
                )
        else:
            assert bin_ranges is not None
            self.image = np.squeeze(data)
            self.bin_ranges = bin_ranges

        self.img_ndim = self.image.ndim
        if (self.img_ndim > 3) or (self.img_ndim < 2):
            raise ValueError("The input image dimension need to be 2 or 3!")
        if self.img_ndim == 2:
            self.slice = self.image

        if self.slice is not None:
            self.slice_corrected = self.slice_transformed = self.slice

        self.rotsym = int(rotsym)
        self.rotsym_angle = int(360 / self.rotsym)
        self.arot = np.array([0] + [self.rotsym_angle] * (self.rotsym - 1))
        self.ascale = np.array([1.0] * self.rotsym)

    def select_slicer(  # pylint: disable=R0914
        self,
        plane: int = 0,
        width: int = 5,
        axis: int = 2,
        apply: bool = False,
    ):
        """Interactive panel to select (hyper)slice from a (hyper)volume.

        Parameters:
            plane: int
                initial value of the plane slider. Defaults to 0.
            width: int
                initial value of the width slider. Defaults to 5.
            axis: int
                Axis along which to slice the image. Defaults to 2.
            apply: bool
                Option to directly apply the values and select the slice.
        """
        matplotlib.use("module://ipympl.backend_nbagg")

        assert self.img_ndim == 3
        selector = slice(plane, plane + width)
        image = np.moveaxis(self.image, axis, 0)
        try:
            img_slice = image[selector, ...].sum(axis=0)
        except AttributeError:
            img_slice = image[selector, ...]

        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(img_slice.T, origin="lower", cmap="terrain_r")

        def update(plane: int, width: int):
            selector = slice(plane, plane + width)
            try:
                img_slice = image[selector, ...].sum(axis=0)
            except AttributeError:
                img_slice = image[selector, ...]
            img.set_data(img_slice.T)
            axmin = np.min(img_slice, axis=(0, 1))
            axmax = np.max(img_slice, axis=(0, 1))
            if axmin < axmax:
                img.set_clim(axmin, axmax)
            ax.set_title(f"Plane[{plane}:{plane+width}]")
            fig.canvas.draw_idle()

        update(plane, width)

        plane_slider = ipw.IntSlider(
            value=plane,
            min=0,
            max=self.image.shape[2] - width,
            step=1,
        )
        width_slider = ipw.IntSlider(value=width, min=1, max=20, step=1)

        ipw.interact(
            update,
            plane=plane_slider,
            width=width_slider,
        )

        def apply_fun(apply: bool):  # pylint: disable=unused-argument

            start = plane_slider.value
            stop = plane_slider.value + width_slider.value

            selector = slice(
                start,
                stop,
            )
            try:
                self.slice = image[selector, ...].sum(axis=0)
            except AttributeError:
                self.slice = image[selector, ...]

            if self.slice is not None:
                self.slice_corrected = self.slice_transformed = self.slice

            img.set_data(self.slice.T)
            axmin = np.min(self.slice, axis=(0, 1))
            axmax = np.max(self.slice, axis=(0, 1))
            if axmin < axmax:
                img.set_clim(axmin, axmax)
            ax.set_title(
                f"Plane[{start}:{stop}]",
            )
            fig.canvas.draw_idle()

            plane_slider.close()
            width_slider.close()
            apply_button.close()

        apply_button = ipw.Button(description="apply")
        display(apply_button)
        apply_button.on_click(apply_fun)

        plt.show()

        if apply:
            apply_fun(True)

    def select_slice(
        self,
        selector: Union[slice, List[int], int],
        axis: int = 2,
    ):
        """Select (hyper)slice from a (hyper)volume.

        Parameters:
            selector: slice object/list/int
                Selector along the specified axis to extract the slice (image). Use
                the construct slice(start, stop, step) to select a range of images
                and sum them. Use an integer to specify only a particular slice.
            axis: int | 2
                Axis along which to select the image.
        """

        if self.img_ndim > 2:
            immage = np.moveaxis(self.image, axis, 0)
            try:
                self.slice = immage[selector, ...].sum(axis=0)
            except AttributeError:
                self.slice = immage[selector, ...]

            if self.slice is not None:
                self.slice_corrected = self.slice_transformed = self.slice

        elif self.img_ndim == 2:
            raise ValueError("Input image dimension is already 2!")

    def add_features(
        self,
        peaks: np.ndarray,
        center_det: str = "centroidnn",
        direction: str = "ccw",
        symscores: bool = False,
        **kwds,
    ):
        """Add features as reference points provided as np.ndarray. If provided,
        detects the center of the points and orders the points.

        Args:
            peaks (np.ndarray):
                Array of peaks, possibly including a center peak. Its shape should be
                (nx2), where n is equal to the rotation symmetry, or the rotation
                symmetry+1, if the center is included
            center_det (str, optional):
                Method to use for detecting the center. Defaults to "centroidnn".
                If no center is included, set center_det to None.
            direction (str, optional):
                Direction for ordering the points. Defaults to "ccw".
            symscores (bool, optional):
                Option to calculate symmetry scores. Defaults to False.

        Raises:
            ValueError: Raised if the number of points does not match the rotsym.
        """
        if center_det is None:
            if peaks.shape[0] != self.rotsym:
                raise ValueError(
                    f"Found '{peaks.shape[0]}' points, ",
                    f"but '{self.rotsym}' required.",
                )
            self.pouter = peaks
            self.pcent = None
        else:
            if peaks.shape[0] != self.rotsym + 1:
                raise ValueError(
                    f"Found '{peaks.shape[0]}' points, ",
                    f"but '{self.rotsym+1}' required",
                )
            self.pcent, self.pouter = po.pointset_center(
                peaks,
                method=center_det,
                ret="cnc",
            )
            if isinstance(self.pcent, np.ndarray):
                self.pcent = tuple(val.item() for val in self.pcent)
        # Order the point landmarks
        self.pouter_ord = po.pointset_order(
            self.pouter,
            direction=direction,
        )

        # Calculate geometric distances
        if self.pcent is not None:
            self.calc_geometric_distances()

        if symscores is True:
            symtype = kwds.pop("symtype", "rotation")
            self.csm_original = self.calc_symmetry_scores(symtype=symtype)

        if self.rotsym == 6 and self.pcent is not None:
            self.mdist = (self.mcvdist + self.mvvdist) / 2
            self.mcvdist = self.mdist
            self.mvvdist = self.mdist

    def feature_extract(  # pylint: disable=too-many-arguments
        self,
        image: np.ndarray = None,
        direction: str = "ccw",
        feature_type: str = "points",
        center_det: str = "centroidnn",
        symscores: bool = True,
        **kwds,
    ):
        """Extract features from the selected 2D slice.
            Currently only point feature detection is implemented.

        Parameters:
        image: 2d array
            The image slice to extract features from.
        direction: str | 'ccw'
            The circular direction to reorder the features in ('cw' or 'ccw').
        type: str | 'points'
            The type of features to extract.
        center_det: str | 'centroidnn'
            Specification of center detection method ('centroidnn', 'centroid', None).
        **kwds: keyword arguments
            Extra keyword arguments for ``symmetrize.pointops.peakdetect2d()``.
        """

        if image is None:
            image = self.slice

        if feature_type == "points":

            # Detect the point landmarks
            self.peaks = po.peakdetect2d(image, **kwds)

            self.add_features(
                peaks=self.peaks,
                center_det=center_det,
                direction=direction,
                symscores=symscores,
                **kwds,
            )
        else:
            raise NotImplementedError

    def calc_geometric_distances(self):
        """Calculate geometric distances involving the center and the vertices.
        Distances calculated include center-vertex and nearest-neighbor vertex-vertex
        distances.
        """

        self.cvdist = po.cvdist(self.pouter_ord, self.pcent)
        self.mcvdist = self.cvdist.mean()
        self.vvdist = po.vvdist(self.pouter_ord)
        self.mvvdist = self.vvdist.mean()

    def calc_symmetry_scores(self, symtype: str = "rotation") -> float:
        """Calculate the symmetry scores from geometric quantities.

        **Paramters**\n
        symtype: str | 'rotation'
            Type of symmetry.
        """

        csm = po.csm(
            self.pcent,
            self.pouter_ord,
            rotsym=self.rotsym,
            type=symtype,
        )

        return csm

    def spline_warp_estimate(
        self,
        image: np.ndarray = None,
        include_center: bool = True,
        fixed_center: bool = True,
        interp_order: int = 1,
        **kwds,
    ) -> np.ndarray:
        """Estimate the spline deformation field using thin plate spline registration.

        **Parameters**\n
        image: 2D array
            Image slice to be corrected.
        include_center: bool | True
            Option to include the image center/centroid in the registration process.
        fixed_center: bool | True
            Option to have a fixed center during registration-based symmetrization.
        iterative: bool | False
            Option to use the iterative approach (may not work in all cases).
        interp_order: int | 1
            Order of interpolation (see ``scipy.ndimage.map_coordinates()``).
        update: bool | False
            Option to keep the spline-deformed image as corrected one.
        ret: bool | False
            Option to return corrected image slice.
        **kwds: keyword arguments
            :landmarks: list/array | self.pouter_ord
                Landmark positions (row, column) used for registration.
            :new_centers: dict | {}
                User-specified center positions for the reference and target sets.
                {'lmkcenter': (row, col), 'targcenter': (row, col)}
        """

        if image is None:
            image = self.slice

        if self.pouter_ord is None:
            self.pouter_ord = po.pointset_order(self.pouter)

        self.prefs = kwds.pop("landmarks", self.pouter_ord)
        self.ptargs = kwds.pop("targets", [])

        include_center = include_center and self.pcent is not None

        # Generate the target point set
        if not self.ptargs:
            self.ptargs = sym.rotVertexGenerator(
                self.pcent,
                fixedvertex=self.pouter_ord[0, :],
                arot=self.arot,
                direction=-1,
                scale=self.ascale,
                ret="all",
            )[1:, :]

        if include_center is True:
            # Include center of image pattern in the registration-based symmetrization
            if fixed_center is True:
                # Add the same center to both the reference and target sets

                self.prefs = np.column_stack((self.prefs.T, self.pcent)).T
                self.ptargs = np.column_stack((self.ptargs.T, self.pcent)).T

            else:  # Add different centers to the reference and target sets
                newcenters = kwds.pop("new_centers", {})
                self.prefs = np.column_stack(
                    (self.prefs.T, newcenters["lmkcenter"]),
                ).T
                self.ptargs = np.column_stack(
                    (self.ptargs.T, newcenters["targcenter"]),
                ).T

        # Non-iterative estimation of deformation field
        self.slice_corrected, splinewarp = tps.tpsWarping(
            self.prefs,
            self.ptargs,
            image,
            None,
            interp_order,
            ret="all",
            **kwds,
        )

        self.reset_deformation(image=image, coordtype="cartesian")

        self.update_deformation(
            splinewarp[0],
            splinewarp[1],
        )

        return self.slice_corrected

    def apply_correction(
        self,
        image: np.ndarray,
        axis: int,
        dfield: np.ndarray = None,
    ) -> np.ndarray:
        """Apply a 2D transform to a stack of 2D images (3D) along a specific axis.

        Parameters:
            image: np.ndarray
                Image which to apply the transformation to
            axis: int
                Axis for slice selection.
            use_composite_transform: bool | False
                Option to use the composite transform involving the rotation.
            update: bool | False
                Option to update the existing figure attributes.
            use_deform_field: bool | False
                Option to use deformation field for distortion correction.
            **kwds: keyword arguments
                ======= ========== =================================
                keyword data type  meaning
                ======= ========== =================================
                dfield  list/tuple row and column deformation field
                warping 2d array   2D transform correction matrix
                ======= ========== =================================
        """

        if dfield is None:
            dfield = np.asarray([self.rdeform_field, self.cdeform_field])

        image_corrected = sym.applyWarping(
            image,
            axis,
            warptype="deform_field",
            dfield=dfield,
        )

        return image_corrected

    def reset_deformation(self, **kwds):
        """Reset the deformation field."""

        image = kwds.pop("image", self.slice)
        coordtype = kwds.pop("coordtype", "cartesian")
        coordmat = sym.coordinate_matrix_2D(
            image,
            coordtype=coordtype,
            stackaxis=0,
        ).astype("float64")

        self.rdeform_field = coordmat[1, ...]
        self.cdeform_field = coordmat[0, ...]

    def update_deformation(self, rdeform: np.ndarray, cdeform: np.ndarray):
        """Update the deformation field by applying the provided column/row
        deformation fields.

        Parameters:
            rdeform, cdeform: 2D array, 2D array
                Row- and column-ordered deformation fields.
        """

        self.rdeform_field = ndi.map_coordinates(
            self.rdeform_field,
            [rdeform, cdeform],
            order=1,
            cval=np.nan,
        )
        self.cdeform_field = ndi.map_coordinates(
            self.cdeform_field,
            [rdeform, cdeform],
            order=1,
            cval=np.nan,
        )

    def coordinate_transform(  # pylint: disable=W0102, R0912, R0914
        self,
        transform_type: str,
        keep: bool = False,
        interp_order: int = 1,
        mapkwds: dict = {},
        **kwds,
    ) -> np.ndarray:
        """Apply a pixel-wise coordinate transform to the image
        by means of the deformation field.

        Parameters:
        type: str
            Type of deformation to apply to image slice.
        keep: bool | False
            Option to keep the specified coordinate transform.
        interp_order: int | 1
            Interpolation order for filling in missed pixels.
        mapkwds: dict | {}
            Additional arguments passed to ``scipy.ndimage.map_coordinates()``.
        **kwds: keyword arguments
            Additional arguments in specific deformation field.
            See ``symmetrize.sym`` module.
        """

        image = kwds.pop("image", self.slice)
        stackax = kwds.pop("stackaxis", 0)
        coordmat = sym.coordinate_matrix_2D(
            image,
            coordtype="homogeneous",
            stackaxis=stackax,
        )

        if transform_type == "translation":
            if "xtrans" in kwds and "ytrans" in kwds:
                tmp = kwds["ytrans"]
                kwds["ytrans"] = kwds["xtrans"]
                kwds["xtrans"] = tmp

            rdisp, cdisp = sym.translationDF(
                coordmat,
                stackaxis=stackax,
                ret="displacement",
                **kwds,
            )
        elif transform_type == "rotation":
            rdisp, cdisp = sym.rotationDF(
                coordmat,
                stackaxis=stackax,
                ret="displacement",
                **kwds,
            )
        elif transform_type == "rotation_auto":
            center = kwds.pop("center", self.pcent)
            # Estimate the optimal rotation angle using intensity symmetry
            angle_auto, _ = sym.sym_pose_estimate(
                image / image.max(),
                center=center,
                **kwds,
            )
            self.adjust_params = dictmerge(
                self.adjust_params,
                {"center": center, "angle": angle_auto},
            )
            rdisp, cdisp = sym.rotationDF(
                coordmat,
                stackaxis=stackax,
                ret="displacement",
                angle=angle_auto,
            )
        elif transform_type == "scaling":
            rdisp, cdisp = sym.scalingDF(
                coordmat,
                stackaxis=stackax,
                ret="displacement",
                **kwds,
            )
        elif (
            transform_type == "scaling_auto"
        ):  # Compare scaling to a reference image
            pass
        elif transform_type == "shearing":
            rdisp, cdisp = sym.shearingDF(
                coordmat,
                stackaxis=stackax,
                ret="displacement",
                **kwds,
            )
        elif transform_type == "homography":
            transform = kwds.pop("transform", np.eye(3))
            rdisp, cdisp = sym.compose_deform_field(
                coordmat,
                mat_transform=transform,
                stackaxis=stackax,
                ret="displacement",
                **kwds,
            )
        self.adjust_params = dictmerge(self.adjust_params, kwds)

        # Compute deformation field
        if stackax == 0:
            rdeform, cdeform = (
                coordmat[1, ...] + rdisp,
                coordmat[0, ...] + cdisp,
            )
        elif stackax == -1:
            rdeform, cdeform = (
                coordmat[..., 1] + rdisp,
                coordmat[..., 0] + cdisp,
            )

        # Resample image in the deformation field
        if (
            image is self.slice
        ):  # resample using all previous displacement fields
            total_rdeform = ndi.map_coordinates(
                self.rdeform_field,
                [rdeform, cdeform],
                order=1,
            )
            total_cdeform = ndi.map_coordinates(
                self.cdeform_field,
                [rdeform, cdeform],
                order=1,
            )
            slice_transformed = ndi.map_coordinates(
                image,
                [total_rdeform, total_cdeform],
                order=interp_order,
                **mapkwds,
            )
            self.slice_transformed = slice_transformed
        else:
            # if external image is provided, apply only the new addional tranformation
            slice_transformed = ndi.map_coordinates(
                image,
                [rdeform, cdeform],
                order=interp_order,
                **mapkwds,
            )

        # Combine deformation fields
        if keep is True:
            self.update_deformation(
                rdeform,
                cdeform,
            )

        return slice_transformed

    def pose_adjustment(  # pylint: disable=R0913, R0914, R0915
        self,
        scale: float = 1,
        xtrans: float = 0,
        ytrans: float = 0,
        angle: float = 0,
        apply: bool = False,
    ):
        """Interactive panel to adjust transformations that are applied to the image.
        Applies first a scaling, next a x/y translation, and last a rotation around
        the center of the image (pixel 256/256).

        Args:
            scale (float, optional):
                Initial value of the scaling slider. Defaults to 1.
            xtrans (float, optional):
                Initial value of the xtrans slider. Defaults to 0.
            ytrans (float, optional):
                Initial value of the ytrans slider. Defaults to 0.
            angle (float, optional):
                Initial value of the angle slider. Defaults to 0.
            apply (bool, optional):
                Option to directly apply the provided transformations.
                Defaults to False.
        """
        matplotlib.use("module://ipympl.backend_nbagg")
        source_image = self.slice_corrected

        transformed_image = source_image

        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(transformed_image.T, origin="lower", cmap="terrain_r")
        center = self._config.get("momentum", {}).get(
            "center_pixel",
            [256, 256],
        )
        ax.axvline(x=center[0])
        ax.axhline(y=center[1])

        def update(scale: float, xtrans: float, ytrans: float, angle: float):
            transformed_image = source_image
            if scale != 1:
                self.transformations["scale"] = scale
                transformed_image = self.coordinate_transform(
                    image=transformed_image,
                    transform_type="scaling",
                    xscale=scale,
                    yscale=scale,
                )
            if xtrans != 0 or ytrans != 0:
                self.transformations["xtrans"] = xtrans
                self.transformations["ytrans"] = ytrans
                transformed_image = self.coordinate_transform(
                    image=transformed_image,
                    transform_type="translation",
                    xtrans=xtrans,
                    ytrans=ytrans,
                )
            if angle != 0:
                self.transformations["angle"] = angle
                transformed_image = self.coordinate_transform(
                    image=transformed_image,
                    transform_type="rotation",
                    angle=angle,
                    center=center,
                )
            img.set_data(transformed_image.T)
            axmin = np.min(transformed_image, axis=(0, 1))
            axmax = np.max(transformed_image, axis=(0, 1))
            if axmin < axmax:
                img.set_clim(axmin, axmax)
            fig.canvas.draw_idle()

        update(scale=scale, xtrans=xtrans, ytrans=ytrans, angle=angle)

        scale_slider = ipw.FloatSlider(
            value=scale,
            min=0.8,
            max=1.2,
            step=0.01,
        )
        xtrans_slider = ipw.FloatSlider(
            value=xtrans,
            min=-200,
            max=200,
            step=1,
        )
        ytrans_slider = ipw.FloatSlider(
            value=ytrans,
            min=-200,
            max=200,
            step=1,
        )
        angle_slider = ipw.FloatSlider(value=angle, min=-180, max=180, step=1)
        ipw.interact(
            update,
            scale=scale_slider,
            xtrans=xtrans_slider,
            ytrans=ytrans_slider,
            angle=angle_slider,
        )

        def apply_func(apply: bool):  # pylint: disable=unused-argument
            if self.transformations.get("scale", 1) != 1:
                self.coordinate_transform(
                    transform_type="scaling",
                    xscale=self.transformations["scale"],
                    yscale=self.transformations["scale"],
                    keep=True,
                )
            if (
                self.transformations.get("xtrans", 0) != 0
                or self.transformations.get("ytrans", 0) != 0
            ):
                self.coordinate_transform(
                    transform_type="translation",
                    xtrans=self.transformations["xtrans"],
                    ytrans=self.transformations["ytrans"],
                    keep=True,
                )
            if self.transformations.get("angle", 0) != 0:
                self.coordinate_transform(
                    transform_type="rotation",
                    angle=self.transformations["angle"],
                    center=center,
                    keep=True,
                )

            img.set_data(self.slice_transformed.T)
            axmin = np.min(self.slice_transformed, axis=(0, 1))
            axmax = np.max(self.slice_transformed, axis=(0, 1))
            if axmin < axmax:
                img.set_clim(axmin, axmax)
            fig.canvas.draw_idle()

            plt.figure()
            subs = 20
            plt.title("Deformation field")
            plt.scatter(
                self.rdeform_field[::subs, ::subs].ravel(),
                self.cdeform_field[::subs, ::subs].ravel(),
                c="b",
            )
            plt.show()
            scale_slider.close()
            xtrans_slider.close()
            ytrans_slider.close()
            angle_slider.close()
            apply_button.close()

        apply_button = ipw.Button(description="apply")
        display(apply_button)
        apply_button.on_click(apply_func)

        plt.show()

        if apply:
            apply_func(True)

    def calc_inverse_dfield(self):
        """Calculate the inverse dfield from the cdeform and rdeform fields"""

        self.inverse_dfield = generate_inverse_dfield(
            self.rdeform_field,
            self.cdeform_field,
            self.bin_ranges,
            self.detector_ranges,
        )

        return self.inverse_dfield

    def view(  # pylint: disable=W0102, R0912, R0913, R0914
        self,
        image: np.ndarray = None,
        origin: str = "lower",
        cmap: str = "terrain_r",
        figsize: Tuple[int, int] = (4, 4),
        points: dict = None,
        annotated: bool = False,
        backend: str = "matplotlib",
        imkwds: dict = {},
        scatterkwds: dict = {},
        cross: bool = False,
        crosshair: bool = False,
        crosshair_radii: List[int] = [50, 100, 150],
        crosshair_thickness: int = 1,
        **kwds,
    ):
        """Display image slice with specified annotations.

        Parameters:
        origin: str | 'lower'
            Figure origin specification ('lower' or 'upper').
        cmap: str | 'terrain_r'
            Colormap specification.
        figsize: tuple/list | (4, 4)
            Figure size.
        points: dict | {}
            Points for annotation.
        annotated: bool | False
            Option for annotation.
        backend: str | 'matplotlib'
            Visualization backend specification.
            :'matplotlib': use static display rendered by matplotlib.
            :'bokeh': use interactive display rendered by bokeh.
        imkwd: dict | {}
            Keyword arguments for ``matplotlib.pyplot.imshow()``.
        crosshair: bool | False
            Display option to plot circles around center self.pcent.
            Works only in bokeh backend.
        crosshair_radii: list | [50,100,150]
            Radii of circles to plot when crosshair option is activated.
        crosshair_thickness: int | 1
            Thickness of crosshair circles.
        **kwds: keyword arguments
            General extra arguments for the plotting procedure.
        """

        if image is None:
            image = self.slice
        num_rows, num_cols = image.shape

        if points is None:
            points = self.features

        if annotated:
            tsr, tsc = kwds.pop("textshift", (3, 3))
            txtsize = kwds.pop("textsize", 12)

        if backend == "matplotlib":
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(image.T, origin=origin, cmap=cmap, **imkwds)

            if cross:
                center = self._config.get("momentum", {}).get(
                    "center_pixel",
                    [256, 256],
                )
                ax.axvline(x=center[0])
                ax.axhline(y=center[1])

            # Add annotation to the figure
            if annotated:
                for (
                    p_keys,  # pylint: disable=unused-variable
                    p_vals,
                ) in points.items():
                    try:
                        ax.scatter(p_vals[:, 0], p_vals[:, 1], **scatterkwds)
                    except IndexError:
                        try:
                            ax.scatter(p_vals[0], p_vals[1], **scatterkwds)
                        except IndexError:
                            pass

                    if p_vals.size > 2:
                        for i_pval, pval in enumerate(p_vals):
                            ax.text(
                                pval[0] + tsc,
                                pval[1] + tsr,
                                str(i_pval),
                                fontsize=txtsize,
                            )

        elif backend == "bokeh":

            output_notebook(hide_banner=True)
            colors = it.cycle(ColorCycle[10])
            ttp = [("(x, y)", "($x, $y)")]
            figsize = kwds.pop("figsize", (320, 300))
            palette = cm2palette(cmap)  # Retrieve palette colors
            fig = pbk.figure(
                plot_width=figsize[0],
                plot_height=figsize[1],
                tooltips=ttp,
                x_range=(0, num_rows),
                y_range=(0, num_cols),
            )
            fig.image(
                image=[image.T],
                x=0,
                y=0,
                dw=num_rows,
                dh=num_cols,
                palette=palette,
                **imkwds,
            )

            if annotated is True:
                for p_keys, p_vals in points.items():

                    try:
                        xcirc, ycirc = p_vals[:, 0], p_vals[:, 1]
                        fig.scatter(
                            xcirc,
                            ycirc,
                            size=8,
                            color=next(colors),
                            **scatterkwds,
                        )
                    except IndexError:
                        try:
                            xcirc, ycirc = p_vals[0], p_vals[1]
                            fig.scatter(
                                xcirc,
                                ycirc,
                                size=8,
                                color=next(colors),
                                **scatterkwds,
                            )
                        except IndexError:
                            pass
            if crosshair and self.pcent is not None:
                for radius in crosshair_radii:
                    fig.annulus(
                        x=[self.pcent[0]],
                        y=[self.pcent[1]],
                        inner_radius=radius - crosshair_thickness,
                        outer_radius=radius,
                        color="red",
                        alpha=0.6,
                    )

            pbk.show(fig)

    def calibrate(  # pylint: disable=W0102, R0913, R0914
        self,
        point_a: Union[np.ndarray, List[int]],
        point_b: Union[np.ndarray, List[int]],
        k_distance: float = None,
        image: np.ndarray = None,
        k_coord_a: Union[np.ndarray, List[float]] = None,
        k_coord_b: Union[np.ndarray, List[float]] = [0.0, 0.0],
        equiscale: bool = True,
    ) -> dict:
        """Momentum axes calibration using the pixel positions of two symmetry points
        (a and b) and the absolute coordinate of a single point (b), defaulted to
        [0., 0.]. All coordinates should be specified in the (row_index, column_index)
        format. See the equiscale option for details on the specifications of point
        coordinates.

        **Parameters**\n
        image: 2D array
            An energy cut of the band structure.
        pxla, pxlb: list/tuple/1D array
            Pixel coordinates of the two symmetry points (a and b). Point b has the
            default coordinates [0., 0.] (see below).
        k_ab: float | None
            The known momentum space distance between the two symmetry points.
        kcoorda: list/tuple/1D array | None
            Momentum coordinates of the symmetry point a.
        kcoordb: list/tuple/1D array | [0., 0.]
            Momentum coordinates of the symmetry point b (krow, kcol), default to
            k-space center.
        equiscale: bool | True
            Option to adopt equal scale along both the row and column directions.
            :True: Use a uniform scale for both x and y directions in the image
            coordinate system. This applies to the situation where the points a and b
            are (close to) parallel with one of the two image axes.
            :False: Calculate the momentum scale for both x and y directions
            separately. This applies to the situation where the points a and b are
            sufficiently different in both x and y directions in the image coordinate
            system.

        Returns:
        calibdict: dictionary with following entries:
            "axes": Tuple of 1D arrays
                Momentum coordinates of the row and column.
            "extent": list
                Extent of the two momentum axis (can be used directly in imshow).
            "grid": Tuple of 2D arrays
                Row and column mesh grid generated from the coordinates
                (can be used directly in pcolormesh).
        """

        if image is None:
            image = self.slice_corrected

        nrows, ncols = image.shape
        point_a, point_b = map(np.array, [point_a, point_b])

        rowdist = range(nrows) - point_b[0]
        coldist = range(ncols) - point_b[1]

        if equiscale is True:
            assert k_distance is not None
            # Use the same conversion factor along both x and y directions
            # (need k_distance)
            pixel_distance = norm(point_a - point_b)
            # Calculate the pixel to momentum conversion factor
            xratio = yratio = k_distance / pixel_distance

        else:
            assert k_coord_a is not None
            # Calculate the conversion factor along x and y directions separately
            # (need k_coord_a)
            kxb, kyb = k_coord_b
            kxa, kya = k_coord_a
            # Calculate the column- and row-wise conversion factor
            xratio = (kxa - kxb) / (point_a[0] - point_b[0])
            yratio = (kya - kyb) / (point_a[1] - point_b[1])

        k_row = rowdist * xratio + k_coord_b[0]
        k_col = coldist * yratio + k_coord_b[1]

        # Calculate other return parameters
        k_rowgrid, k_colgrid = np.meshgrid(k_row, k_col)

        # Assemble into return dictionary
        self.calibration = {}
        self.calibration["axes"] = (k_row, k_col)
        self.calibration["extent"] = (k_row[0], k_row[-1], k_col[0], k_col[-1])
        self.calibration["coeffs"] = (xratio, yratio)
        self.calibration["grid"] = (k_rowgrid, k_colgrid)
        self.calibration["x_center"] = point_b[0] - k_coord_b[0] / xratio
        self.calibration["y_center"] = point_b[1] - k_coord_b[1] / yratio
        # copy parameters for applying calibration
        try:
            self.calibration["rstart"] = self.bin_ranges[0][0]
            self.calibration["cstart"] = self.bin_ranges[1][0]
            self.calibration["rstep"] = (
                self.bin_ranges[0][1] - self.bin_ranges[0][0]
            ) / nrows
            self.calibration["cstep"] = (
                self.bin_ranges[1][1] - self.bin_ranges[1][0]
            ) / ncols
        except (AttributeError, IndexError):
            pass

        return self.calibration

    def apply_distortion_correction(  # pylint:disable=too-many-arguments
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        x_column: str = "X",
        y_column: str = "Y",
        new_x_column: str = "Xm",
        new_y_column: str = "Ym",
        **kwds: dict,
    ) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
        """Calculate and replace the X and Y values with their distortion-corrected
        version. This method can be reused.

        :Parameters:
            df : dataframe
                Dataframe to apply the distotion correction to.
            X, Y: | 'X', 'Y'
                Labels of the columns before momentum distortion correction.
            newX, newY: | 'Xm', 'Ym'
                Labels of the columns after momentum distortion correction.

        :Return:
            dataframe with added columns
        """

        if "dfield" in kwds:
            dfield = np.asarray(kwds.pop("dfield"))
        elif "rdeform_field" in kwds and "cdeform_field" in kwds:
            rdeform_field = np.asarray(kwds.pop("rdeform_field"))
            cdeform_field = np.asarray(kwds.pop("cdeform_field"))
            dfield = generate_inverse_dfield(
                rdeform_field,
                cdeform_field,
                self.bin_ranges,
                self.detector_ranges,
            )
        else:
            if self.inverse_dfield is not None:
                dfield = self.inverse_dfield
            else:
                self.inverse_dfield = generate_inverse_dfield(
                    self.rdeform_field,
                    self.cdeform_field,
                    self.bin_ranges,
                    self.detector_ranges,
                )
                dfield = self.inverse_dfield

        out_df = df.map_partitions(
            apply_dfield,
            dfield,
            x_column=x_column,
            y_column=y_column,
            new_x_column=new_x_column,
            new_y_column=new_y_column,
            **kwds,
        )
        return out_df

    def append_k_axis(  # pylint: disable=too-many-arguments
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        x_column: str = "Xm",
        y_column: str = "Ym",
        new_x_column: str = "kx",
        new_y_column: str = "ky",
        **kwds,
    ) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
        """Calculate and append the k axis coordinates (kx, ky) to the events dataframe.

        :Parameters:
            df : dataframe
                Dataframe to apply the distotion correction to.
            X, Y: | 'Xm', 'Ym'
                Labels of the source columns
            newX, newY: | 'ky', 'ky'
                Labels of the destination columns
            **kwds:
                additional keywords for the momentum conversion

        :Return:
            dataframe with added columns
        """

        try:
            (
                df[new_x_column],
                df[new_y_column],
            ) = detector_coordiantes_2_k_koordinates(
                r_det=df[x_column],
                c_det=df[y_column],
                r_start=self.calibration["rstart"],
                c_start=self.calibration["cstart"],
                r_center=self.calibration["x_center"],
                c_center=self.calibration["y_center"],
                r_conversion=self.calibration["coeffs"][0],
                c_conversion=self.calibration["coeffs"][1],
                r_step=self.calibration["rstep"],
                c_step=self.calibration["cstep"],
            )
        except KeyError:
            (
                df[new_x_column],
                df[new_y_column],
            ) = detector_coordiantes_2_k_koordinates(
                r_det=df[x_column],
                c_det=df[y_column],
                **kwds,
            )

        return df


def cm2palette(cmap_name):
    """Convert certain matplotlib colormap (cm) to bokeh palette.

    **Parameter**\n
    cmap_name: str
        Name of the colormap/palette.

    **Return**\n
    palette: list
        List of colors in hex representation (a bokoeh palette).
    """

    if cmap_name in bp.all_palettes.keys():
        palette_func = getattr(bp, cmap_name)
        palette = palette_func

    else:
        palette_func = getattr(cm, cmap_name)
        mpl_cm_rgb = (255 * palette_func(range(256))).astype("int")
        palette = [RGB(*tuple(rgb)).to_hex() for rgb in mpl_cm_rgb]

    return palette


def dictmerge(
    main_dict: dict,
    other_entries: Union[List[dict], Tuple[dict], dict],
) -> dict:
    """
    Merge a dictionary with other dictionaries.

    **Parameters**\n
    main_dict: dict
        Main dictionary.
    others: list/tuple/dict
        Other dictionary or composite dictionarized elements.

    **Return**\n
    Dmain_dict: dict
        Merged dictionary.
    """

    if isinstance(
        other_entries,
        (
            list,
            tuple,
        ),
    ):  # Merge main_dict with a list or tuple of dictionaries
        for oth in other_entries:
            main_dict = {**main_dict, **oth}

    elif isinstance(other_entries, dict):  # Merge D with a single dictionary
        main_dict = {**main_dict, **other_entries}

    return main_dict


def detector_coordiantes_2_k_koordinates(  # pylint: disable=too-many-arguments
    r_det: float,
    c_det: float,
    r_start: float,
    c_start: float,
    r_center: float,
    c_center: float,
    r_conversion: float,
    c_conversion: float,
    r_step: float,
    c_step: float,
) -> Tuple[float, float]:
    """
    Conversion from detector coordinates (rdet, cdet) to momentum coordinates (kr, kc).
    """

    r_det0 = r_start + r_step * r_center
    c_det0 = c_start + c_step * c_center
    k_r = r_conversion * ((r_det - r_det0) / r_step)
    k_c = c_conversion * ((c_det - c_det0) / c_step)

    return (k_r, k_c)


def apply_dfield(  # pylint: disable=too-many-arguments
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    dfield: np.ndarray,
    x_column: str,
    y_column: str,
    new_x_column: str,
    new_y_column: str,
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """
    Application of the inverse displacement-field to the dataframe coordinates

    :Parameters:
        df : dataframe
            Dataframe to apply the distotion correction to.
        dfield:
            The distortion correction field. 3D matrix,
            with column and row distortion fields stacked along the first dimension.
        X, Y:
            Labels of the source columns
        newX, newY:
            Labels of the destination columns

    :Return:
        dataframe with added columns
    """

    x = df[x_column]
    y = df[y_column]

    df[new_x_column], df[new_y_column] = (
        dfield[0, np.int16(x), np.int16(y)],
        dfield[1, np.int16(x), np.int16(y)],
    )
    return df


def generate_inverse_dfield(
    rdeform_field: np.ndarray,
    cdeform_field: np.ndarray,
    bin_ranges: List[Tuple],
    detector_ranges: List[Tuple],
) -> np.ndarray:
    """
    Generate inverse deformation field using inperpolation with griddata.
    Assuming the binning range of the input ``rdeform_field`` and ``cdeform_field``
    covers the whole detector.

    :Parameters:
        rdeform_field, cdeform_field: 2d array, 2d array
            Row-wise and column-wise deformation fields.

    :Return:
        dfield:
            inverse 3D deformation field stacked along the first dimension
    """

    print(
        "Calculating inverse deformation field, this might take a moment...",
    )

    # Interpolate to 2048x2048 grid of the detector coordinates
    r_mesh, c_mesh = np.meshgrid(
        np.linspace(
            detector_ranges[0][0],
            cdeform_field.shape[0],
            detector_ranges[0][1],
            endpoint=False,
        ),
        np.linspace(
            detector_ranges[1][0],
            cdeform_field.shape[1],
            detector_ranges[1][1],
            endpoint=False,
        ),
        sparse=False,
        indexing="ij",
    )

    bin_step = (
        np.asarray(bin_ranges)[0:2][:, 1] - np.asarray(bin_ranges)[0:2][:, 0]
    ) / cdeform_field.shape
    rc_position = []  # row/column position in c/rdeform_field
    r_dest = []  # destination pixel row position
    c_dest = []  # destination pixel column position
    for i in np.arange(cdeform_field.shape[0]):
        for j in np.arange(cdeform_field.shape[1]):
            if not np.isnan(rdeform_field[i, j]) and not np.isnan(
                cdeform_field[i, j],
            ):
                rc_position.append(
                    [
                        rdeform_field[i, j] + bin_ranges[0][0] / bin_step[0],
                        cdeform_field[i, j] + bin_ranges[0][0] / bin_step[1],
                    ],
                )
                r_dest.append(
                    bin_step[0] * i + bin_ranges[0][0],
                )
                c_dest.append(
                    bin_step[1] * j + bin_ranges[1][0],
                )

    inv_rdeform_field = griddata(
        np.asarray(rc_position),
        r_dest,
        (r_mesh, c_mesh),
    )

    inv_cdeform_field = griddata(
        np.asarray(rc_position),
        c_dest,
        (r_mesh, c_mesh),
    )

    inverse_dfield = np.asarray([inv_rdeform_field, inv_cdeform_field])

    return inverse_dfield
