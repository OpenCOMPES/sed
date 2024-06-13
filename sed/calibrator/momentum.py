"""sed.calibrator.momentum module. Code for momentum calibration and distortion
correction. Mostly ported from https://github.com/mpes-kit/mpes.
"""
import itertools as it
from copy import deepcopy
from datetime import datetime
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
from joblib import delayed
from joblib import Parallel
from matplotlib import cm
from numpy.linalg import norm
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from symmetrize import pointops as po
from symmetrize import sym
from symmetrize import tps


class MomentumCorrector:
    """
    Momentum distortion correction and momentum calibration workflow functions.

    Args:
        data (Union[xr.DataArray, np.ndarray], optional): Multidimensional hypervolume
            containing the data. Defaults to None.
        bin_ranges (List[Tuple], optional): Binning ranges of the data volume, if
            provided as np.ndarray. Defaults to None.
        rotsym (int, optional): Rotational symmetry of the data. Defaults to 6.
        config (dict, optional): Config dictionary. Defaults to None.
    """

    def __init__(
        self,
        data: Union[xr.DataArray, np.ndarray] = None,
        bin_ranges: List[Tuple] = None,
        rotsym: int = 6,
        config: dict = None,
    ):
        """Constructor of the MomentumCorrector class.

        Args:
            data (Union[xr.DataArray, np.ndarray], optional): Multidimensional
                hypervolume containing the data. Defaults to None.
            bin_ranges (List[Tuple], optional): Binning ranges of the data volume,
                if provided as np.ndarray. Defaults to None.
            rotsym (int, optional): Rotational symmetry of the data. Defaults to 6.
            config (dict, optional): Config dictionary. Defaults to None.
        """
        if config is None:
            config = {}

        self._config = config

        self.image: np.ndarray = None
        self.img_ndim: int = None
        self.slice: np.ndarray = None
        self.slice_corrected: np.ndarray = None
        self.slice_transformed: np.ndarray = None
        self.bin_ranges: List[Tuple] = self._config["momentum"].get("bin_ranges", [])

        if data is not None:
            self.load_data(data=data, bin_ranges=bin_ranges)

        self.detector_ranges = self._config["momentum"]["detector_ranges"]

        self.rotsym = int(rotsym)
        self.rotsym_angle = int(360 / self.rotsym)
        self.arot = np.array([0] + [self.rotsym_angle] * (self.rotsym - 1))
        self.ascale = np.array([1.0] * self.rotsym)
        self.peaks: np.ndarray = None
        self.include_center: bool = False
        self.use_center: bool = False
        self.pouter: np.ndarray = None
        self.pcent: Tuple[float, ...] = None
        self.pouter_ord: np.ndarray = None
        self.prefs: np.ndarray = None
        self.ptargs: np.ndarray = None
        self.csm_original: float = np.nan
        self.mdist: float = np.nan
        self.mcvdist: float = np.nan
        self.mvvdist: float = np.nan
        self.cvdist: np.ndarray = np.array(np.nan)
        self.vvdist: np.ndarray = np.array(np.nan)
        self.rdeform_field: np.ndarray = None
        self.cdeform_field: np.ndarray = None
        self.rdeform_field_bkp: np.ndarray = None
        self.cdeform_field_bkp: np.ndarray = None
        self.inverse_dfield: np.ndarray = None
        self.dfield_updated: bool = False
        self.transformations: Dict[str, Any] = self._config["momentum"].get("transformations", {})
        self.correction: Dict[str, Any] = self._config["momentum"].get("correction", {})
        self.adjust_params: Dict[str, Any] = {}
        self.calibration: Dict[str, Any] = self._config["momentum"].get("calibration", {})

        self.x_column = self._config["dataframe"]["x_column"]
        self.y_column = self._config["dataframe"]["y_column"]
        self.corrected_x_column = self._config["dataframe"]["corrected_x_column"]
        self.corrected_y_column = self._config["dataframe"]["corrected_y_column"]
        self.kx_column = self._config["dataframe"]["kx_column"]
        self.ky_column = self._config["dataframe"]["ky_column"]

        self._state: int = 0

    @property
    def features(self) -> dict:
        """Dictionary of detected features for the symmetrization process.
        ``self.features`` is a derived attribute from existing ones.

        Returns:
            dict: Dict containing features "verts" and "center".
        """
        feature_dict = {
            "verts": np.asarray(self.__dict__.get("pouter_ord", [])),
            "center": np.asarray(self.__dict__.get("pcent", [])),
        }

        return feature_dict

    @property
    def symscores(self) -> dict:
        """Dictionary of symmetry-related scores.

        Returns:
            dict: Dictionary containing symmetry scores.
        """
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
    ):
        """Load binned data into the momentum calibrator class

        Args:
            data (Union[xr.DataArray, np.ndarray]):
                2D or 3D data array, either as np.ndarray or xr.DataArray.
            bin_ranges (List[Tuple], optional):
                Binning ranges. Needs to be provided in case the data are given
                as np.ndarray. Otherwise, they are determined from the coords of
                the xr.DataArray. Defaults to None.

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
                        2 * data.coords[axis][-1].values - data.coords[axis][-2].values,  # endpoint
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

    def select_slicer(
        self,
        plane: int = 0,
        width: int = 5,
        axis: int = 2,
        apply: bool = False,
    ):
        """Interactive panel to select (hyper)slice from a (hyper)volume.

        Args:
            plane (int, optional): initial value of the plane slider. Defaults to 0.
            width (int, optional): initial value of the width slider. Defaults to 5.
            axis (int, optional): Axis along which to slice the image. Defaults to 2.
            apply (bool, optional):  Option to directly apply the values and select the
                slice. Defaults to False.
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

        def apply_fun(apply: bool):  # noqa: ARG001
            start = plane_slider.value
            stop = plane_slider.value + width_slider.value

            selector = slice(
                start,
                stop,
            )
            self.select_slice(selector=selector, axis=axis)

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

        Args:
            selector (Union[slice, List[int], int]):
                Selector along the specified axis to extract the slice (image). Use
                the construct slice(start, stop, step) to select a range of images
                and sum them. Use an integer to specify only a particular slice.
            axis (int, optional): Axis along which to select the image. Defaults to 2.

        Raises:
            ValueError: Raised if self.image is already 2D.
        """
        if self.img_ndim > 2:
            image = np.moveaxis(self.image, axis, 0)
            try:
                self.slice = image[selector, ...].sum(axis=0)
            except AttributeError:
                self.slice = image[selector, ...]

            if self.slice is not None:
                self.slice_corrected = self.slice_transformed = self.slice

        elif self.img_ndim == 2:
            raise ValueError("Input image dimension is already 2!")

    def add_features(
        self,
        features: np.ndarray,
        direction: str = "ccw",
        rotsym: int = 6,
        symscores: bool = True,
        **kwds,
    ):
        """Add features as reference points provided as np.ndarray. If provided,
        detects the center of the points and orders the points.

        Args:
            features (np.ndarray):
                Array of landmarks, possibly including a center peak. Its shape should
                be (n,2), where n is equal to the rotation symmetry, or the rotation
                symmetry+1, if the center is included.
            direction (str, optional):
                Direction for ordering the points. Defaults to "ccw".
            symscores (bool, optional):
                Option to calculate symmetry scores. Defaults to False.
            **kwds: Keyword arguments.

                - **symtype** (str): Type of symmetry scores to calculte
                  if symscores is True. Defaults to "rotation".

        Raises:
            ValueError: Raised if the number of points does not match the rotsym.
        """
        self.rotsym = int(rotsym)
        self.rotsym_angle = int(360 / self.rotsym)
        self.arot = np.array([0] + [self.rotsym_angle] * (self.rotsym - 1))
        self.ascale = np.array([1.0] * self.rotsym)

        if features.shape[0] == self.rotsym:  # assume no center present
            self.pcent, self.pouter = po.pointset_center(
                features,
                method="centroid",
            )
            self.include_center = False
        elif features.shape[0] == self.rotsym + 1:  # assume center included
            self.pcent, self.pouter = po.pointset_center(
                features,
                method="centroidnn",
            )
            self.include_center = True
        else:
            raise ValueError(
                f"Found {features.shape[0]} points, ",
                f"but {self.rotsym} or {self.rotsym+1} (incl.center) required.",
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

    def feature_extract(
        self,
        image: np.ndarray = None,
        direction: str = "ccw",
        feature_type: str = "points",
        rotsym: int = 6,
        symscores: bool = True,
        **kwds,
    ):
        """Extract features from the selected 2D slice.
        Currently only point feature detection is implemented.

        Args:
            image (np.ndarray, optional):
                The (2D) image slice to extract features from.
                Defaults to self.slice
            direction (str, optional):
                The circular direction to reorder the features in ('cw' or 'ccw').
                Defaults to "ccw".
            feature_type (str, optional):
                The type of features to extract. Defaults to "points".
            rotsym (int, optional): Rotational symmetry of the data. Defaults to 6.
            symscores (bool, optional):
                Option for calculating symmetry scores. Defaults to True.
            **kwds:
                Extra keyword arguments for ``symmetrize.pointops.peakdetect2d()``.

        Raises:
            NotImplementedError:
                Raised for undefined feature_types.
        """
        if image is None:
            if self.slice is not None:
                image = self.slice
            else:
                raise ValueError("No image loaded for feature extraction!")

        if feature_type == "points":
            # Detect the point landmarks
            self.peaks = po.peakdetect2d(image, **kwds)

            self.add_features(
                features=self.peaks,
                direction=direction,
                rotsym=rotsym,
                symscores=symscores,
                **kwds,
            )
        else:
            raise NotImplementedError

    def feature_select(
        self,
        image: np.ndarray = None,
        features: np.ndarray = None,
        include_center: bool = True,
        rotsym: int = 6,
        apply: bool = False,
        **kwds,
    ):
        """Extract features from the selected 2D slice.
        Currently only point feature detection is implemented.

        Args:
            image (np.ndarray, optional):
                The (2D) image slice to extract features from.
                Defaults to self.slice
            include_center (bool, optional):
                Option to include the image center/centroid in the registration
                process. Defaults to True.
            features (np.ndarray, optional):
                Array of landmarks, possibly including a center peak. Its shape should
                be (n,2), where n is equal to the rotation symmetry, or the rotation
                symmetry+1, if the center is included.
                If omitted, an array filled with zeros is generated.
            rotsym (int, optional): Rotational symmetry of the data. Defaults to 6.
            apply (bool, optional): Option to directly store the features in the class.
                Defaults to False.
            **kwds:
                Extra keyword arguments for ``symmetrize.pointops.peakdetect2d()``.

        Raises:
            ValueError: If no valid image is found from which to ge the coordinates.
        """
        matplotlib.use("module://ipympl.backend_nbagg")
        if image is None:
            if self.slice is not None:
                image = self.slice
            else:
                raise ValueError("No valid image loaded!")

        fig, ax = plt.subplots(1, 1)
        ax.imshow(image.T, origin="lower", cmap="terrain_r")

        if features is None:
            features = np.zeros((rotsym + (include_center), 2))

        markers = []
        for peak in features:
            markers.append(ax.plot(peak[0], peak[1], "o")[0])

        def update_point_no(
            point_no: int,
        ):
            fig.canvas.draw_idle()

            point_x = features[point_no][0]
            point_y = features[point_no][1]

            point_input_x.value = point_x
            point_input_y.value = point_y

        def update_point_pos(
            point_x: float,
            point_y: float,
        ):
            fig.canvas.draw_idle()
            point_no = point_no_input.value
            features[point_no][0] = point_x
            features[point_no][1] = point_y

            markers[point_no].set_xdata(point_x)
            markers[point_no].set_ydata(point_y)

        point_no_input = ipw.Dropdown(
            options=range(features.shape[0]),
            description="Point:",
        )

        point_input_x = ipw.FloatText(features[0][0])
        point_input_y = ipw.FloatText(features[0][1])
        ipw.interact(
            update_point_no,
            point_no=point_no_input,
        )
        ipw.interact(
            update_point_pos,
            point_y=point_input_y,
            point_x=point_input_x,
        )

        def onclick(event):
            point_input_x.value = event.xdata
            point_input_y.value = event.ydata
            point_no_input.value = (point_no_input.value + 1) % features.shape[0]

        cid = fig.canvas.mpl_connect("button_press_event", onclick)

        def apply_func(apply: bool):  # noqa: ARG001
            fig.canvas.mpl_disconnect(cid)

            point_no_input.close()
            point_input_x.close()
            point_input_y.close()
            apply_button.close()

            fig.canvas.draw_idle()

            self.add_features(
                features=features,
                rotsym=rotsym,
                **kwds,
            )

        apply_button = ipw.Button(description="apply")
        display(apply_button)
        apply_button.on_click(apply_func)

        if apply:
            apply_func(True)

        plt.show()

    def calc_geometric_distances(self) -> None:
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

        Args:
            symtype (str, optional): Type of symmetry score to calculate.
                Defaults to "rotation".

        Returns:
            float: Calculated symmetry score.
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
        use_center: bool = None,
        fixed_center: bool = True,
        interp_order: int = 1,
        ascale: Union[float, list, tuple, np.ndarray] = None,
        verbose: bool = True,
        **kwds,
    ) -> np.ndarray:
        """Estimate the spline deformation field using thin plate spline registration.

        Args:
            image (np.ndarray, optional):
                2D array. Image slice to be corrected. Defaults to self.slice.
            use_center (bool, optional):
                Option to use the image center/centroid in the registration
                process. Defaults to config value, or True.
            fixed_center (bool, optional):
                Option to have a fixed center during registration-based
                symmetrization. Defaults to True.
            interp_order (int, optional):
                Order of interpolation (see ``scipy.ndimage.map_coordinates()``).
                Defaults to 1.
            ascale: (Union[float, np.ndarray], optional): Scale parameter determining a realtive
                scale for each symmetry feature. If provided as single float, rotsym has to be 4.
                This parameter describes the relative scaling between the two orthogonal symmetry
                directions (for an orthorhombic system). This requires the correction points to be
                located along the principal axes (X/Y points of the Brillouin zone). Otherwise, an
                array with ``rotsym`` elements is expected, containing relative scales for each
                feature. Defaults to an array of equal scales.
            verbose (bool, optional): Option to report the used landmarks for correction.
                Defaults to True.
            **kwds: keyword arguments:

                - **landmarks**: (list/array): Landmark positions (row, column) used
                  for registration. Defaults to  self.pouter_ord
                - **targets**: (list/array): Target positions (row, column) used for
                  registration. If empty, it will be generated by
                  ``symmetrize.rotVertexGenerator()``.
                - **new_centers**: (dict): User-specified center positions for the
                  reference and target sets. {'lmkcenter': (row, col),
                  'targcenter': (row, col)}
        Returns:
            np.ndarray: The corrected image.
        """
        if image is None:
            if self.slice is not None:
                image = self.slice
            else:
                image = np.zeros(self._config["momentum"]["bins"][0:2])
                self.bin_ranges = self._config["momentum"]["ranges"]

        if self.pouter_ord is None:
            if self.pouter is not None:
                self.pouter_ord = po.pointset_order(self.pouter)
                self.correction["creation_date"] = datetime.now().timestamp()
                self.correction["creation_date"] = datetime.now().timestamp()
            else:
                try:
                    features = np.asarray(
                        self.correction["feature_points"],
                    )
                    rotsym = self.correction["rotation_symmetry"]
                    include_center = self.correction["include_center"]
                    if not include_center and len(features) > rotsym:
                        features = features[:rotsym, :]
                    ascale = self.correction.get("ascale", None)
                    if ascale is not None:
                        ascale = np.asarray(ascale)

                    if verbose:
                        if "creation_date" in self.correction:
                            datestring = datetime.fromtimestamp(
                                self.correction["creation_date"],
                            ).strftime(
                                "%m/%d/%Y, %H:%M:%S",
                            )
                            print(
                                "No landmarks defined, using momentum correction parameters "
                                f"generated on {datestring}",
                            )
                        else:
                            print(
                                "No landmarks defined, using momentum correction parameters "
                                "from config.",
                            )
                except KeyError as exc:
                    raise ValueError(
                        "No valid landmarks defined, and no landmarks found in configuration!",
                    ) from exc

                self.add_features(features=features, rotsym=rotsym, include_center=include_center)

        else:
            self.correction["creation_date"] = datetime.now().timestamp()

        if ascale is not None:
            if isinstance(ascale, (int, float, np.floating, np.integer)):
                if self.rotsym != 4:
                    raise ValueError(
                        "Providing ascale as scalar number is only valid for 'rotsym'==4.",
                    )
                self.ascale = np.array([1.0, ascale, 1.0, ascale])
            elif isinstance(ascale, (tuple, list, np.ndarray)):
                if len(ascale) != len(self.ascale):
                    raise ValueError(
                        f"ascale needs to be of length 'rotsym', but has length {len(ascale)}.",
                    )
                self.ascale = np.asarray(ascale)
            else:
                raise TypeError(
                    (
                        "ascale needs to be a single number or a list/tuple/np.ndarray of length ",
                        f"'rotsym' ({self.rotsym})!",
                    ),
                )

        if use_center is None:
            try:
                use_center = self.correction["use_center"]
            except KeyError:
                use_center = True
        self.use_center = use_center

        self.prefs = kwds.pop("landmarks", self.pouter_ord)
        self.ptargs = kwds.pop("targets", [])

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

        if use_center is True:
            # Use center of image pattern in the registration-based symmetrization
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
        corrected_image, splinewarp = tps.tpsWarping(
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

        # save backup copies to reset transformations
        self.rdeform_field_bkp = self.rdeform_field
        self.cdeform_field_bkp = self.cdeform_field

        self.correction["outer_points"] = self.pouter_ord
        self.correction["center_point"] = np.asarray(self.pcent)
        self.correction["reference_points"] = self.prefs
        self.correction["target_points"] = self.ptargs
        self.correction["rotation_symmetry"] = self.rotsym
        self.correction["use_center"] = self.use_center
        self.correction["include_center"] = self.include_center
        if self.include_center:
            self.correction["feature_points"] = np.concatenate(
                (self.pouter_ord, np.asarray([self.pcent])),
            )
        else:
            self.correction["feature_points"] = self.pouter_ord
        self.correction["ascale"] = self.ascale

        if self.slice is not None:
            self.slice_corrected = corrected_image

        if verbose:
            print("Calulated thin spline correction based on the following landmarks:")
            print(f"pouter: {self.pouter}")
            if use_center:
                print(f"pcent: {self.pcent}")

        return corrected_image

    def apply_correction(
        self,
        image: np.ndarray,
        axis: int,
        dfield: np.ndarray = None,
    ) -> np.ndarray:
        """Apply a 2D transform to a stack of 2D images (3D) along a specific axis.

        Args:
            image (np.ndarray): Image which to apply the transformation to
            axis (int): Axis for slice selection.
            dfield (np.ndarray, optional): row and column deformation field.
                Defaults to [self.rdeform_field, self.cdeformfield].

        Returns:
            np.ndarray: The corrected image.
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
        """Reset the deformation field.

        Args:
            **kwds: keyword arguments:

                - **image**: the image to base the deformation fields on. Its sizes are
                  used. Defaults to self.slice
                - **coordtype**: The coordinate system to use. Defaults to 'cartesian'.
        """
        image = kwds.pop("image", self.slice)
        coordtype = kwds.pop("coordtype", "cartesian")
        coordmat = sym.coordinate_matrix_2D(
            image,
            coordtype=coordtype,
            stackaxis=0,
        ).astype("float64")

        self.rdeform_field = coordmat[1, ...]
        self.cdeform_field = coordmat[0, ...]

        self.dfield_updated = True

    def update_deformation(self, rdeform: np.ndarray, cdeform: np.ndarray):
        """Update the class deformation field by applying the provided column/row
        deformation fields.

        Parameters:
            rdeform (np.ndarray): 2D array of row-ordered deformation field.
            cdeform (np.ndarray): 2D array of column-ordered deformation field.
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

        self.dfield_updated = True

    def coordinate_transform(
        self,
        transform_type: str,
        keep: bool = False,
        interp_order: int = 1,
        mapkwds: dict = None,
        **kwds,
    ) -> np.ndarray:
        """Apply a pixel-wise coordinate transform to the image
        by means of the deformation field.

        Args:
            transform_type (str): Type of deformation to apply to image slice. Possible
                values are:

                - translation.
                - rotation.
                - rotation_auto.
                - scaling.
                - scaling_auto.
                - homomorphy.

            keep (bool, optional): Option to keep the specified coordinate transform in
                the class. Defaults to False.
            interp_order (int, optional): Interpolation order for filling in missed
                pixels. Defaults to 1.
            mapkwds (dict, optional): Additional arguments passed to
                ``scipy.ndimage.map_coordinates()``. Defaults to None.
            **kwds: keyword arguments.
                Additional arguments in specific deformation field.
                See ``symmetrize.sym`` module.
        Returns:
            np.ndarray: The corrected image.
        """
        if mapkwds is None:
            mapkwds = {}

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
        elif transform_type == "scaling_auto":  # Compare scaling to a reference image
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
        if image is self.slice:  # resample using all previous displacement fields
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
            self.adjust_params["applied"] = True
            self.adjust_params = dictmerge(self.adjust_params, kwds)

        return slice_transformed

    def pose_adjustment(
        self,
        transformations: Dict[str, Any] = None,
        apply: bool = False,
        reset: bool = True,
        verbose: bool = True,
        **kwds,
    ):
        """Interactive panel to adjust transformations that are applied to the image.
        Applies first a scaling, next a x/y translation, and last a rotation around
        the center of the image (pixel 256/256).

        Args:
            transformations (dict, optional): Dictionary with transformations.
                Defaults to self.transformations or config["momentum"]["transformtions"].
            apply (bool, optional):
                Option to directly apply the provided transformations.
                Defaults to False.
            reset (bool, optional):
                Option to reset the correction before transformation. Defaults to True.
            verbose (bool, optional):
                Option to report the performed transformations. Defaults to True.
            **kwds: Keyword parameters defining defaults for the transformations:

                - **scale** (float): Initial value of the scaling slider.
                - **xtrans** (float): Initial value of the xtrans slider.
                - **ytrans** (float): Initial value of the ytrans slider.
                - **angle** (float): Initial value of the angle slider.
        """
        matplotlib.use("module://ipympl.backend_nbagg")
        if self.slice_corrected is None or not self.slice_corrected.any():
            if self.slice is None or not self.slice.any():
                self.slice = np.zeros(self._config["momentum"]["bins"][0:2])
            source_image = self.slice
            plot = False
        else:
            source_image = self.slice_corrected
            plot = True

        transformed_image = source_image

        if reset:
            if self.rdeform_field_bkp is not None and self.cdeform_field_bkp is not None:
                self.rdeform_field = self.rdeform_field_bkp
                self.cdeform_field = self.cdeform_field_bkp
            else:
                self.reset_deformation()

        center = self._config["momentum"]["center_pixel"]
        if plot:
            fig, ax = plt.subplots(1, 1)
            img = ax.imshow(transformed_image.T, origin="lower", cmap="terrain_r")
            ax.axvline(x=center[0])
            ax.axhline(y=center[1])

        if transformations is None:
            transformations = deepcopy(self.transformations)

        if len(kwds) > 0:
            for key, value in kwds.items():
                transformations[key] = value

        elif "creation_date" in transformations and verbose:
            datestring = datetime.fromtimestamp(transformations["creation_date"]).strftime(
                "%m/%d/%Y, %H:%M:%S",
            )
            print(f"Using transformation parameters generated on {datestring}")

        def update(scale: float, xtrans: float, ytrans: float, angle: float):
            transformed_image = source_image
            if scale != 1:
                transformations["scale"] = scale
                transformed_image = self.coordinate_transform(
                    image=transformed_image,
                    transform_type="scaling",
                    xscale=scale,
                    yscale=scale,
                )
            if xtrans != 0:
                transformations["xtrans"] = xtrans
            if ytrans != 0:
                transformations["ytrans"] = ytrans
            if xtrans != 0 or ytrans != 0:
                transformed_image = self.coordinate_transform(
                    image=transformed_image,
                    transform_type="translation",
                    xtrans=xtrans,
                    ytrans=ytrans,
                )
            if angle != 0:
                transformations["angle"] = angle
                transformed_image = self.coordinate_transform(
                    image=transformed_image,
                    transform_type="rotation",
                    angle=angle,
                    center=center,
                )
            if plot:
                img.set_data(transformed_image.T)
                axmin = np.min(transformed_image, axis=(0, 1))
                axmax = np.max(transformed_image, axis=(0, 1))
                if axmin < axmax:
                    img.set_clim(axmin, axmax)
                fig.canvas.draw_idle()

        update(
            scale=transformations.get("scale", 1),
            xtrans=transformations.get("xtrans", 0),
            ytrans=transformations.get("ytrans", 0),
            angle=transformations.get("angle", 0),
        )

        scale_slider = ipw.FloatSlider(
            value=transformations.get("scale", 1),
            min=0.8,
            max=1.2,
            step=0.01,
        )
        xtrans_slider = ipw.FloatSlider(
            value=transformations.get("xtrans", 0),
            min=-200,
            max=200,
            step=1,
        )
        ytrans_slider = ipw.FloatSlider(
            value=transformations.get("ytrans", 0),
            min=-200,
            max=200,
            step=1,
        )
        angle_slider = ipw.FloatSlider(
            value=transformations.get("angle", 0),
            min=-180,
            max=180,
            step=1,
        )
        results_box = ipw.Output()
        ipw.interact(
            update,
            scale=scale_slider,
            xtrans=xtrans_slider,
            ytrans=ytrans_slider,
            angle=angle_slider,
        )

        def apply_func(apply: bool):  # noqa: ARG001
            if transformations.get("scale", 1) != 1:
                self.coordinate_transform(
                    transform_type="scaling",
                    xscale=transformations["scale"],
                    yscale=transformations["scale"],
                    keep=True,
                )
                if verbose:
                    with results_box:
                        print(f"Applied scaling with scale={transformations['scale']}.")
            if transformations.get("xtrans", 0) != 0 or transformations.get("ytrans", 0) != 0:
                self.coordinate_transform(
                    transform_type="translation",
                    xtrans=transformations.get("xtrans", 0),
                    ytrans=transformations.get("ytrans", 0),
                    keep=True,
                )
                if verbose:
                    with results_box:
                        print(
                            f"Applied translation with (xtrans={transformations.get('xtrans', 0)},",
                            f"ytrans={transformations.get('ytrans', 0)}).",
                        )
            if transformations.get("angle", 0) != 0:
                self.coordinate_transform(
                    transform_type="rotation",
                    angle=transformations["angle"],
                    center=center,
                    keep=True,
                )
                if verbose:
                    with results_box:
                        print(f"Applied rotation with angle={transformations['angle']}.")

                display(results_box)

            if plot:
                img.set_data(self.slice_transformed.T)
                axmin = np.min(self.slice_transformed, axis=(0, 1))
                axmax = np.max(self.slice_transformed, axis=(0, 1))
                if axmin < axmax:
                    img.set_clim(axmin, axmax)
                fig.canvas.draw_idle()

            if transformations != self.transformations:
                transformations["creation_date"] = datetime.now().timestamp()
                self.transformations = transformations

            if verbose:
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

        if plot:
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

    def view(  # pylint: disable=dangerous-default-value
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

        Args:
            image (np.ndarray, optional): The image to plot. Defaults to self.slice.
            origin (str, optional): Figure origin specification ('lower' or 'upper').
                Defaults to "lower".
            cmap (str, optional):  Colormap specification. Defaults to "terrain_r".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (4, 4).
            points (dict, optional): Points for annotation. Defaults to None.
            annotated (bool, optional): Option to add annotation. Defaults to False.
            backend (str, optional): Visualization backend specification. Defaults to
                "matplotlib".

                - 'matplotlib': use static display rendered by matplotlib.
                - 'bokeh': use interactive display rendered by bokeh.

            imkwds (dict, optional): Keyword arguments for
                ``matplotlib.pyplot.imshow()``. Defaults to {}.
            scatterkwds (dict, optional): Keyword arguments for
                ``matplotlib.pyplot.scatter()``. Defaults to {}.
            cross (bool, optional): Option to display a horizontal/vertical lines at
                self.pcent. Defaults to False.
            crosshair (bool, optional): Display option to plot circles around center
                self.pcent. Works only in bokeh backend. Defaults to False.
            crosshair_radii (List[int], optional): Pixel radii of circles to plot when
                crosshair option is activated. Defaults to [50, 100, 150].
            crosshair_thickness (int, optional): Thickness of crosshair circles.
                Defaults to 1.
            **kwds: keyword arguments.
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
                center = self._config["momentum"]["center_pixel"]
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
                width=figsize[0],
                height=figsize[1],
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

    def select_k_range(
        self,
        point_a: Union[np.ndarray, List[int]] = None,
        point_b: Union[np.ndarray, List[int]] = None,
        k_distance: float = None,
        k_coord_a: Union[np.ndarray, List[float]] = None,
        k_coord_b: Union[np.ndarray, List[float]] = np.array([0.0, 0.0]),
        equiscale: bool = True,
        apply: bool = False,
    ):
        """Interactive selection function for features for the Momentum axes calibra-
        tion. It allows the user to select the pixel positions of two symmetry points
        (a and b) and the k-space distance of the two. Alternatively, the corrdinates
        of both points can be provided. See the equiscale option for details on the
        specifications of point coordinates.

        Args:
            point_a (Union[np.ndarray, List[int]], optional): Pixel coordinates of the
                symmetry point a.
            point_b (Union[np.ndarray, List[int]], optional): Pixel coordinates of the
                symmetry point b. Defaults to the center pixel of the image, defined by
                config["momentum"]["center_pixel"].
            k_distance (float, optional): The known momentum space distance between the
                two symmetry points.
            k_coord_a (Union[np.ndarray, List[float]], optional): Momentum coordinate
                of the symmetry points a. Only valid if equiscale=False.
            k_coord_b (Union[np.ndarray, List[float]], optional): Momentum coordinate
                of the symmetry points b. Only valid if equiscale=False. Defaults to
                the k-space center np.array([0.0, 0.0]).
            equiscale (bool, optional): Option to adopt equal scale along both the x
                and y directions.

                - **True**: Use a uniform scale for both x and y directions in the
                  image coordinate system. This applies to the situation where
                  k_distance is given and the points a and b are (close to) parallel
                  with one of the two image axes.
                - **False**: Calculate the momentum scale for both x and y directions
                  separately. This applies to the situation where the points a and b
                  are sufficiently different in both x and y directions in the image
                  coordinate system.

                Defaults to 'True'.

            apply (bool, optional): Option to directly store the calibration parameters
                to the class. Defaults to False.

        Raises:
            ValueError: If no valid image is found from which to ge the coordinates.
        """
        matplotlib.use("module://ipympl.backend_nbagg")
        if self.slice_transformed is not None:
            image = self.slice_transformed
        elif self.slice_corrected is not None:
            image = self.slice_corrected
        elif self.slice is not None:
            image = self.slice
        else:
            raise ValueError("No valid image loaded!")

        if point_b is None:
            point_b = self._config["momentum"]["center_pixel"]

        if point_a is None:
            point_a = [0, 0]

        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(image.T, origin="lower", cmap="terrain_r")

        (marker_a,) = ax.plot(point_a[0], point_a[1], "o")
        (marker_b,) = ax.plot(point_b[0], point_b[1], "ro")

        def update(
            point_a_x: int,
            point_a_y: int,
            point_b_x: int,
            point_b_y: int,
            k_distance: float,  # noqa: ARG001
        ):
            fig.canvas.draw_idle()
            marker_a.set_xdata(point_a_x)
            marker_a.set_ydata(point_a_y)
            marker_b.set_xdata(point_b_x)
            marker_b.set_ydata(point_b_y)

        point_a_input_x = ipw.IntText(point_a[0])
        point_a_input_y = ipw.IntText(point_a[1])
        point_b_input_x = ipw.IntText(point_b[0])
        point_b_input_y = ipw.IntText(point_b[1])
        k_distance_input = ipw.FloatText(k_distance)
        ipw.interact(
            update,
            point_a_x=point_a_input_x,
            point_a_y=point_a_input_y,
            point_b_x=point_b_input_x,
            point_b_y=point_b_input_y,
            k_distance=k_distance_input,
        )

        self._state = 0

        def onclick(event):
            if self._state == 0:
                point_a_input_x.value = event.xdata
                point_a_input_y.value = event.ydata
                self._state = 1
            else:
                point_b_input_x.value = event.xdata
                point_b_input_y.value = event.ydata
                self._state = 0

        cid = fig.canvas.mpl_connect("button_press_event", onclick)

        def apply_func(apply: bool):  # noqa: ARG001
            point_a = [point_a_input_x.value, point_a_input_y.value]
            point_b = [point_b_input_x.value, point_b_input_y.value]
            calibration = self.calibrate(
                point_a=point_a,
                point_b=point_b,
                k_distance=k_distance,
                equiscale=equiscale,
                k_coord_a=k_coord_a,
                k_coord_b=k_coord_b,
            )

            img.set_extent(calibration["extent"])
            plt.title("Momentum calibrated data")
            plt.xlabel("$k_x$", fontsize=15)
            plt.ylabel("$k_y$", fontsize=15)
            ax.axhline(0)
            ax.axvline(0)

            fig.canvas.mpl_disconnect(cid)

            point_a_input_x.close()
            point_a_input_y.close()
            point_b_input_x.close()
            point_b_input_y.close()
            k_distance_input.close()
            apply_button.close()

            fig.canvas.draw_idle()

        apply_button = ipw.Button(description="apply")
        display(apply_button)
        apply_button.on_click(apply_func)

        if apply:
            apply_func(True)

        plt.show()

    def calibrate(
        self,
        point_a: Union[np.ndarray, List[int]],
        point_b: Union[np.ndarray, List[int]],
        k_distance: float = None,
        k_coord_a: Union[np.ndarray, List[float]] = None,
        k_coord_b: Union[np.ndarray, List[float]] = np.array([0.0, 0.0]),
        equiscale: bool = True,
        image: np.ndarray = None,
    ) -> dict:
        """Momentum axes calibration using the pixel positions of two symmetry points
        (a and b) and the absolute coordinate of a single point (b), defaulted to
        [0., 0.]. All coordinates should be specified in the (x/y), i.e. (column_index,
        row_index) format. See the equiscale option for details on the specifications
        of point coordinates.

        Args:
            point_a (Union[np.ndarray, List[int]], optional): Pixel coordinates of the
                symmetry point a.
            point_b (Union[np.ndarray, List[int]], optional): Pixel coordinates of the
                symmetry point b. Defaults to the center pixel of the image, defined by
                config["momentum"]["center_pixel"].
            k_distance (float, optional): The known momentum space distance between the
                two symmetry points.
            k_coord_a (Union[np.ndarray, List[float]], optional): Momentum coordinate
                of the symmetry points a. Only valid if equiscale=False.
            k_coord_b (Union[np.ndarray, List[float]], optional): Momentum coordinate
                of the symmetry points b. Only valid if equiscale=False. Defaults to
                the k-space center np.array([0.0, 0.0]).
            equiscale (bool, optional): Option to adopt equal scale along both the x
                and y directions.

                - **True**: Use a uniform scale for both x and y directions in the
                  image coordinate system. This applies to the situation where
                  k_distance is given and the points a and b are (close to) parallel
                  with one of the two image axes.
                - **False**: Calculate the momentum scale for both x and y directions
                  separately. This applies to the situation where the points a and b
                  are sufficiently different in both x and y directions in the image
                  coordinate system.

                Defaults to 'True'.
            image (np.ndarray, optional): The energy slice for which to return the
                calibration. Defaults to self.slice_corrected.

        Returns:
            dict: dictionary with following entries:

                - "axes": Tuple of 1D arrays
                  Momentum coordinates of the row and column.
                - "extent": list
                  Extent of the two momentum axis (can be used directly in imshow).
                - "grid": Tuple of 2D arrays
                  Row and column mesh grid generated from the coordinates
                  (can be used directly in pcolormesh).
                - "coeffs": Tuple of (x, y) calibration coefficients
                - "x_center", "y_center": Pixel positions of the k-space center
                - "cstart", "rstart": Detector positions of the image used for
                  calibration
                - "cstep", "rstep": Step size of detector coordinates in the image
                  used for calibration
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
        self.calibration["creation_date"] = datetime.now().timestamp()
        self.calibration["kx_axis"] = k_row
        self.calibration["ky_axis"] = k_col
        self.calibration["grid"] = (k_rowgrid, k_colgrid)
        self.calibration["extent"] = (k_row[0], k_row[-1], k_col[0], k_col[-1])
        self.calibration["kx_scale"] = xratio
        self.calibration["ky_scale"] = yratio
        self.calibration["x_center"] = point_b[0] - k_coord_b[0] / xratio
        self.calibration["y_center"] = point_b[1] - k_coord_b[1] / yratio
        # copy parameters for applying calibration
        try:
            self.calibration["rstart"] = self.bin_ranges[0][0]
            self.calibration["cstart"] = self.bin_ranges[1][0]
            self.calibration["rstep"] = (self.bin_ranges[0][1] - self.bin_ranges[0][0]) / nrows
            self.calibration["cstep"] = (self.bin_ranges[1][1] - self.bin_ranges[1][0]) / ncols
        except (AttributeError, IndexError):
            pass

        return self.calibration

    def apply_corrections(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        x_column: str = None,
        y_column: str = None,
        new_x_column: str = None,
        new_y_column: str = None,
        verbose: bool = True,
        **kwds,
    ) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
        """Calculate and replace the X and Y values with their distortion-corrected
        version.

        Args:
            df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to apply
                the distotion correction to.
            x_column (str, optional): Label of the 'X' column before momentum
                distortion correction. Defaults to config["momentum"]["x_column"].
            y_column (str, optional): Label of the 'Y' column before momentum
                distortion correction. Defaults to config["momentum"]["y_column"].
            new_x_column (str, optional): Label of the 'X' column after momentum
                distortion correction.
                Defaults to config["momentum"]["corrected_x_column"].
            new_y_column (str, optional): Label of the 'Y' column after momentum
                distortion correction.
                Defaults to config["momentum"]["corrected_y_column"].
            verbose (bool, optional): Option to report the used landmarks for correction.
                Defaults to True.
            **kwds: Keyword arguments:

                - **dfield**: Inverse dfield
                - **cdeform_field**, **rdeform_field**: Column- and row-wise forward
                  deformation fields.

                Additional keyword arguments are passed to ``apply_dfield``.

        Returns:
            Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]: Dataframe with
            added columns and momentum correction metadata dictionary.
        """
        if x_column is None:
            x_column = self.x_column
        if y_column is None:
            y_column = self.y_column

        if new_x_column is None:
            new_x_column = self.corrected_x_column
        if new_y_column is None:
            new_y_column = self.corrected_y_column

        if self.inverse_dfield is None or self.dfield_updated:
            if self.rdeform_field is None and self.cdeform_field is None:
                if self.correction or self.transformations:
                    if self.correction:
                        # Generate spline warp from class features or config
                        self.spline_warp_estimate(verbose=verbose)
                    if self.transformations:
                        # Apply config pose adjustments
                        self.pose_adjustment()
                else:
                    raise ValueError("No corrections or transformations defined!")

            self.inverse_dfield = generate_inverse_dfield(
                self.rdeform_field,
                self.cdeform_field,
                self.bin_ranges,
                self.detector_ranges,
            )
            self.dfield_updated = False

        out_df = df.map_partitions(
            apply_dfield,
            dfield=self.inverse_dfield,
            x_column=x_column,
            y_column=y_column,
            new_x_column=new_x_column,
            new_y_column=new_y_column,
            detector_ranges=self.detector_ranges,
            **kwds,
        )

        metadata = self.gather_correction_metadata()

        return out_df, metadata

    def gather_correction_metadata(self) -> dict:
        """Collect meta data for momentum correction.

        Returns:
            dict: generated correction metadata dictionary.
        """
        metadata: Dict[Any, Any] = {}
        if len(self.correction) > 0:
            metadata["correction"] = self.correction
            metadata["correction"]["applied"] = True
            metadata["correction"]["cdeform_field"] = self.cdeform_field
            metadata["correction"]["rdeform_field"] = self.rdeform_field
            try:
                metadata["correction"]["creation_date"] = self.correction["creation_date"]
            except KeyError:
                pass
        if len(self.adjust_params) > 0:
            metadata["registration"] = self.adjust_params
            metadata["registration"]["creation_date"] = datetime.now().timestamp()
            metadata["registration"]["applied"] = True
            metadata["registration"]["depends_on"] = (
                "/entry/process/registration/tranformations/rot_z"
                if "angle" in metadata["registration"] and metadata["registration"]["angle"]
                else "/entry/process/registration/tranformations/trans_y"
                if "xtrans" in metadata["registration"] and metadata["registration"]["xtrans"]
                else "/entry/process/registration/tranformations/trans_x"
                if "ytrans" in metadata["registration"] and metadata["registration"]["ytrans"]
                else "."
            )
            if (
                "ytrans" in metadata["registration"] and metadata["registration"]["ytrans"]
            ):  # swapped definitions
                metadata["registration"]["trans_x"] = {}
                metadata["registration"]["trans_x"]["value"] = metadata["registration"]["ytrans"]
                metadata["registration"]["trans_x"]["type"] = "translation"
                metadata["registration"]["trans_x"]["units"] = "pixel"
                metadata["registration"]["trans_x"]["vector"] = np.asarray(
                    [1.0, 0.0, 0.0],
                )
                metadata["registration"]["trans_x"]["depends_on"] = "."
            if "xtrans" in metadata["registration"] and metadata["registration"]["xtrans"]:
                metadata["registration"]["trans_y"] = {}
                metadata["registration"]["trans_y"]["value"] = metadata["registration"]["xtrans"]
                metadata["registration"]["trans_y"]["type"] = "translation"
                metadata["registration"]["trans_y"]["units"] = "pixel"
                metadata["registration"]["trans_y"]["vector"] = np.asarray(
                    [0.0, 1.0, 0.0],
                )
                metadata["registration"]["trans_y"]["depends_on"] = (
                    "/entry/process/registration/tranformations/trans_x"
                    if "ytrans" in metadata["registration"] and metadata["registration"]["ytrans"]
                    else "."
                )
            if "angle" in metadata["registration"] and metadata["registration"]["angle"]:
                metadata["registration"]["rot_z"] = {}
                metadata["registration"]["rot_z"]["value"] = metadata["registration"]["angle"]
                metadata["registration"]["rot_z"]["type"] = "rotation"
                metadata["registration"]["rot_z"]["units"] = "degrees"
                metadata["registration"]["rot_z"]["vector"] = np.asarray(
                    [0.0, 0.0, 1.0],
                )
                metadata["registration"]["rot_z"]["offset"] = np.concatenate(
                    (metadata["registration"]["center"], [0.0]),
                )
                metadata["registration"]["rot_z"]["depends_on"] = (
                    "/entry/process/registration/tranformations/trans_y"
                    if "xtrans" in metadata["registration"] and metadata["registration"]["xtrans"]
                    else "/entry/process/registration/tranformations/trans_x"
                    if "ytrans" in metadata["registration"] and metadata["registration"]["ytrans"]
                    else "."
                )

        return metadata

    def append_k_axis(
        self,
        df: Union[pd.DataFrame, dask.dataframe.DataFrame],
        x_column: str = None,
        y_column: str = None,
        new_x_column: str = None,
        new_y_column: str = None,
        calibration: dict = None,
        **kwds,
    ) -> Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]:
        """Calculate and append the k axis coordinates (kx, ky) to the events dataframe.

        Args:
            df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to apply the
                distotion correction to.
            x_column (str, optional): Label of the source 'X' column.
                Defaults to config["momentum"]["corrected_x_column"] or
                config["momentum"]["x_column"] (whichever is present).
            y_column (str, optional): Label of the source 'Y' column.
                Defaults to config["momentum"]["corrected_y_column"] or
                config["momentum"]["y_column"] (whichever is present).
            new_x_column (str, optional): Label of the destination 'X' column after
                momentum calibration. Defaults to config["momentum"]["kx_column"].
            new_y_column (str, optional): Label of the destination 'Y' column after
                momentum calibration. Defaults to config["momentum"]["ky_column"].
            calibration (dict, optional): Dictionary containing calibration parameters.
                Defaults to 'self.calibration' or config["momentum"]["calibration"].
            **kwds: Keyword parameters for momentum calibration. Parameters are added
                to the calibration dictionary.

        Returns:
            Tuple[Union[pd.DataFrame, dask.dataframe.DataFrame], dict]: Dataframe with
            added columns and momentum calibration metadata dictionary.
        """
        if x_column is None:
            if self.corrected_x_column in df.columns:
                x_column = self.corrected_x_column
            else:
                x_column = self.x_column
        if y_column is None:
            if self.corrected_y_column in df.columns:
                y_column = self.corrected_y_column
            else:
                y_column = self.y_column

        if new_x_column is None:
            new_x_column = self.kx_column

        if new_y_column is None:
            new_y_column = self.ky_column

        # pylint: disable=duplicate-code
        if calibration is None:
            calibration = deepcopy(self.calibration)

        if len(kwds) > 0:
            for key, value in kwds.items():
                calibration[key] = value
            calibration["creation_date"] = datetime.now().timestamp()

        try:
            (df[new_x_column], df[new_y_column]) = detector_coordiantes_2_k_koordinates(
                r_det=df[x_column],
                c_det=df[y_column],
                r_start=calibration["rstart"],
                c_start=calibration["cstart"],
                r_center=calibration["x_center"],
                c_center=calibration["y_center"],
                r_conversion=calibration["kx_scale"],
                c_conversion=calibration["ky_scale"],
                r_step=calibration["rstep"],
                c_step=calibration["cstep"],
            )
        except KeyError as exc:
            raise ValueError(
                "Required calibration parameters missing!",
            ) from exc

        metadata = self.gather_calibration_metadata(calibration=calibration)

        return df, metadata

    def gather_calibration_metadata(self, calibration: dict = None) -> dict:
        """Collect meta data for momentum calibration

        Args:
            calibration (dict, optional): Dictionary with momentum calibration
                parameters. If omitted will be taken from the class.

        Returns:
            dict: Generated metadata dictionary.
        """
        if calibration is None:
            calibration = self.calibration
        metadata: Dict[Any, Any] = {}
        try:
            metadata["creation_date"] = calibration["creation_date"]
        except KeyError:
            pass
        metadata["applied"] = True
        metadata["calibration"] = calibration
        # create empty calibrated axis entries, if they are not present.
        if "kx_axis" not in metadata["calibration"]:
            metadata["calibration"]["kx_axis"] = 0
        if "ky_axis" not in metadata["calibration"]:
            metadata["calibration"]["ky_axis"] = 0

        return metadata


def cm2palette(cmap_name: str) -> list:
    """Convert certain matplotlib colormap (cm) to bokeh palette.

    Args:
        cmap_name (str): Name of the colormap/palette.

    Returns:
        list: List of colors in hex representation (a bokoeh palette).
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
    """Merge a dictionary with other dictionaries.

    Args:
        main_dict (dict): Main dictionary.
        other_entries (Union[List[dict], Tuple[dict], dict]):
            Other dictionary or composite dictionarized elements.

    Returns:
        dict: Merged dictionary.
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


def detector_coordiantes_2_k_koordinates(
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
    """Conversion from detector coordinates (rdet, cdet) to momentum coordinates
    (kr, kc).

    Args:
        r_det (float): Row detector coordinates.
        c_det (float): Column detector coordinates.
        r_start (float): Start value for row detector coordinates.
        c_start (float): Start value for column detector coordinates.
        r_center (float): Center value for row detector coordinates.
        c_center (float): Center value for column detector coordinates.
        r_conversion (float): Row conversion factor.
        c_conversion (float): Column conversion factor.
        r_step (float): Row stepping factor.
        c_step (float): Column stepping factor.

    Returns:
        Tuple[float, float]: Converted momentum space row/column coordinates.
    """
    r_det0 = r_start + r_step * r_center
    c_det0 = c_start + c_step * c_center
    k_r = r_conversion * ((r_det - r_det0) / r_step)
    k_c = c_conversion * ((c_det - c_det0) / c_step)

    return (k_r, k_c)


def apply_dfield(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    dfield: np.ndarray,
    x_column: str,
    y_column: str,
    new_x_column: str,
    new_y_column: str,
    detector_ranges: List[Tuple],
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Application of the inverse displacement-field to the dataframe coordinates.

    Args:
        df (Union[pd.DataFrame, dask.dataframe.DataFrame]): Dataframe to apply the
            distotion correction to.
        dfield (np.ndarray): The distortion correction field. 3D matrix,
            with column and row distortion fields stacked along the first dimension.
        x_column (str): Label of the 'X' source column.
        y_column (str): Label of the 'Y' source column.
        new_x_column (str): Label of the 'X' destination column.
        new_y_column (str): Label of the 'Y' destination column.
        detector_ranges (List[Tuple]): tuple of pixel ranges of the detector x/y
            coordinates

    Returns:
        Union[pd.DataFrame, dask.dataframe.DataFrame]: dataframe with added columns
    """
    x = df[x_column]
    y = df[y_column]

    r_axis_steps = (detector_ranges[0][1] - detector_ranges[0][0]) / dfield[0].shape[0]
    c_axis_steps = (detector_ranges[1][1] - detector_ranges[1][0]) / dfield[0].shape[1]

    df[new_x_column], df[new_y_column] = (
        map_coordinates(dfield[0], (x, y), order=1) * r_axis_steps,
        map_coordinates(dfield[1], (x, y), order=1) * c_axis_steps,
    )
    return df


def generate_inverse_dfield(
    rdeform_field: np.ndarray,
    cdeform_field: np.ndarray,
    bin_ranges: List[Tuple],
    detector_ranges: List[Tuple],
) -> np.ndarray:
    """Generate inverse deformation field using inperpolation with griddata.
    Assuming the binning range of the input ``rdeform_field`` and ``cdeform_field``
    covers the whole detector.

    Args:
        rdeform_field (np.ndarray): Row-wise deformation field.
        cdeform_field (np.ndarray): Column-wise deformation field.
        bin_ranges (List[Tuple]): Detector ranges of the binned coordinates.
        detector_ranges (List[Tuple]): Ranges of detector coordinates to interpolate to.

    Returns:
        np.ndarray: The calculated inverse deformation field (row/column)
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

    ret = Parallel(n_jobs=2)(
        delayed(griddata)(np.asarray(rc_position), np.asarray(arg), (r_mesh, c_mesh))
        for arg in [r_dest, c_dest]
    )

    inverse_dfield = np.asarray([ret[0], ret[1]])

    return inverse_dfield


def load_dfield(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load inverse dfield from file

    Args:
        file (str): Path to file containing the inverse dfield

    Returns:
        np.ndarray: the loaded inverse deformation field
    """
    rdeform_field: np.ndarray = None
    cdeform_field: np.ndarray = None

    try:
        dfield = np.load(file)
        rdeform_field = dfield[0]
        cdeform_field = dfield[1]

    except FileNotFoundError:
        pass

    return rdeform_field, cdeform_field
