# Note: some of the functions presented here were
# inspired by https://github.com/mpes-kit/mpes
from typing import Tuple
from typing import Union

import dask.dataframe
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from symmetrize import sym, tps, pointops as po
import scipy.ndimage as ndi
from functools import partial
from numpy.linalg import norm
import matplotlib.pyplot as plt
import bokeh.plotting as pbk
from bokeh.io import output_notebook
from bokeh.palettes import Category10 as ColorCycle
import bokeh.palettes as bp
from bokeh.colors import RGB
import itertools as it
from funcy import project
import cv2


class MomentumCorrector():
    """
    Momentum distortion correction and momentum calibration workflow.
    """

    def __init__(self, image, rotsym=6, config:dict={}):
        """
        **Parameters**\n
        image: 3d array
            Volumetric band structure data.
        rotsym: int | 6
            Order of rotational symmetry.
        """

        self.image = np.squeeze(image)
        self.imgndim = image.ndim
        if (self.imgndim > 3) or (self.imgndim < 2):
            raise ValueError('The input image dimension need to be 2 or 3!')
        if (self.imgndim == 2):
            self.slice = self.image

        self._config = config

        self.rotsym = int(rotsym)
        self.rotsym_angle = int(360 / self.rotsym)
        self.arot = np.array([0] + [self.rotsym_angle]*(self.rotsym-1))
        self.ascale = np.array([1.0]*self.rotsym)
        self.adjust_params = {}
        self.binranges = []
        self.binsteps = []

    @property
    def features(self):
        """ Dictionary of detected features for the symmetrization process.
        ``self.features`` is a derived attribute from existing ones.
        """

        feature_dict = {'verts':np.asarray(self.__dict__.get('pouter_ord', [])),
                        'center':np.asarray(self.__dict__.get('pcent', []))}

        return feature_dict

    @property
    def symscores(self):
        """ Dictionary of symmetry-related scores.
        """

        sym_dict = {'csm_original':self.__dict__.get('csm_original', ''),
                    'csm_current':self.__dict__.get('csm_current', ''),
                    'arm_original':self.__dict__.get('arm_original', ''),
                    'arm_current':self.__dict__.get('arm_current', '')}

        return sym_dict

    def selectSlice2D(self, selector, axis=2):
        """ Select (hyper)slice from a (hyper)volume.

        **Parameters**\n
        selector: slice object/list/int
            Selector along the specified axis to extract the slice (image).
            Use the construct slice(start, stop, step) to select a range of images and sum them.
            Use an integer to specify only a particular slice.
        axis: int | 2
            Axis along which to select the image.
        """

        if self.imgndim > 2:
            im = np.moveaxis(self.image, axis, 0)
            try:
                self.slice = im[selector,...].sum(axis=0)
            except:
                self.slice = im[selector,...]

        elif self.imgndim == 2:
            raise ValueError('Input image dimension is already 2!')

            
    def importBinningParameters(self, parp):
        """ Import parameters of binning used for correction image from parallelHDF5Processor Class instance
        
        **Parameters**\n
        parp: instance of the ``ParallelHDF5Processor`` class
            Import parameters used for creation of the distortion-corrected image.            
        """
        
        if hasattr(parp, '__class__'):
            self.binranges = parp.binranges
            self.binsteps = parp.binsteps
        else:
            raise ValueError('Not a valid parallelHDF5Processor class instance!')
            

    def featureExtract(self, image, direction='ccw', type='points', center_det='centroidnn',
                        symscores=True, **kwds):
        """ Extract features from the selected 2D slice.
            Currently only point feature detection is implemented.

        **Parameters**\n
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

        self.resetDeformation(image=image, coordtype='cartesian')

        if type == 'points':

            self.center_detection_method = center_det
            symtype = kwds.pop('symtype', 'rotation')

            # Detect the point landmarks
            self.peaks = po.peakdetect2d(image, **kwds)
            if center_det is None:
                if self.peaks.shape[0] != self.rotsym:
                    print(f"Found '{self.peaks.shape[0]}' points, but '{self.rotsym}' required.")
                self.pouter = self.peaks
                self.pcent = None
            else:
                if self.peaks.shape[0] != self.rotsym+1:
                    print(f"Found '{self.peaks.shape[0]}' points, but '{self.rotsym+1}' required")
                self.pcent, self.pouter = po.pointset_center(self.peaks, method=center_det, ret='cnc')
                self.pcent = tuple(self.pcent)
            # Order the point landmarks
            self.pouter_ord = po.pointset_order(self.pouter, direction=direction)
            try:
                self.area_old = po.polyarea(coords=self.pouter_ord, coord_order='rc')
            except:
                pass

            # Calculate geometric distances
            self.calcGeometricDistances()

            if symscores == True:
                self.csm_original = self.calcSymmetryScores(symtype=symtype)

            if self.rotsym == 6:
                self.mdist = (self.mcvdist + self.mvvdist) / 2
                self.mcvdist = self.mdist
                self.mvvdist = self.mdist

        else:
            raise NotImplementedError


    def linWarpEstimate(self, weights=(1, 1, 1), optfunc='minimize', optmethod='Nelder-Mead',
                        ret=True, warpkwds={}, **kwds):
        """ Estimate the homography-based deformation field using landmark correspondences.

        **Parameters**\n
        weights: tuple/list/array | (1, 1, 1)
            Weights added to the terms in the optimizer. The terms are assigned
            to the cost functions of (1) centeredness, (2) center-vertex symmetry,
            (3) vertex-vertex symmetry, respectively.
        optfunc, optmethod: str/func, str | 'minimize', 'Nelder-Mead'
            Name of the optimizer function and the optimization method.
            See description in ``mpes.analysis.sym.target_set_optimize()``.
        ret: bool | True
            Specify if returning the corrected image slice.
        warpkwds: dictionary | {}
            Additional arguments passed to ``symmetrize.sym.imgWarping()``.
        **kwds: keyword arguments
            ========= ========== =============================================
            keyword   data type  meaning
            ========= ========== =============================================
            niter     int        Maximum number of iterations
            landmarks list/array Symmetry landmarks selected for registration
            fitinit   tuple/list Initial conditions for fitting
            ========= ========== =============================================

        **Return**\n
            Corrected 2D image slice (when ``ret=True`` is specified in the arguments).
        """

        landmarks = kwds.pop('landmarks', self.pouter_ord)
        # Set up the initial condition for the optimization for symmetrization
        fitinit = np.asarray([self.arot, self.ascale]).ravel()
        self.init = kwds.pop('fitinit', fitinit)

        self.ptargs, _ = sym.target_set_optimize(self.init, landmarks, self.pcent, self.mcvdist,
                        self.mvvdist, direction=1, weights=weights, optfunc=optfunc,
                        optmethod=optmethod, **kwds)

        # Calculate warped image and landmark positions
        self.slice_corrected, self.linwarp = sym.imgWarping(self.slice, landmarks=landmarks,
                        targs=self.ptargs, **warpkwds)

        if ret:
            return self.slice_corrected

    def calcGeometricDistances(self):
        """ Calculate geometric distances involving the center and the vertices.
        Distances calculated include center-vertex and nearest-neighbor vertex-vertex distances.
        """

        self.cvdist = po.cvdist(self.pouter_ord, self.pcent)
        self.mcvdist = self.cvdist.mean()
        self.vvdist = po.vvdist(self.pouter_ord)
        self.mvvdist = self.vvdist.mean()

    def calcSymmetryScores(self, symtype='rotation'):
        """ Calculate the symmetry scores from geometric quantities.

        **Paramters**\n
        symtype: str | 'rotation'
            Type of symmetry.
        """

        csm = po.csm(self.pcent, self.pouter_ord, rotsym=self.rotsym, type=symtype)

        return csm

    @staticmethod
    def transform(points, transmat):
        """ Coordinate transform of a point set in the (row, column) formulation.

        **Parameters**\n
        points: list/array
            Cartesian pixel coordinates of the points to be transformed.
        transmat: 2D array
            The transform matrix.

        **Return**\n
            Transformed point coordinates.
        """

        pts_cart_trans = sym.pointsetTransform(np.roll(points, shift=1, axis=1), transmat)

        return np.roll(pts_cart_trans, shift=1, axis=1)

    def splineWarpEstimate(self, image, include_center=True, fixed_center=True, iterative=False,
                            interp_order=1, update=False, ret=False, **kwds):
        """ Estimate the spline deformation field using thin plate spline registration.

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

        self.prefs = kwds.pop('landmarks', self.pouter_ord)
        self.ptargs = kwds.pop('targets', [])

        # Generate the target point set
        if not self.ptargs:
            self.ptargs = sym.rotVertexGenerator(self.pcent, fixedvertex=self.pouter_ord[0,:], arot=self.arot,
                        direction=-1, scale=self.ascale, ret='all')[1:,:]

        if include_center == True:
            # Include center of image pattern in the registration-based symmetrization
            if fixed_center == True: # Add the same center to both the reference and target sets

                self.prefs = np.column_stack((self.prefs.T, self.pcent)).T
                self.ptargs = np.column_stack((self.ptargs.T, self.pcent)).T

            else: # Add different centers to the reference and target sets
                newcenters = kwds.pop('new_centers', {})
                self.prefs = np.column_stack((self.prefs.T, newcenters['lmkcenter'])).T
                self.ptargs = np.column_stack((self.ptargs.T, newcenters['targcenter'])).T

        if iterative == False: # Non-iterative estimation of deformation field
            self.slice_transformed, self.splinewarp = tps.tpsWarping(self.prefs, self.ptargs,
                            image, None, interp_order, ret='all', **kwds)

        else: # Iterative estimation of deformation field
            # ptsw, H, rst = sym.target_set_optimize(init, lm, tuple(cen), mcd0, mcd0, ima[None,:,:],
                        # niter=30, direction=-1, weights=(1, 1, 1), ftol=1e-8)
            pass

        # Update the deformation field
        coordmat = sym.coordinate_matrix_2D(image, coordtype='cartesian', stackaxis=0).astype('float64')
        self.updateDeformation(self.splinewarp[0], self.splinewarp[1], reset=True, image=image, coordtype='cartesian')

        if update == True:
            self.slice_corrected = self.slice_transformed.copy()

        if ret:
            return self.slice_transformed

    def rotate(self, angle='auto', ret=False, **kwds):
        """ Rotate 2D image in the homogeneous coordinate.

        **Parameters**\n
        angle: float/str
            Angle of rotation (specify 'auto' to use automated estimation).
        ret: bool | False
            Return specification (True/False)
        **kwds: keyword arguments
            ======= ========== =======================================
            keyword data type  meaning
            ======= ========== =======================================
            image   2d array   2D image for correction
            center  tuple/list pixel coordinates of the image center
            scale   float      scaling factor in rotation
            ======= ========== =======================================
            See ``symmetrize.sym.sym_pose_estimate()`` for other keywords.
        """

        image = kwds.pop('image', self.slice)
        center = kwds.pop('center', self.pcent)
        scale = kwds.pop('scale', 1)

        # Automatic determination of the best pose based on grid search within an angular range
        if angle == 'auto':
            center = tuple(np.asarray(center).astype('int'))
            angle_auto, _ = sym.sym_pose_estimate(image/image.max(), center, **kwds)
            self.image_rot, rotmat = _rotate2d(image, center, angle_auto, scale)
        # Rotate image by the specified angle
        else:
            self.image_rot, rotmat = _rotate2d(image, center, angle, scale)

        # Compose the rotation matrix with the previously determined warping matrix
        self.composite_linwarp = np.dot(rotmat, self.linwarp)

        if ret:
            return rotmat

    def correct(self, axis, use_composite_transform=False, update=False, use_deform_field=False,
                updatekwds={}, **kwds):
        """ Apply a 2D transform to a stack of 2D images (3D) along a specific axis.

        **Parameters**\n
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
            image   2d array   3D image for correction
            dfield  list/tuple row and column deformation field
            warping 2d array   2D transform correction matrix
            ======= ========== =================================
        """

        image = kwds.pop('image', self.image)

        if use_deform_field == True:
            dfield = kwds.pop('dfield', [self.rdeform_field, self.cdeform_field])
            self.image_corrected = sym.applyWarping(image, axis, warptype='deform_field', dfield=dfield)

        else:
            if use_composite_transform == True:
                hgmat = kwds.pop('warping', self.composite_linwarp)
            else:
                hgmat = kwds.pop('warping', self.linwarp)
            self.image_corrected = sym.applyWarping(image, axis, warptype='matrix', hgmat=hgmat)

        # Update image features using corrected image
        if update != False:
            if update == True:
                self.update('all', **updatekwds)
            else:
                self.update(update, **updatekwds)

    def applyDeformation(self, image, ret=True, **kwds):
        """ Apply the deformation field to a specified image slice.

        **Parameters**\n
        image: 2D array
            Image slice to apply the deformation.
        ret: bool | True
            Option to return the image after deformation.
        **kwds: keyword arguments
            :rdeform, cdeform: 2D array, 2D array | self.rdeform_field, self.cdeform_field
                Row- and column-ordered deformation fields.
            :interp_order: int | 1
                Interpolation order.
            :others:
                See ``scipy.ndimage.map_coordinates()``.
        """

        rdeform = kwds.pop('rdeform', self.rdeform_field)
        cdeform = kwds.pop('cdeform', self.cdeform_field)
        order = kwds.pop('interp_order', 1)
        imdeformed = ndi.map_coordinates(image, [rdeform, cdeform], order=order, **kwds)

        if ret == True:
            return imdeformed

    def resetDeformation(self, **kwds):
        """ Reset the deformation field.
        """

        image = kwds.pop('image', self.slice)
        coordtype = kwds.pop('coordtype', 'cartesian')
        coordmat = sym.coordinate_matrix_2D(image, coordtype=coordtype, stackaxis=0).astype('float64')

        self.rdeform_field = coordmat[1,...]
        self.cdeform_field = coordmat[0,...]

    def updateDeformation(self, rdeform, cdeform, reset=False, **kwds):
        """ Update the deformation field.

        **Parameters**\n
        rdeform, cdeform: 2D array, 2D array
            Row- and column-ordered deformation fields.
        reset: bool | False
            Option to reset the deformation field.
        **kwds: keyword arguments
            See ``mpes.analysis.MomentumCorrector.resetDeformation()``.
        """

        if reset == True:
            self.resetDeformation(**kwds)

        self.rdeform_field = ndi.map_coordinates(self.rdeform_field, [rdeform, cdeform], order=1)
        self.cdeform_field = ndi.map_coordinates(self.cdeform_field, [rdeform, cdeform], order=1)

    def coordinateTransform(self, type, keep=False, ret=False, interp_order=1,
                            mapkwds={}, **kwds):
        """ Apply a pixel-wise coordinate transform to an image.

        **Parameters**\n
        type: str
            Type of deformation to apply to image slice.
        keep: bool | False
            Option to keep the specified coordinate transform.
        ret: bool | False
            Option to return transformed image slice.
        interp_order: int | 1
            Interpolation order for filling in missed pixels.
        mapkwds: dict | {}
            Additional arguments passed to ``scipy.ndimage.map_coordinates()``.
        **kwds: keyword arguments
            Additional arguments in specific deformation field. See ``symmetrize.sym`` module.
        """

        image = kwds.pop('image', self.slice)
        stackax = kwds.pop('stackaxis', 0)
        coordmat = sym.coordinate_matrix_2D(image, coordtype='homogeneous', stackaxis=stackax)

        if type == 'translation':
            rdisp, cdisp = sym.translationDF(coordmat, stackaxis=stackax, ret='displacement', **kwds)
        elif type == 'rotation':
            rdisp, cdisp = sym.rotationDF(coordmat, stackaxis=stackax, ret='displacement', **kwds)
        elif type == 'rotation_auto':
            center = kwds.pop('center', (0, 0))
            # Estimate the optimal rotation angle using intensity symmetry
            angle_auto, _ = sym.sym_pose_estimate(image/image.max(), center=center, **kwds)
            self.adjust_params = dictmerge(self.adjust_params, {'center': center, 'angle': angle_auto})
            rdisp, cdisp = sym.rotationDF(coordmat, stackaxis=stackax, ret='displacement', angle=angle_auto)
        elif type == 'scaling':
            rdisp, cdisp = sym.scalingDF(coordmat, stackaxis=stackax, ret='displacement', **kwds)
        elif type == 'scaling_auto': # Compare scaling to a reference image
            pass
        elif type == 'shearing':
            rdisp, cdisp = sym.shearingDF(coordmat, stackaxis=stackax, ret='displacement', **kwds)
        elif type == 'homography':
            transform = kwds.pop('transform', np.eye(3))
            rdisp, cdisp = sym.compose_deform_field(coordmat, mat_transform=transform,
                                stackaxis=stackax, ret='displacement', **kwds)
        self.adjust_params = dictmerge(self.adjust_params, kwds)

        # Compute deformation field
        if stackax == 0:
            rdeform, cdeform = coordmat[1,...] + rdisp, coordmat[0,...] + cdisp
        elif stackax == -1:
            rdeform, cdeform = coordmat[...,1] + rdisp, coordmat[...,0] + cdisp

        # Resample image in the deformation field
        if (image is self.slice): # resample using all previous displacement fields
            total_rdeform = ndi.map_coordinates(self.rdeform_field, [rdeform, cdeform], order=1)
            total_cdeform = ndi.map_coordinates(self.cdeform_field, [rdeform, cdeform], order=1)
            self.slice_transformed = ndi.map_coordinates(image, [total_rdeform, total_cdeform],
                                    order=interp_order, **mapkwds)
        else: # if external image is provided, apply only the new addional tranformation
            self.slice_transformed = ndi.map_coordinates(image, [rdeform, cdeform],order=interp_order, **mapkwds)
            
        # Combine deformation fields
        if keep == True:
            self.updateDeformation(rdeform, cdeform, reset=False, image=image, coordtype='cartesian')

        if ret == True:
            return self.slice_transformed

    def view(self, origin='lower', cmap='terrain_r', figsize=(4, 4), points={}, annotated=False,
            display=True, backend='matplotlib', ret=False, imkwds={}, scatterkwds={}, crosshair=False, radii=[50,100,150], crosshair_thickness=1, **kwds):
        """ Display image slice with specified annotations.

        **Parameters**\n
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
        display: bool | True
            Display option when using ``bokeh`` to render interactively.
        backend: str | 'matplotlib'
            Visualization backend specification.
            :'matplotlib': use static display rendered by matplotlib.
            :'bokeh': use interactive display rendered by bokeh.
        ret: bool | False
            Option to return figure and axis objects.
        imkwd: dict | {}
            Keyword arguments for ``matplotlib.pyplot.imshow()``.
        crosshair: bool | False
            Display option to plot circles around center self.pcent. Works only in bokeh backend.
        radii: list | [50,100,150]
            Radii of circles to plot when crosshair optin is activated.
        crosshair_thickness: int | 1
            Thickness of crosshair circles.
        **kwds: keyword arguments
            General extra arguments for the plotting procedure.
        """

        image = kwds.pop('image', self.slice)
        nr, nc = image.shape
        xrg = kwds.pop('xaxis', (0, nc))
        yrg = kwds.pop('yaxis', (0, nr))

        if annotated:
            tsr, tsc = kwds.pop('textshift', (3, 3))
            txtsize = kwds.pop('textsize', 12)

        if backend == 'matplotlib':
            f, ax = plt.subplots(figsize=figsize)
            ax.imshow(image, origin=origin, cmap=cmap, **imkwds)

            # Add annotation to the figure
            if annotated:
                for pk, pvs in points.items():

                    try:
                        ax.scatter(pvs[:,1], pvs[:,0], **scatterkwds)
                    except:
                        ax.scatter(pvs[1], pvs[0], **scatterkwds)

                    if pvs.size > 2:
                        for ipv, pv in enumerate(pvs):
                            ax.text(pv[1]+tsc, pv[0]+tsr, str(ipv), fontsize=txtsize)

        elif backend == 'bokeh':

            output_notebook(hide_banner=True)
            colors = it.cycle(ColorCycle[10])
            ttp = [('(x, y)', '($x, $y)')]
            figsize = kwds.pop('figsize', (320, 300))
            palette = cm2palette(cmap) # Retrieve palette colors
            f = pbk.figure(plot_width=figsize[0], plot_height=figsize[1],
                            tooltips=ttp, x_range=(0, nc), y_range=(0, nr))
            f.image(image=[image], x=0, y=0, dw=nc, dh=nr, palette=palette, **imkwds)

            if annotated == True:
                for pk, pvs in points.items():

                    try:
                        xcirc, ycirc = pvs[:,1], pvs[:,0]
                        f.scatter(xcirc, ycirc, size=8, color=next(colors), **scatterkwds)
                    except:
                        xcirc, ycirc = pvs[1], pvs[0]
                        f.scatter(xcirc, ycirc, size=8, color=next(colors), **scatterkwds)

            if crosshair:
              for radius in radii:
                f.annulus(x=[self.pcent[1]], y=[self.pcent[0]], inner_radius=radius-crosshair_thickness, outer_radius=radius, color="red", alpha=0.6)
                                
            if display:
                pbk.show(f)

        if ret:
            try:
                return f, ax
            except:
                return f

    def calibrate(self, image, point_from, point_to, dist, ret='coeffs', **kwds):
        """ Calibration of the momentum axes. Obtain all calibration-related values,
        return only the ones requested.

        **Parameters**\n
        image: 2d array
            Image slice to construct the calibration function.
        point_from, point_to: list/tuple, list/tuple
            Pixel coordinates of the two special points in (row, col) ordering.
        dist: float
            Distance between the two selected points in inverse Angstrom.
        ret: str | 'coeffs'
            Specification of return values ('axes', 'extent', 'coeffs', 'grid', 'func', 'all').
        **kwds: keyword arguments
            See arguments in ``mpes.analysis.calibrateE()``.

        **Return**\n
            Specified calibration parameters in a dictionary.
        """

        self.calibration = calibrateK(image, point_from, point_to, dist, ret='all', **kwds)
        
        # Store coordinates of BZ center
        self.BZcenter = point_to

        if ret != False:
            try:
                return project(self.calibration, [ret])
            except:
                return project(self.calibration, ret)


def cm2palette(cmapName):
    """ Convert certain matplotlib colormap (cm) to bokeh palette.

    **Parameter**\n
    cmapName: str
        Name of the colormap/palette.

    **Return**\n
    palette: list
        List of colors in hex representation (a bokoeh palette).
    """

    if cmapName in bp.all_palettes.keys():
        palette = eval('bp.' + cmapName)

    else:
        mpl_cm_rgb = (255 * eval('cm.' + cmapName)(range(256))).astype('int')
        palette = [RGB(*tuple(rgb)).to_hex() for rgb in mpl_cm_rgb]

    return palette


def _rotate2d(image, center, angle, scale=1):
    """
    2D matrix scaled rotation carried out in the homogenous coordinate.

    **Parameters**\n
    image: 2d array
        Image matrix.
    center: tuple/list
        Center of the image (row pixel, column pixel).
    angle: numeric
        Angle of image rotation.
    scale: numeric | 1
        Scale of image rotation.

    **Returns**\n
    image_rot: 2d array
        Rotated image matrix.
    rotmat: 2d array
        Rotation matrix in the homogeneous coordinate system.
    """

    rotmat = cv2.getRotationMatrix2D(center, angle=angle, scale=scale)
    # Construct rotation matrix in homogeneous coordinate
    rotmat = np.concatenate((rotmat, np.array([0, 0, 1], ndmin=2)), axis=0)

    image_rot = cv2.warpPerspective(image, rotmat, image.shape)

    return image_rot, rotmat


def dictmerge(D, others):
    """
    Merge a dictionary with other dictionaries.

    **Parameters**\n
    D: dict
        Main dictionary.
    others: list/tuple/dict
        Other dictionary or composite dictionarized elements.

    **Return**\n
    D: dict
        Merged dictionary.
    """

    if type(others) in (list, tuple): # Merge D with a list or tuple of dictionaries
        for oth in others:
            D = {**D, **oth}

    elif type(others) == dict: # Merge D with a single dictionary
        D = {**D, **others}

    return D
    

def calibrateK(img, pxla, pxlb, k_ab=None, kcoorda=None, kcoordb=[0., 0.], equiscale=False, ret=['axes']):
    """
    Momentum axes calibration using the pixel positions of two symmetry points (a and b)
    and the absolute coordinate of a single point (b), defaulted to [0., 0.]. All coordinates
    should be specified in the (row_index, column_index) format. See the equiscale option for
    details on the specifications of point coordinates.

    **Parameters**\n
    img: 2D array
        An energy cut of the band structure.
    pxla, pxlb: list/tuple/1D array
        Pixel coordinates of the two symmetry points (a and b). Point b has the
        default coordinates [0., 0.] (see below).
    k_ab: float | None
        The known momentum space distance between the two symmetry points.
    kcoorda: list/tuple/1D array | None
        Momentum coordinates of the symmetry point a.
    kcoordb: list/tuple/1D array | [0., 0.]
        Momentum coordinates of the symmetry point b (krow, kcol), default to k-space center.
    equiscale: bool | False
        Option to adopt equal scale along both the row and column directions.
        :True: Use a uniform scale for both x and y directions in the image coordinate system.
        This applies to the situation where the points a and b are (close to) parallel with one
        of the two image axes.
        :False: Calculate the momentum scale for both x and y directions separately. This applies
        to the situation where the points a and b are sufficiently different in both x and y directions
        in the image coordinate system.
    ret: list | ['axes']
        Return type specification, options include 'axes', 'extent', 'coeffs', 'grid', 'func', 'all'.

    **Returns**\n
    k_row, k_col: 1D array
        Momentum coordinates of the row and column.
    axis_extent: list
        Extent of the two momentum axis (can be used directly in imshow).
    k_rowgrid, k_colgrid: 2D array
        Row and column mesh grid generated from the coordinates
        (can be used directly in pcolormesh).
    """

    nr, nc = img.shape
    pxla, pxlb = map(np.array, [pxla, pxlb])

    rowdist = range(nr) - pxlb[0]
    coldist = range(nc) - pxlb[1]

    if equiscale == True:
        # Use the same conversion factor along both x and y directions (need k_ab)
        d_ab = norm(pxla - pxlb)
        # Calculate the pixel to momentum conversion factor
        xratio = yratio = k_ab / d_ab

    else:
        # Calculate the conversion factor along x and y directions separately (need coorda)
        dy_ab, dx_ab = pxla - pxlb
        kyb, kxb = kcoordb
        kya, kxa = kcoorda
        # Calculate the column- and row-wise conversion factor
        xratio = (kxa - kxb) / (pxla[1] - pxlb[1])
        yratio = (kya - kyb) / (pxla[0] - pxlb[0])

    k_row = rowdist * yratio + kcoordb[0]
    k_col = coldist * xratio + kcoordb[1]

    # Calculate other return parameters
    pfunc = partial(detector_coordiantes_2_k_koordinates, fr=yratio, fc=xratio)
    k_rowgrid, k_colgrid = np.meshgrid(k_row, k_col)

    # Assemble into return dictionary
    kcalibdict = {}
    kcalibdict['axes'] = (k_row, k_col)
    kcalibdict['extent'] = (k_col[0], k_col[-1], k_row[0], k_row[-1])
    kcalibdict['coeffs'] = (yratio, xratio)
    kcalibdict['grid'] = (k_rowgrid, k_colgrid)

    if ret == 'all':
        return kcalibdict
    elif ret == 'func':
        return pfunc
    else:
        return project(kcalibdict, ret)


def apply_distortion_correction(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    X: str = "X",
    Y: str = "Y",
    newX: str = "Xm",
    newY: str = "Ym",
    type: str = "mattrans",
    **kwds: dict,
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Calculate and replace the X and Y values with their distortion-corrected version.
    This method can be reused.

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

    if type == "mattrans":  # Apply matrix transform
        if "warping" in kwds:
            warping = kwds.pop("warping")
            df[newX], df[newY] = perspective_transform(
                df[X],
                df[Y],
                warping,
                **kwds,
            )
            return df
        else:
            raise NotImplementedError
    elif type == "tps" or type == "tps_matrix":
        if "dfield" in kwds:
            dfield = kwds.pop("dfield")
            out_df = df.map_partitions(
                apply_dfield,
                dfield,
                X=X,
                Y=Y,
                newX=newX,
                newY=newY,
                **kwds,
            )
            return out_df
        elif "rdeform_field" in kwds and "cdeform_field" in kwds:
            rdeform_field = kwds.pop("rdeform_field")
            cdeform_field = kwds.pop("cdeform_field")
            print(
                "Calculating inverse Deformation Field, might take a moment...",
            )
            dfield = generate_dfield(rdeform_field, cdeform_field)
            out_df = df.map_partitions(
                apply_dfield,
                dfield,
                X=X,
                Y=Y,
                newX=newX,
                newY=newY,
                **kwds,
            )
            return out_df
        else:
            raise NotImplementedError


def append_k_axis(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    x0: float,
    y0: float,
    X: str = "X",
    Y: str = "Y",
    newX: str = "kx",
    newY: str = "ky",
    **kwds: dict,
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """Calculate and append the k axis coordinates (kx, ky) to the events dataframe.
    This method can be reused.

    :Parameters:
        df : dataframe
            Dataframe to apply the distotion correction to.
        x0:
            center of the k-image in row pixel coordinates
        y0:
            center of the k-image in column pixel coordinates
        X, Y: | 'X', 'Y'
            Labels of the source columns
        newX, newY: | 'ky', 'ky'
            Labels of the destination columns
        **kwds:
            additional keywords for the momentum conversion

    :Return:
        dataframe with added columns
    """

    if "fr" in kwds and "fc" in kwds:
        df[newX], df[newY] = detector_coordiantes_2_k_koordinates(
            rdet=df[X],
            cdet=df[Y],
            r0=x0,
            c0=y0,
            **kwds,
        )
        return df

    else:
        raise NotImplementedError


def detector_coordiantes_2_k_koordinates(
    rdet: float,
    cdet: float,
    rstart: float,
    cstart: float,
    r0: float,
    c0: float,
    fr: float,
    fc: float,
    rstep: float,
    cstep: float,
) -> Tuple[float, float]:
    """
    Conversion from detector coordinates (rdet, cdet) to momentum coordinates (kr, kc).
    """

    rdet0 = rstart + rstep * r0
    cdet0 = cstart + cstep * c0
    kr = fr * ((rdet - rdet0) / rstep)
    kc = fc * ((cdet - cdet0) / cstep)

    return (kr, kc)


def apply_dfield(
    df: Union[pd.DataFrame, dask.dataframe.DataFrame],
    dfield: np.ndarray,
    X: str = "X",
    Y: str = "Y",
    newX: str = "Xm",
    newY: str = "Ym",
) -> Union[pd.DataFrame, dask.dataframe.DataFrame]:
    """
    Application of the inverse displacement-field to the dataframe coordinates

    :Parameters:
        df : dataframe
            Dataframe to apply the distotion correction to.
        dfield:
            The distortion correction field. 3D matrix,
            with column and row distortion fields stacked along the first dimension.
        X, Y: | 'X', 'Y'
            Labels of the source columns
        newX, newY: | 'Xm', 'Ym'
            Labels of the destination columns

    :Return:
        dataframe with added columns
    """

    x = df[X]
    y = df[Y]

    df[newX], df[newY] = (
        dfield[0, np.int16(x), np.int16(y)],
        dfield[1, np.int16(x), np.int16(y)],
    )
    return df


def generate_dfield(
    rdeform_field: np.ndarray,
    cdeform_field: np.ndarray,
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
    # Interpolate to 2048x2048 grid of the detector coordinates
    grid_x, grid_y = np.mgrid[
        0 : cdeform_field.shape[0] : (cdeform_field.shape[0] / 2048),
        0 : cdeform_field.shape[1] : (cdeform_field.shape[1] / 2048),
    ]
    XY = []
    Z = []
    for i in np.arange(cdeform_field.shape[0]):
        for j in np.arange(cdeform_field.shape[1]):
            XY.append([rdeform_field[i, j], cdeform_field[i, j]])
            Z.append(2048 / cdeform_field.shape[0] * i)

    inv_rdeform_field = griddata(np.asarray(XY), Z, (grid_x, grid_y))

    XY = []
    Z = []
    for i in np.arange(cdeform_field.shape[0]):
        for j in np.arange(cdeform_field.shape[1]):
            XY.append([rdeform_field[i, j], cdeform_field[i, j]])
            Z.append(2048 / cdeform_field.shape[1] * j)

    inv_cdeform_field = griddata(np.asarray(XY), Z, (grid_x, grid_y))

    # TODO: what to do about the nans at the boundary? leave or fill with zeros?
    # inv_rdeform_field = np.nan_to_num(inv_rdeform_field)
    # inv_rdeform_field = np.nan_to_num(inv_cdeform_field)

    dfield = np.asarray([inv_rdeform_field, inv_cdeform_field])

    return dfield


def perspective_transform(
    x: float,
    y: float,
    M: np.ndarray,
) -> Tuple[float, float]:
    """Implementation of the perspective transform (homography) in 2D.

    :Parameters:
        x, y: numeric, numeric
            Pixel coordinates of the original point.
        M: 2d array
            Perspective transform matrix.

    :Return:
        xtrans, ytrans: numeric, numeric
            Pixel coordinates after projective/perspective transform.
    """

    denom = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    xtrans = (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / denom
    ytrans = (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / denom

    return xtrans, ytrans
