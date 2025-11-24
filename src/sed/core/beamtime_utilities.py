"""Beamtime utilities for diagnostic plotting and data overview.

This module provides tools for creating interactive diagnostic dashboards
to monitor beamtime data quality and characteristics.
"""

from typing import List, Tuple
import sys
import numpy as np
import dask
from bokeh.plotting import curdoc, figure, show
from bokeh.layouts import gridplot
import bokeh.palettes as bp
from bokeh.io import output_notebook
from dask.diagnostics import ProgressBar

import sed


class HiddenPrints:
    """Context manager to suppress stdout."""
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        

def make_bins(**kwargs) -> Tuple[List, List, List]: 
    """ translation method for defining bins in the hextof style.

    example:
    >>> make_bins(dldPosX=(0, 100, 1), dldTimeSteps=(0, 100, 1))
    ([100, 100], ['mass', 'time'], [(0.5, 99.5), (0.5, 99.5)])
    
    usage:
    sp.compute(*make_bins(dldPosX=(0, 100, 1), dldTimeSteps=(0, 100, 1)))
    """ 
    bins: List[int] = []
    axes: List[str] = []
    ranges: List[List[float,float]] = []
    for k,v in kwargs.items():
        axes.append(k)
        start, stop, step = v
        n_bins = np.floor((stop-start)/step).astype(int)
        bins.append(n_bins)
        ranges.append([start+step/2, start+(n_bins-0.5)*step])
    return bins, axes, ranges


def defBins(lb, ub, st):
    """Create bin edges for histogram binning.
    
    Args:
        lb: Lower bound
        ub: Upper bound
        st: Step size
        
    Returns:
        np.ndarray: Array of bin centers
    """
    return np.linspace(lb + st/2, ub - st/2, int(np.round((ub - lb) / st)))


def synthetizei0monitor(df, i0in):
    """Synthesize i0 monitor values from binned data.
    
    Args:
        df: DataFrame partition with trainId and pulseId
        i0in: Input i0 monitor data
        
    Returns:
        Synthesized i0 monitor values
    """
    return i0in[((np.floor(df['trainId'])/20).astype(int))*60+(df['pulseId']/10).astype(int)]


def get_minmaxstd(dd, channel, dropna=False):
    """Get min, max, and std of a channel from dask dataframe.
    
    Args:
        dd: Dask dataframe
        channel: Channel name to compute statistics for
        dropna: Whether to drop NA values before computing
        
    Returns:
        Tuple of (min, max, std) values
    """
    if dropna:
        to_compute = dd[channel].dropna().min(), dd[channel].dropna().max(), dd[channel].dropna().std()
    else:
        to_compute = dd[channel].min(), dd[channel].max(), dd[channel].std()
    with ProgressBar():
        return dask.compute(to_compute)


def plot_dashboard(prc, window=100, plot=True, pbar=False, channels=None, filters=None):
    """Create diagnostic plots for run overview.
    
    This function creates comprehensive interactive plots using Bokeh to monitor
    beamtime data quality. It generates plots for:
    - Pulse vs train statistics
    - GMD (Gas Monitor Detector) readings (if available)
    - BAM (Beam Arrival Monitor) data (if available)
    - Delay stage positions (if available)
    - DLD (Delay Line Detector) patterns
    - i0 monitor
    - Optical diode (if available)
    
    Args:
        prc (SedProcessor): SedProcessor object with loaded data
        window (int): Window size for rolling average of BAM and GMD. Default: 100
        plot (bool): Whether to show plots automatically. Default: True
        pbar (bool): Whether to show progress bars. Default: False
        channels (list): List of channel names to include in plots. If None, uses default list.
                        Only channels that exist in the dataframe will be plotted.
                        Default: ['trainId', 'pulseId', 'gmdBda', 'delayStage', 'bam', 'opticalDiode']
        filters (dict): Dictionary of column filters with [min, max] ranges.
                       Example: {'pulseId': [0, 500], 'dldPosX': [0, 1100]}
                       If None, uses default ranges.
        
    Returns:
        tuple: Tuple of bokeh figure lists (pulse_plots, train_plots, dld_plots, laser_plots)
        
    Raises:
        ValueError: If dataframe is empty or invalid
    """
    context = dask.diagnostics.ProgressBar() if pbar else HiddenPrints()
    with context:

        curdoc().theme = 'dark_minimal'
        df = prc.dataframe
        
        # Apply filters if provided
        if filters is None:
            # Default filters
            filters = {
                'pulseId': [0, 500],
                'dldPosX': [0, 1100],
                'dldPosY': [0, 820]
            }
        
        # Apply each filter if the column exists
        for col, (min_val, max_val) in filters.items():
            if col in df.columns:
                df = df[(df[col] >= min_val) & (df[col] < max_val)]
        
        prc.dataframe = df
        
        first = prc.dataframe['trainId'].head(1).values[0]
        last = prc.dataframe['trainId'].tail(1).values[0]

        # with dask.diagnostics.ProgressBar():
        df = prc.dataframe[['electronId','trainId']].copy()
        train_id = df.groupby('trainId').count()
        
        # Define channels to plot - use provided list or default
        if channels is None:
            channels = ['trainId', 'pulseId', 'gmdBda', 'delayStage', 'bam', 'opticalDiode']
        
        # Only include channels that actually exist in the dataframe
        cols = [c for c in channels if c in prc.dataframe.columns]
        
        # Ensure essential columns are present
        if 'trainId' not in cols or 'pulseId' not in cols:
            raise ValueError("Essential columns 'trainId' and 'pulseId' must be present in the dataframe")

        df = prc.dataframe[cols].copy()
        tail = df.tail(1)
        
        # Drop columns with None values (keep essential columns)
        for c in list(cols):
            if c in ['trainId', 'pulseId']:
                continue
            if tail[c].values[0] is None:
                df = df.drop(c, axis=1)
                cols.remove(c)
        
        # df['trainId'] -= first
        df_train = df.groupby('trainId').mean()
        df_pulse = df.groupby('pulseId').mean()
        df_train_roll = df_train.rolling(window, center=True).mean()
        if pbar:
            print('computing grouped dataframes')
        train_id, df_train, df_pulse, df_train_roll = dask.compute(
            train_id, df_train, df_pulse, df_train_roll
        )

        axes = ['pulseId','trainId'] 
        bins = [defBins(0,600,10),defBins(first-10,last+10,20)] 
        print('binning pulseId vs trainId')
        res_pulse_vs_train = prc.compute(bins=bins, axes=axes, pbar=pbar)
        train_binsize = (last-first)//len(bins[1])

        i0 = res_pulse_vs_train.stack(stacked_dim=['trainId','pulseId']) 
        df['trainId'] -= first
        df['i0Monitor'] = df.map_partitions(synthetizei0monitor, i0in=i0.values) 
        # with dask.diagnostics.ProgressBar(minimum=2):
        if pbar:
            print('computing i0Monitor')
        a3max = np.floor(df['i0Monitor'].max().compute()*1.1)
        axes = ['i0Monitor','pulseId'] 
        bins = [defBins(-0.5,a3max+.5,2),defBins(0,500,10)]
        if pbar:
            print('binning i0Monitor')
        # res_i0monitor = prc.compute(bins=bins, axes=axes, pbar=pbar) 
        res_i0monitor = sed.binning.bin_dataframe(df=df,bins=bins,axes=axes,pbar=pbar)
        a1 = [] 
        a2 = [] 
        for i in range(res_i0monitor.shape[1]):
            a1.append(float(res_i0monitor.i0Monitor[res_i0monitor.isel(pulseId=i).argmax()])) 
            a2.append(float(res_i0monitor.pulseId[i]))
        a1 = np.array(a1)
        a2 = np.array(a2)

        train_plots = [] 
        pulse_plots = [] 
        dld_plots = []
        laser_plots = []
        # pulseId vs trainId
        xrange = float(res_pulse_vs_train.coords['trainId'].min()), float(res_pulse_vs_train.coords['trainId'].max())
        yrange = float(res_pulse_vs_train.coords['pulseId'].min()), float(res_pulse_vs_train.coords['pulseId'].max())
        p = figure(x_range=xrange, y_range=yrange, )
        p.image(
            image=[res_pulse_vs_train.values], 
            x=xrange[0], 
            y=yrange[0], 
            dw=xrange[1]-xrange[0], 
            dh=yrange[1]-yrange[0], 
            palette=bp.Magma[256],
        )
        p.xaxis.axis_label = 'trainId'
        p.yaxis.axis_label = 'pulseId'
        p.title.text = f'pulseId vs trainId'
        p.xaxis.major_label_orientation = np.pi/4
        train_plots.append(p)

        # trainId

        p = figure()
        x = train_id.index.values
        y = train_id['electronId'].values
        p.dot(x,y, size=20, color='white',alpha=.3, legend_label=f'mean {float(train_id.mean()):.1f}')
        x= res_pulse_vs_train.coords['trainId'].values
        y = res_pulse_vs_train.sum('pulseId').values/train_binsize
        p.line(x,y, line_width=2, color='red')
        #set y range to 0-500
        p.y_range.start = 0
        p.xaxis.axis_label = 'trainId'
        p.yaxis.axis_label = 'counts'
        p.title.text = f'Counts vs TrainId'
        p.xaxis.major_label_orientation = np.pi/4
        train_plots.append(p)


        # i0Monitor vs pulseId
        xrange = float(res_i0monitor.coords['pulseId'].min()), float(res_i0monitor.coords['pulseId'].max()) 
        yrange = float(res_i0monitor.coords['i0Monitor'].min()), float(res_i0monitor.coords['i0Monitor'].max()) 
        p = figure(x_range=xrange, y_range=yrange, ) 
        p.image( image=[res_i0monitor.values], x=xrange[0], y=yrange[0], dw=xrange[1]-xrange[0], dh=yrange[1]-yrange[0], palette=bp.Magma[256], )
        p.line(a2,a1,color='white',line_width=2) 
        p.xaxis.axis_label = 'pulseId' 
        p.yaxis.axis_label = 'i0Monitor' 
        p.title.text = f'i0Monitor vs pulseId' 
        pulse_plots.append(p) 
        
        # pulseId
        p = figure() 
        x = res_i0monitor.coords['pulseId'].values 
        y = res_i0monitor.sum('i0Monitor').values 
        p.dot(x,y, size=20, color='white') 
        p.xaxis.axis_label = 'pulseId' 
        p.yaxis.axis_label = 'counts' 
        p.title.text = f'Counts vs PulseId' 
        pulse_plots.append(p) 

        # GMD plots (only if available)
        if 'gmdBda' in df_train.columns:
            # GMD vs trainId
            p = figure()
            x = train_id.index.values
            y = df_train['gmdBda'].values
            p.dot(x,y, size=20, color='white', alpha=0.5)
            xr = df_train_roll.index.values
            yr = df_train_roll['gmdBda'].values
            p.line(xr,yr,color='red',line_width=2) 
            p.xaxis.axis_label = 'trainId'
            p.yaxis.axis_label = 'gmdBda'
            p.title.text = f'gmdBDA vs trainId'
            p.xaxis.major_label_orientation = np.pi/4
            train_plots.append(p)

            # GMD vs pulseId
            if 'gmdBda' in df_pulse.columns:
                p = figure()
                x = df_pulse.index.values
                y = df_pulse['gmdBda'].values
                p.dot(x,y, size=20, color='white')
                p.xaxis.axis_label = 'pulseId'
                p.yaxis.axis_label = 'gmdBda'
                p.title.text = f'gmdBDA vs pulseId'
                pulse_plots.append(p)

        # BAM plots (only if available)
        if 'bam' in df_train.columns:
            # BAM vs trainId
            p = figure()
            x = train_id.index.values
            y = df_train['bam'].values
            p.dot(x,y, size=20, color='white', alpha=0.5)
            xr = df_train_roll.index.values
            yr = df_train_roll['bam'].values
            p.line(xr,yr,color='red',line_width=2) 
            p.xaxis.axis_label = 'trainId'
            p.yaxis.axis_label = 'bam'
            p.title.text = f'BAM vs trainId'
            p.xaxis.major_label_orientation = np.pi/4
            train_plots.append(p)

            # BAM vs pulseId
            if 'bam' in df_pulse.columns:
                p = figure()
                x = df_pulse.index.values
                y = df_pulse['bam'].values
                p.dot(x,y, size=20, color='white')
                p.xaxis.axis_label = 'pulseId'
                p.yaxis.axis_label = 'bam'
                p.title.text = f'BAM vs pulseId'
                pulse_plots.append(p)

        # dld plots
        bins = 100
        axes = ['dldPosX','dldPosY','dldTimeSteps']
        ranges = [[0,1100], [0,820], [12000,15000]]
        # for ax in axes:
        #     ranges.append([prc.dataframe[ax].quantile(0.02), prc.dataframe[ax].quantile(0.98)])
        # ranges = dask.compute(*ranges)
        if pbar:
            print('binning dld')
        res = prc.compute(bins=bins, ranges=ranges, axes=axes, pbar=pbar)
        img_xy = res.sum('dldTimeSteps')
        edc = res.sum(['dldPosX','dldPosY'])
        #get maxima of edc
        center = int(edc.argmax('dldTimeSteps'))
        img_xy = res.isel(dldTimeSteps=slice(center-2,center+2)).sum('dldTimeSteps')

        p = figure()
        x = img_xy.coords['dldPosX'].values
        y = img_xy.coords['dldPosY'].values
        img_xy = img_xy/img_xy.max()
        p.image(image=[img_xy.values], x=x[0], y=y[0], dw=x[-1]-x[0], dh=y[-1]-y[0], palette=bp.Magma[256])
        p.xaxis.axis_label = 'dldPosX'
        p.yaxis.axis_label = 'dldPosY'
        p.title.text = f'dldPosX vs dldPosY'
        dld_plots.append(p)

        p = figure()
        x = edc.coords['dldTimeSteps'].values
        y = edc.values
        p.dot(x,y, size=10, color='white')
        p.line(x,y, color='white')
        p.xaxis.axis_label = 'dldTimeSteps'
        p.yaxis.axis_label = 'counts'
        p.title.text = f'dldTimeSteps'
        p.xaxis.major_label_orientation = np.pi/4
        dld_plots.append(p)

        # Laser/delay stage plots (only if available)
        if 'delayStage' in df_train.columns:
            p = figure()
            x = train_id.index.values
            y = df_train['delayStage'].values
            p.dot(x,y, size=20, color='white', alpha=0.5)
            xr = df_train_roll.index.values
            yr = df_train_roll['delayStage'].values
            p.line(xr,yr,color='red',line_width=2) 
            p.xaxis.axis_label = 'trainId'
            p.yaxis.axis_label = 'delayStage'
            p.title.text = f'delayStage Position vs trainId'
            p.xaxis.major_label_orientation = np.pi/4
            laser_plots.append(p)

        # Optical diode plots (only if available)
        if 'opticalDiode' in df_train.columns:
            p = figure()
            x = train_id.index.values
            xr = df_train_roll.index.values
            y = df_train['opticalDiode'].values
            yr = df_train_roll['opticalDiode'].values
            p.dot(x,y, size=20, color='white', alpha=0.5)
            p.line(xr,yr,color='red',line_width=2) 
            p.xaxis.axis_label = 'trainId'
            p.yaxis.axis_label = 'OpticalDiode'
            p.title.text = f'OpticalDiode vs trainId'
            p.xaxis.major_label_orientation = np.pi/4
            laser_plots.append(p)


        if plot:
            output_notebook() 
            plots = [*train_plots, *pulse_plots, *dld_plots, *laser_plots]
            grid = gridplot(plots, ncols=4, width=300, height=300)
            show(grid)
        else:
            return pulse_plots, train_plots, dld_plots, laser_plots

