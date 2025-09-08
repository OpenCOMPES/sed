from typing import List,Tuple
import sys, os
from datetime import datetime
from tqdm.auto import trange, tqdm

# import numeric packages
import numpy as np
import xarray as xr
import pandas as pd
import dask
from bokeh.plotting import curdoc, figure, show
from bokeh.layouts import gridplot
import bokeh.palettes as bp
from bokeh.io import output_notebook

from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

# Avoid circular import - SedProcessor will be passed as parameter
# from sed import SedProcessor
import sed


DATA_RAW_DIR = "/gpfs/current/raw/hdf/online-0/fl1user3"

def get_available_runs(path=None):
    """return a list of run numbers available from the DAQ at the given path"""
    if path is None:
        path = DATA_RAW_DIR
    allRunNumbers = []
    for num in [int(x.split('_')[4][3:]) for x in os.listdir(path)]:
        if num not in allRunNumbers:
            allRunNumbers.append(num)
    return sorted(allRunNumbers)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

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


def defBins(lb,ub,st):
    return np.linspace(lb+st/2, ub-st/2, int(np.round((ub-lb)/st)))


def pulse_vs_train_diagnostic_plot(run_number):
    """ plot the pulseId vs TrainId."""
    if isinstance(run_number, int):
        prc = SedProcessor(runs=[run_number], config=config, collect_metadata=False)
    else:
        prc = run_number
    first = prc.dataframe['trainId'].head().values[0]
    last = prc.dataframe['trainId'].tail().values[-1]
    axes = ['pulseId','trainId']
    bins = [500,1000]
    ranges = [0,500], [first,last]
    res = prc.compute(bins=bins, ranges=ranges, axes=axes)
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    res.T.plot.imshow(robust=True,ax=ax[0])
    res.sum('trainId').plot(ax=ax[1])
    ax[0].set_title('electrons per shot')
    ax[1].set_title('PulseID count distribution')
    fig.suptitle(f'run {prc.loader.runs}')
    return res
    

def synthetizei0monitor(df, i0in):
    return i0in[((np.floor(df['trainId'])/20).astype(int))*60+(df['pulseId']/10).astype(int)]


def i0Plot(prc,ax1,ax2):
    trainIds=get_minmaxstd(prc.dataframe,'trainId')[0]
    axes = ['pulseId','trainId']
    bins = [defBins(0,600,10),defBins(trainIds[0]-10,trainIds[1]+10,20)]
    res = prc.compute(bins=bins, axes=axes)
    res.plot.imshow(ax=ax1,robust=True)
    
    i0=res.stack(stacked_dim=['trainId','pulseId'])
    
    prc.dataframe['trainId'] -= trainIds[0]
    
    a3 = prc.dataframe.map_partitions(synthetizei0monitor, i0in=i0.values)
    prc.dataframe['i0Monitor'] = a3
    axes = ['i0Monitor','pulseId']
    bins = [defBins(-0.5,200.5,2),defBins(0,500,10)]
    res1 = prc.compute(bins=bins, axes=axes )   
    res1.plot.imshow(ax=ax2,robust=True)

    a1 = []
    a2 = []
    for i in range(res1.shape[1]):
        a1.append(res1.i0Monitor[res1.isel(pulseId=i).argmax()])
        a2.append(res1.pulseId[i])

    ax2.plot(a2,a1,marker='.',color='red')


def bamPlot(prc,ax):
    axes = ['bam','pulseId']
    bins = [defBins(-5500,-4500,1),defBins(0,500,1)]
    res = prc.compute(bins=bins, axes=axes)
    res.plot.imshow(ax=ax,robust=True)
    

def optiPlot(prc,ax):
    axes = ['opticalDiode','pulseId']
    bins = [defBins(0,3.4,0.01),defBins(0,500,1)]
    res = prc.compute(bins=bins, axes=axes)
    res.plot.imshow(ax=ax,robust=True)


def debugPlots(prc):
    fig,ax = plt.subplots(2,2,figsize=(6,4),layout='constrained')
    # optiPlot(prc=prc,ax=ax[0][0])
    bamPlot(prc=prc,ax=ax[1][0])
    i0Plot(prc=prc,ax1=ax[0][1],ax2=ax[1][1])
    
    



    # ax[0][0].set_title(f'bam {runs[0]}')


def get_minmaxstd(dd,channel,dropna=False):
    
    if dropna:
        to_compute = dd[channel].dropna().min(),dd[channel].dropna().max(), dd[channel].dropna().std()
    else:
        to_compute = dd[channel].min(),dd[channel].max(), dd[channel].std()
    with ProgressBar():
        return dask.compute(to_compute)


def plot_dashboard(prc, window=100, plot=True, pbar=False):
    """ Diagnostic plots for run overview.
    
    Args:
        prc (SedProcessor): SedProcessor object with loaded data
        window (int): window size for rolling average of BAM and GMD
        
    Returns:
        tuple: tuple of bokeh figures
    """

    context = dask.diagnostics.ProgressBar() if pbar else HiddenPrints()
    with context:

        curdoc().theme = 'dark_minimal'
        df = prc.dataframe
        df = df[(df['pulseId'] < 500) & (df['pulseId'] >= 0)]
        df = df[(df['dldPosX'] > 0) & (df['dldPosX'] < 1100)]
        df = df[(df['dldPosY'] > 0) & (df['dldPosY'] < 820)]
        prc.dataframe = df
        
        first = prc.dataframe['trainId'].head(1).values[0]
        last = prc.dataframe['trainId'].tail(1).values[0]

        # with dask.diagnostics.ProgressBar():
        df = prc.dataframe[['electronId','trainId']].copy()
        train_id = df.groupby('trainId').count()
        cols = ['trainId','pulseId','gmdBda','delayStage','bam','opticalDiode']
        df = prc.dataframe[cols].copy()
        tail = df.tail(1)
        for c in cols:
            if tail[c].values[0] is None:
                df = df.drop(c, axis=1)
        
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

        # GMD
        p = figure()
        x = train_id.index.values
        y = df_train['gmdBda'].values
        p.dot(x,y, size=20, color='white', alpha=0.5)
        xr = df_train_roll.index.values
        yr = df_train_roll['gmdBda'].values
        p.line(xr,yr,color='red',line_width=2) 
        p.xaxis.axis_label = 'trainId'
        p.yaxis.axis_label = 'bam'
        p.title.text = f'gmdBDA vs trainId'
        p.xaxis.major_label_orientation = np.pi/4
        train_plots.append(p)

        p = figure()
        x = df_pulse.index.values
        y = df_pulse['gmdBda'].values
        p.dot(x,y, size=20, color='white')
        p.xaxis.axis_label = 'pulseId'
        p.yaxis.axis_label = 'bam'
        p.title.text = f'gmdBDA vs pulseId'
        pulse_plots.append(p)

        #BAM
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

        # laser plots
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

        p = figure()
        x = train_id.index.values
        xr = df_train_roll.index.values
        if 'opticalDiode' in df_train.columns:
            y = df_train['opticalDiode'].values
            yr = df_train_roll['opticalDiode'].values
        else:
            y= np.zeros_like(x)
            yr = np.zeros_like(xr)
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

#########################
#       OBSOLETE        #
#########################

def runInfo(runNumber,path=None):
    """ Fetch statistical information from some of the columns, to populate a runs overview"""
    if path is None:
        path = PATH_TO_RAW_DATA
    
    dld_time = '/FL1/Experiment/PG/Hextof/Detector/control info'
    dld_detector_id = '/FL1/Experiment/PG/Hextof/Detector/control info'
    dld_sector_id = '/FL1/Experiment/PG/Hextof/Detector/control info'
    dld_aux_0 = '/FL1/Experiment/PG/Hextof/Detector/monitor 0'
    dld_aux_1 = '/FL1/Experiment/PG/Hextof/Detector/monitor 0'
    dld_pos_x = '/FL1/Experiment/PG/Hextof/Detector/monitor 1'
    dld_pos_y = '/FL1/Experiment/PG/Hextof/Detector/monitor 2'
    dld_microbunch_id = '/FL1/Experiment/PG/Hextof/Detector/monitor 3'
    delay_stage = '/uncategorised/FLASH1_USER2/FLASH.SYNC/LASER.LOCK.EXP/FLASH1.MOD1.PG.OSC/FMC0.MD22.1.POSITION.RD/dset'
    streak_cam = '/FL1/Experiment/Pump probe laser/streak camera delay time'
    optical_diode = '/FL1/Experiment/PG/SIS8300 100MHz ADC/CH9/pulse energy/TD'
    i0_monitor = '/FL1/Experiment/PG/SIS8300 100MHz ADC/CH6/TD'
    bam = '/FL1/Electron Diagnostic/BAM/4DBC3/electron bunch arrival time (low charge)'
    gmd_bda = '/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy BDA copy'
    gmd_tunnel = '/FL1/Photon Diagnostic/GMD/Pulse resolved energy/energy tunnel copy'
    macro_bunch_pulse_id = '/FL1/Timing/Bunch pattern/train index 1.sts'
    time_stamp = '/Timing/time stamp/fl1user2'
    monochromatorPhotonEnergy = '/FL1/Beamlines/PG/Monochromator/monochromator photon energy'

    from beamtimedaqaccess import BeamtimeDaqAccess, accessHdf

    daqAccess = BeamtimeDaqAccess.create(path)
    hdfAccess  = accessHdf(path)

    time,pId = hdfAccess.allValuesOfRun(time_stamp,runNumber)
    timeStamp = (time[0,0], time[-1,0])
    MBunches = pId[1]-pId[0]
    dd =  {
        'runNumber':runNumber,
        'timeFrom':datetime.utcfromtimestamp(timeStamp[0]).strftime('%Y-%m-%d %H:%M:%S'),
        'timeTo':datetime.utcfromtimestamp(timeStamp[1]).strftime('%Y-%m-%d %H:%M:%S'),
        'timeStampFrom':timeStamp[0],
        'timeStampTo':timeStamp[1],
        'macrobunchFrom':pId[0],
        'macrobunchTo':pId[1], 
        'macrobunhes':MBunches, 
    }
    try:
        values,pId = hdfAccess.allValuesOfRun(dld_time,runNumber)
        x = values.flatten()
        x = x[x>0]
        x = x[x<1000000]
        n_el = len(x)
#         dd['microbunches'] = values.shape[1]
        dd['n_el'] = n_el
        dd['elPerMb']= n_el/MBunches
        values,pId = hdfAccess.allValuesOfRun(dld_aux_0,runNumber)
        df = pd.DataFrame(values).dropna().mean()
        dd['sa'] = df[0]
        dd['tof'] = df[1]
        dd['ext'] = df[2]
        dd['aux4'] = df[3]
        dd['aux16'] = df[15]
    except:
        dd['microbunches'] = np.nan
        dd['n_el'] = np.nan
        dd['elPerMb']= np.nan
        dd['sa'] = np.nan
        dd['tof'] = np.nan
        dd['ext'] = np.nan
        dd['aux4'] = np.nan
        dd['aux16'] = np.nan
        
    values, pId = hdfAccess.allValuesOfRun(monochromatorPhotonEnergy,runNumber)
    df = pd.DataFrame(values.flatten()).dropna()
    dd['monochromatorPhotonEnergy'] = float(df.mean())
    dd['monochromatorPhotonEnergy_std'] = float(df.std())

    values,pId = hdfAccess.allValuesOfRun(delay_stage,runNumber)
    df = pd.DataFrame(values[:,0]).dropna()
    dd['delayFrom'] = float(df.min())
    dd['delayTo'] = float(df.max())
    dd['delayAmp'] = dd['delayTo'] - dd['delayFrom']
    
    values,pId = hdfAccess.allValuesOfRun(optical_diode,runNumber)
    df = pd.DataFrame(values.flatten()).dropna()
    dd['opticalDiode_mean'] = float(df.mean())
    dd['opticalDiode_std'] = float(df.std())
    dd['opticalDiode_median'] = float(df.median())
    
    values, pId = hdfAccess.allValuesOfRun(i0_monitor,runNumber)
    df = pd.DataFrame(values.flatten()).dropna()
    dd['i0_monitor_mean'] = float(df.mean())
    dd['i0_monitor_std'] = float(df.std())
    dd['i0_monitor_median'] = float(df.median())

#     try:
#         values, pId = hdfAccess.allValuesOfRun(streak_cam,runNumber)
#         df = pd.DataFrame(values.flatten()).dropna()
#         dd['streakCam_mean'] = float(df.mean())
#         dd['streakCam_std'] = float(df.std())
#     except:
#         dd['streakCam_mean'] = np.nan
#         dd['streakCam_std'] = np.nan
    
    values, pId = hdfAccess.allValuesOfRun(bam,runNumber)
#     if np.isnan(dd['microbunches']):
#         dd['microbunches'] = values.shape[1]

    df = pd.DataFrame(values.flatten()).dropna()
    dd['bam_mean'] = float(df.mean())
    dd['bam_std'] = float(df.std())
    
    values, pId = hdfAccess.allValuesOfRun(gmd_bda,runNumber)
    df = pd.DataFrame(values.flatten()).dropna()
    dd['gmd_bda_mean'] = float(df.mean())
    dd['gmd_bda_std'] = float(df.std())
    
    values, pId = hdfAccess.allValuesOfRun(gmd_tunnel,runNumber)
    df = pd.DataFrame(values.flatten()).dropna()
    dd['gmd_tunnel_mean'] = float(df.mean())
    dd['gmd_tunnel_std'] = float(df.std())
    
    df = pd.DataFrame(dd,index=[runNumber])

    return pd.DataFrame(dd,index=[runNumber])


def updateRunInfoCSV(file,data_dir,skip_runs=[],reload_last=True):
    available_runs = get_available_runs(data_dir)
    try:
        df = pd.read_csv(file,index_col=0)
        if reload_last:
            df = df.drop(index=df.index[-1])
        new_runs = [x for x in available_runs if x not in df.index]
         
    except:
        df = None
        new_runs = available_runs
    for run in skip_runs:
        new_runs.remove(run)
    for run in tqdm(new_runs):
        print(f'Loading run {run}')
        try:
            df = runInfo(run)
            with open(file, 'a') as f:
                df.to_csv(f, header=run==available_runs[0])
        except Exception as e:
            if not 'There is no file with specified run number' in str(e):
                print(f'run {run}:\n', type(e),e)
            else:
                print(f'No file for run Number {run}')
    
   
def get_temperature(runNumber,Mb_step=1000, path=None):
    """ returns cryo temperature and sample temperature for macrobunches """
    if path is None:
        path = PATH_TO_RAW_DATA
    from beamtimedaqaccess import BeamtimeDaqAccess, accessHdf

    daqAccess = BeamtimeDaqAccess.create(path)
    hdfAccess  = accessHdf(path)
   # import numpy as np
    time,pId = hdfAccess.allValuesOfRun('/Timing/time stamp/fl1user2',runNumber)
    timeStamp = (time[0,0], time[-1,0])
    MBunches = pId[1]-pId[0]
    cryo_res=[]
    sample_res=[]
    bunches=[]
    i=pId[0]
    while i<pId[1]-Mb_step:
        cryo=hdfAccess.valuesOfInterval('/FL1/Experiment/PG/Hextof/Detector/monitor 0', pulseIdInterval=(i,i+Mb_step))[...,4]
        cryo_mean = cryo[np.logical_not(np.isnan(cryo))].mean()
        cryo_res.append(cryo_mean)
        sample=hdfAccess.valuesOfInterval('/FL1/Experiment/PG/Hextof/Detector/monitor 0', pulseIdInterval=(i,i+Mb_step))[...,5]
        sample_mean = sample[np.logical_not(np.isnan(sample))].mean()
        sample_res.append(sample_mean)
        bunches.append(i)
        i+=Mb_step
    return cryo_res,sample_res,bunches
