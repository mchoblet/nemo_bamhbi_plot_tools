#Packages needed

#GENERAL
import glob #for getting filepaths
from types import SimpleNamespace 
# Suppress all runtime warnings
import warnings
warnings.filterwarnings("ignore")
import pdb
def debug(): pdb.set_trace() #shortcut
import os

#GRIDDED DATA HANDLING
import xarray as xr
xr.set_options(keep_attrs=True) #VERY IMPORTANT HERE, BUT BEWARE OF CONSEQUENCES WHEN USING UNITS!
import numpy as np

#GENERAL PLOTTING 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.dates as md
import seaborn as sns #for the colorpalette

#CARTOPY
import cartopy.crs as ccrs
projection=ccrs.PlateCarree()
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#MISCELLANEA
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

"""
#enable smooth scrolling (only works in jupyter notebook)
%%javascript

Jupyter.keyboard_manager.command_shortcuts.remove_shortcut("up");
Jupyter.keyboard_manager.command_shortcuts.remove_shortcut("down");
"""

#set reasonable matplotlib fontsize settings from the start
mpl.rcParams.update({
    'font.size': 12,         # Controls default text sizes
    'axes.labelsize': 14,    # Fontsize of the x and y labels
    'axes.titlesize': 16,    # Fontsize of the axes title
    'xtick.labelsize': 12,   # Fontsize of the tick labels
    'ytick.labelsize': 12,   # Fontsize of the tick labels
    'xtick.minor.width': 1.0,# Thickness of the minor x-axis ticks
    'ytick.minor.width': 1.0,# Thickness of the minor y-axis ticks
    'xtick.major.size': 8,   # Length of the major x-axis ticks in points
    'ytick.major.size': 8,   # Length of the major y-axis ticks in points
    'xtick.minor.size': 4,   # Length of the minor x-axis ticks in points
    'ytick.minor.size': 4,   # Length of the minor y-axis ticks in points
    'axes.grid': True,          # Show grid or not
    'grid.color': 'gray',       # Color of the grid lines
    'grid.linestyle': '--',     # Style of the grid lines
    'grid.alpha': 0.5,          # Transparency of the grid lines
    'legend.fontsize': 12,      # Fontsize of the legend
    'legend.title_fontsize': 14,# Fontsize of the legend title
    'legend.frameon': True,     # Draw a frame around the legend
    'legend.edgecolor': 'black',# Color of the legend edge
    'legend.facecolor': 'white',# Background color of the legend
    'figure.figsize': (8, 5),   # Figure size in inches
    'figure.dpi': 300           # Dots per inch
})

#for resetting values
#mpl.rcParams.update(matplotlib.rcParamsDefault)


"""
PREPROCESSING
"""


def convert_xy(ds,y='nav_lat',x='nav_lon',grid=None):
    """
    convert latitudes and longitudes from nested to normal.
    The basic output format is a bit unconvenient for me when working with xarray
    
    Input:
        ds: xarray dataset (any shape)
        y: name of latitudes in coordinates
        x: name of longitudes in coordinates
    
    Output:
        ds: Dataset with simple 1D coordinates for latitudes and longitudes.
    """
    latitudes = ds[y].values
    longitudes = ds[x].values
    # Flatten latitudes and longitudes. get rid of -1 that are sometimes in the model data
    flat_latitudes = np.setdiff1d(np.unique(latitudes.flatten()),[-1])
    flat_longitudes = np.setdiff1d(np.unique(longitudes.flatten()),[-1])

    
    ds[y]=('lat',flat_latitudes)
    ds[x]=('lon',flat_longitudes)
    ds=ds.rename({x:'lon',y:'lat'})
    if grid is None:
        x='x';y='y'
    else:
        x='x_'+grid; y='y_'+grid;
    
    ds=ds.rename({x:'lon',y:'lat'})

    #rename 'inner' grid variables
    for a in [x for x in list(ds.dims)  if 'inner' in x]:
        if 'y' in a:
            ds=ds.rename({a:'lat'})
        elif 'x' in a:
            ds=ds.rename({a:'lon'})
    ds=ds.set_index({'lat':'lat','lon':'lon'})
    return ds
    
def mask_data(ds,ls_mask):
    """
    Applying a land sea mask is straight forward to do with the xarray where function.
    Coordinates with ls_mask 0 recieve value nan.
    """
    #ensure that latitude and longitude values are the same (depending on the grid (U,V) they might not fit the ls_mask)
    #if ~np.array_equal(ls_mask['lat'].values,ds['lat'].values):
    ds['lat']=ls_mask['lat'].values
    
    #if ~np.array_equal(ls_mask['lat'].values,ds['lon'].values):
    ds['lon']=ls_mask['lon'].values
    
    return xr.where(ls_mask,ds,np.nan)

"""
FUNCTIONS TO LOAD NEMO-BAMHBI SPECIFIC DATA
"""

def load_domain(
              domain_path=None, #domain either a filepath, or an xarray dataarray. latter needed when reading in data (used for masking)
              y='nav_lat', #name of y-dimension
              x='nav_lon', #name of x-dimension
              depth_var='deptht', #name of depth variable. eventually rename that later to match your model data (depends if data on U_, V_, W_ grid)
              top_level='top_level',
             ):
    """
    Preprocessing wrapper for domain file
    
    1. Load 
    2. Convert Lat/Lon
    3. Land Sea Masking

    """
    
    if domain_path==None:
        print('Nothing loaded. Please indicate File path to domain file of model run')
    else:
        global domain #such that it can be accessed for sure
        domain=xr.open_dataset(domain_path).squeeze().rename({'nav_lev':depth_var})
        domain=convert_xy(domain)
        global ls_mask
        ls_mask=domain[top_level]
    return domain,ls_mask

def load_data(data_paths,
              load=True, #set false if data bigger than RAM! this will take some time to load, but its better to do it at some point before plotting.
              domain_data=None, #either specify a domain data array, or I will just take the global one.
              grid=None, #e.g. for T variables name this argument 'grid_T'
              time_var='time_counter',
              ben_lev='benlvl', #also keep benthic level variables
              variabs=None, #either a list of variables
              domain_kwargs={
              'y':'nav_lat', #name of y-dimension
              'x':'nav_lon', #name of x-dimension
              'depth_var':'deptht', #name of depth variable. eventually rename that later to match your model data (depends if data on U_, V_, W_ grid)
              'top_level':'top_level',
              },
):
    """
    Preprocessing wrapper for data file
    
    1. Load
    1B. Select relevant variables
    2. Convert Lat/Lon
    3. Select 2D,3d,4D variables
    4. Land Sea Masking
    """
    #set name of dimensions according to the grid. If  grid is None, just deduce grid based on file path name.
    #y='nav_lat' #name of y-dimension
    #x='nav_lon' #name of x-dimension
    #depth_var='deptht' #name of depth variable. eventually rename that later to match your model data (depends if data on U_, V_, W_ grid)

    #just take some filepath
    if isinstance(data_paths,list):
         p=os.path.splitext(os.path.basename(data_paths[0]))[0]
    else:
         p=os.path.splitext(os.path.basename(data_paths))[0]
    if ('btrc_T' in p) or ('ptrc_T' in p):
        y='nav_lat'
        x='nav_lon' 
        depth_var='deptht'
        grid=None
    elif 'grid_T' in p:
        y='nav_lat_grid_T';
        x='nav_lon_grid_T'
        depth_var='deptht'
        grid='grid_T'
    elif 'grid_U' in p:
        y='nav_lat_grid_U';
        x='nav_lon_grid_U'
        depth_var='depthu'
        grid='grid_U'
    elif 'grid_V' in p:
        y='nav_lat_grid_V';
        x='nav_lon_grid_V'
        depth_var='depthv'
        grid='grid_V'
    elif 'grid_W' in p:
        y='nav_lat';
        x='nav_lon'
        depth_var='depthw'
        grid=None
    #LOAD
    if variabs is not None:
        preprocess=lambda ds: ds[variabs]
    else:
        preprocess=None
    #try:
    data=xr.open_mfdataset(data_paths,preprocess=preprocess)
    #except:
    #    data=xr.open_mfdataset(data_paths,concat_dim='time',combine='nested')
    if load:
        data=data.load()
    #CONVERT LAT/LON
    data=convert_xy(data,y=y,x=x,grid=grid)

    #SELECT 2D/3D/4D Variables
    lat_var='lat'
    lon_var='lon'

    d2=sorted([lat_var,lon_var])
    d3=sorted([time_var,lat_var,lon_var])
    d4=sorted([time_var,lat_var,lon_var,depth_var])    
    d4b=sorted([time_var,lat_var,lon_var,ben_lev])    
    to_keep = [var for var in data.data_vars if sorted(data[var].dims) in [d2, d3, d4,d4b]]
    
    data=data[to_keep]

    #MASK DATA
    if type(domain_data)==xr.core.dataset.Dataset:
        data=mask_data(data,domain_data[domain_kwargs['top_level']])
    elif type(domain_data)==str:
        domain2=load_domain(domain_data,y=domain_kwargs['y'],x=domain_kwargs['x'],depth_var=domain_kwargs['depth_var'],top_level=domain_kwargs['top_level'])
        data=mask_data(data,domain2[domain_kwargs['top_level']])
    return data


#river discharge data: read from the forcing data (lat,lon,time). I provide scripts to calculate river discharge for high and low resolution
#discharge. The indices come from analysing where the forcing data is not nan

def load_rivers(path_prefix,domain,mode='LR',lim=None,time=['1960','2022']):
    """
    Input:
        path_prefix: 
            e.g. '/gpfs/scratch/acad/bsmfc/mchoblet/BSFS_BIO/GEO_LR/runoff/runoff_y*.nc'
            or '/gpfs/scratch/acad/bsmfc/mchoblet/BSFS_BIO/GEO/runoff/mast_runoff_y*.nc'
        domain:
            domain file used to get surface area (needed for unit conversion)
        mode (refering to the spatial model resolution that you use)
            LR: 'low resolution forcing'
            HR: 'low resolution'
        lim:
            integer, restrict river data to 'lim' files
    Output:
        Diictionary with one entry for each river.
    """
    
    rivers_hr_dict={
    'danube_1':[183,97],
    'danube_2':[178,95],
    'danube_3':[172,95],
    'danube_4':[169,92],
    'danube_5':[159,90],
    'dnestr':[207,124],
    'dnepr_1':[227,163],
    'dnepr_2':[228,163],
    'kizil':[34,343],
    'rioni':[52,569],
    'sakarya':[8,143],
    }

    rivers_lr_dict={
        'danube_1':[29, 13],
        'danube_2':[31, 13],
        'danube_3':[34, 13],
        'dnestr':[39, 18],
        'dnepr': [41, 22],
        'kizil':[7, 46],
        'rioni':[10, 77],
        'sakarya': [3, 17],
    }

    if mode.lower()=='hr':
        rivers_dict=rivers_hr_dict
        danubes=['danube_'+str(i) for i in range(1,6)]
        dneprs=['dnepr_1','dnepr_2']
    else:
        rivers_dict=rivers_lr_dict
        danubes=['danube_'+str(i) for i in range(1,4)]
        dneprs=['dnepr']    

    #get rivers from North western shelf in a list
    nws=[n for nn in [danubes,dneprs,['dnestr']] for n in nn]
    
    #Get paths and combine data
    rivers=glob.glob(path_prefix)
    rivers.sort()
    data=xr.open_mfdataset(rivers,concat_dim='time',combine='nested').load()

    start=int(''.join(filter(str.isdigit, rivers[0])))
    end=int(''.join(filter(str.isdigit, rivers[-1])))+1
    #debug()
    times=xr.cftime_range(start=str(start),end=str(end),freq='MS')[:-1]

    data['time']=times.to_datetimeindex()
    
    #compute cell area
    area=area_2d(domain)
    
    #Put Data into dict
    final_dic={}
    sum_l=[]
    for k,v in rivers_dict.items():
        d=(data['sorunoff'].isel(lat=v[0],lon=v[1])*area.isel(lat=v[0],lon=v[1]))/1000
        d.attrs['unit']='m³/s'
        sum_l.append(d.rename(k).drop(('lat','lon','time_counter')))
        final_dic[k]=d
    #also do the sum for total river discharge
    sum_rivers=xr.merge(sum_l).to_array(dim='new').sum('new')
    final_dic['all']=sum_rivers
    
    #do north western shelf sum (DDD)
    final_dic['nws']=xr.merge([final_dic[k].drop(('lat','lon')).rename(k) for k in nws]).to_array(dim='new').sum('new')
    
    #do danubes and dniepr sums
    if mode.lower()=='hr':
        final_dic['dniepr']=xr.merge([final_dic[k].drop(('lat','lon')).rename(k) for k in dneprs]).to_array(dim='new').sum('new')
    final_dic['danube']=xr.merge([final_dic[k].drop(('lat','lon')).rename(k) for k in danubes]).to_array(dim='new').sum('new')

    ##To-Do: Add BGC variables
    
    return final_dic


"""
FUNCTIONS TO MASK DATA (DEPTH, LATITUDE, LONGITUDE).
The original shape of data is kept, just masking out values. 

"""

def mask_depth(data,domain,d0=0,d1=100,mode='data',depth_var='deptht'):
    """
    mask all values of all positions where bottom_depth not in [d0,d1] with nan.
    Can be used to focus on the North Western Shelf for instance.

    Input:
        2d/3d/4d data

    
    Output:
        Masked 2d/3d/4d data or mask according to option mode (data/mask)
    """
    if depth_var in data.dims:
        dep=get_bottom(data,domain)
    elif depth_var not in data.coords:
        #add depth from domain file to data (depth_var) and level
        dep=domain[depth_var].values[domain['bottom_level'].values]
        data[depth_var]=(('lat','lon'),dep)
        dep=data[depth_var].to_dataset()
    else:
        dep=data
    depth=mask_data(dep[depth_var],ls_mask)
    mask=xr.where((depth>=d0) & (depth<=d1),1,0)
    mask=mask_data(mask,ls_mask)

    if mode=='data':
        data=xr.where(mask,data,np.nan)
        out=data
    elif mode=='mask':
        out=mask
        
    return out
    
def mask_latlon(data,lats,lons,ls_mask):
    """
    Mask data according to latitude and longitude bounds. The purpose are regional mean computations.
    For plots focussing on a specific region just cut the data to a specific region (e.g. the Black Sea North Western Shelf
    .sel(lat=slice(42.2,47),lon=slice(27.5,34.1))

    Input:
        data: either regular 2D,3D,4D data, or simple 2d mask
        lat=[lat1,lat2] latitudinal bounds of region to keep 
        lon=[lon1,lon2] longitudinal bounds of region to keep
    Output:
        mask (or masked data)
    
    """
    data=data.where(((data.lat>=lats[0]) & (data.lat<=lats[1]) & (data.lon>=lons[0]) & (data.lon<=lons[1])))
    data=mask_data(data,ls_mask)
    return data

def mask_margin(mask):
    """
    TO-DO: Return Margin of a mask (relevant for transport)
    """
    return

def mask_closest_coastline(data):
    """
    TO-DO: The masking in mask_margin can create lines disconnected from the coast. 
    This function adds straight lines to the neares coast
    """
    return
    

def get_bottom(data,domain,depth_var='deptht'):
    """
    Straight forward index selection using bottom level in domain.
    Index needs to be reduced by one
    """
    bot=domain['bottom_level'].drop('time_counter') #bottom level index
    bot=xr.where(bot==0,1,bot) #replace zeroes by one to get the right depth for land positions (else -1, -> last value)
    return data.isel({depth_var:(bot-1)})

def cell_volume(domain,grid='T'):
    """
    Cell volume computed from scaling factors.
    grid='T'/'U'/V'/W'
    output in m^3
    """
    idx=['e1X','e2X','e3X_0']
    idx=[i.replace('X',grid.lower()) for i in idx]
    vol=domain[idx[0]]*domain[idx[1]]*domain[idx[2]]       
    return vol

def area_2d(domain,grid='T'):
    """
    Compute horizontal surface of grid cells.
    We tacitly assume that the x/y sizes are independent of depth (which is usually the case)
    output in m^2
    """
    idx=['e1X','e2X']
    idx=[i.replace('X',grid.lower()) for i in idx]
    area=domain[idx[0]]*domain[idx[1]]
    return area



"""
HELPERS FOR PLOTTING FUNCTIONS
reduce repeating parts
"""

def spatial_stat(d,stats):
    stat_l=[]
    for m in stats:
        if m=='mean': stat=d.mean(('lat','lon'),skipna=True)
        elif m=='std': stat=d.std(('lat','lon'),skipna=True)
        elif m=='median': stat=d.median(('lat','lon'),skipna=True)
        elif m=='min': stat=d.min(('lat','lon'),skipna=True)
        elif m=='max': stat=d.max(('lat','lon'),skipna=True)
        stat_l.append(stat.values)
    stat_l=np.concatenate([stat_l])
    return stat_l





def alignYaxes(axes, align_values=None,nticks=None):
    '''
    Align the ticks of multiple y axes

    By stackoverflow user Jason: https://stackoverflow.com/questions/26752464/how-do-i-align-gridlines-for-two-y-axis-scales

    Modification: 
        - Slightly modified (log2/2**) instead of 10 to get an acceptable range for rivers. Also else I would get negative values.
        - To-Do: Investigate better what this function is actually doing

    Args:
        axes (list): list of axes objects whose yaxis ticks are to be aligned.
    Keyword Args:
        align_values (None or list/tuple): if not None, should be a list/tuple
            of floats with same length as <axes>. Values in <align_values>
            define where the corresponding axes should be aligned up. E.g.
            [0, 100, -22.5] means the 0 in axes[0], 100 in axes[1] and -22.5
            in axes[2] would be aligned up. If None, align (approximately)
            the lowest ticks in all axes.
    Returns:
        new_ticks (list): a list of new ticks for each axis in <axes>.

        A new sets of ticks are computed for each axis in <axes> but with equal
        length.
    '''
    from matplotlib.pyplot import MaxNLocator

    nax=len(axes)
    ticks=[aii.get_yticks() for aii in axes]
    if align_values is None:
        aligns=[ticks[ii][0] for ii in range(nax)]
        #aligns=[np.median(ticks[ii]) for ii in range(nax)]
    else:
        if len(align_values) != nax:
            raise Exception("Length of <axes> doesn't equal that of <align_values>.")
        aligns=align_values

    bounds=[aii.get_ylim() for aii in axes]

    # align at some points
    ticks_align=[ticks[ii]-aligns[ii] for ii in range(nax)]

    # scale the range to 1-100
    ranges=[tii[-1]-tii[0] for tii in ticks]
    #lgs=[-np.log10(rii)+2. for rii in ranges]
    lgs=[-np.log2(rii)+2. for rii in ranges]
    igs=[np.floor(ii) for ii in lgs]
    #log_ticks=[ticks_align[ii]*(10.**igs[ii]) for ii in range(nax)]
    log_ticks=[ticks_align[ii]*(2.**igs[ii]) for ii in range(nax)]

    # put all axes ticks into a single array, then compute new ticks for all
    comb_ticks=np.concatenate(log_ticks)
    comb_ticks.sort()
    steps=[1, 2, 2.5, 3, 4, 5, 8, 10]
    steps=None
    if nticks==None:
        nticks='auto'
        locator=MaxNLocator(nbins=nticks, steps=steps,integer=True)
    else:
        min_ticks=nticks
        locator=MaxNLocator(nbins=nticks, steps=steps,integer=True,min_n_ticks=min_ticks)

    new_ticks=locator.tick_values(comb_ticks[0], comb_ticks[-1])
    #new_ticks=[new_ticks/10.**igs[ii] for ii in range(nax)]
    new_ticks=[new_ticks/2.**igs[ii] for ii in range(nax)]
    new_ticks=[new_ticks[ii]+aligns[ii] for ii in range(nax)]

    # find the lower bound
    idx_l=0
    for i in range(len(new_ticks[0])):
        if any([new_ticks[jj][i] > bounds[jj][0] for jj in range(nax)]):
            idx_l=i-1
            break

    # find the upper bound
    idx_r=0
    for i in range(len(new_ticks[0])):
        if all([new_ticks[jj][i] > bounds[jj][1] for jj in range(nax)]):
            idx_r=i
            break

    # trim tick lists by bounds
    new_ticks=[tii[idx_l:idx_r+1] for tii in new_ticks]

    # set ticks for each axis
    for axii, tii in zip(axes, new_ticks):
        axii.set_yticks(tii)

    return new_ticks



"""
Statistic functions/wrapper.
"""

def temp_resampling_stats(data,time_name='time_counter',averaging='all',mode='mean',diff_modus=None,specific_time=None,mask=None):
    """
    Apply temporal resampling procedure to data. I keep the data as grouped/sampled object and only apply the statistics in a separate function.
    I als compute the statistics directly here. this is not super efficient, but at least, this keeps the code overseeable.

    time_name:
        - time_counter for nemo-bamhbi, time for wam
    
    averaging:
        - the time period for which to do the averaging
    mode:
        - the statistics to compute (mean,std,sum,min,max,quant_5,quant_95,'nan_count','zero_count','nan_count_percentage','zero_count_percentage')
    modus:
        - normal mode just takes the statistics. diff_all substracts the overall statistic (e.g.)mean, diff_all_rel does the same in relative terms.
        - diff substracts statistic for specif time period (e.g. mean from all decembers), diff_rel does the same but computes relative difference (in %)
    """
    
    special_stats_1=['nan_count','zero_count']
    special_stats_2=['nan_count_percentage','zero_count_percentage']
    special_stats=special_stats_1 + special_stats_2
    
    def stats(data,mode):
        """
        data either resampled or grouped, that way the dimension is automatically selected right
        """
        if mode=='mean':
            data=data.mean(time_name,skipna=True)
        elif mode=='median':
            data=data.median(time_name,skipna=True)
        elif mode=='std':
            data=data.std(time_name,skipna=True)
        elif mode=='sum':
            data=data.sum(time_name,skipna=True)
        elif mode=='min':
            data=data.min(time_name,skipna=True)
        elif mode=='max':
            data=data.max(time_name,skipna=True)
        elif mode=='quant_5':
            data=data.quantile(0.05,dim=time_name,skipna=True)
        elif mode=='quant_95':
            data=data.quantile(0.95,dim=time_name,skipna=True)
        return data

    def stats_spec(data,mode,averaging):
        """
        special statistics (nan counting or zero counting. it has to be applied before taking the temporal resampling)
        
        data either resampled or grouped, that way the dimension is automatically selected right
        """
        data=data.transpose(time_name,...) #bring time axis to first dimension to have an easier life...
        #first operation
        if (mode=='nan_count') or (mode=='nan_count_percentage'):
            data=np.isnan(data)
        elif (mode=='zero_count') or (mode=='zero_count_percentage'):
            data=(data==0)

        if averaging=='all':
            data=data
        elif averaging=='seasonal':
            data=data.groupby(data[time_name].dt.season)    
        elif averaging=='seasonal_t':
            data=data.resample({time_name:'QS-DEC'})
        elif averaging=='yearly':
            data=data.resample({time_name:'YS'})
        elif averaging=='monthly':
            data=data.groupby(data[time_name].dt.month)
        elif averaging=='monthly_t':
            data=data.resample({time_name:'MS'})
        
        length=data.count(time_name)
        data=data.sum(time_name)
        
        if mode=='nan_count_percentage' or mode=='zero_count_percentage':
            data=data/length*100
        
        return data
    
    if averaging=='all':
        if mode in special_stats:
            mean=stats_spec(data,mode,averaging)
        else:
            mean=stats(data,mode)
        mean=mean.assign_attrs({time_name:['Average']})
        if (diff_modus=='diff') or (diff_modus=='diff_rel'):
            mean=xr.full_like(mean,np.nan)
    elif averaging=='seasonal':
        if mode in special_stats:
            mean=stats_spec(data,mode,averaging)
        else:
            mean=stats(data.groupby(data[time_name].dt.season),mode)#.mean()
        mean=mean.sortby(xr.DataArray(['DJF','MAM','JJA', 'SON'],dims=['season'])) #put that into right order
        mean=mean.rename({'season':time_name})
    elif averaging=='seasonal_t':
        if mode in special_stats:
            mean=stats_spec(data,mode,averaging)
        else:
            mean=stats(data.resample({time_name:'QS-DEC'}),mode)#.mean()        
        if diff_modus=='diff':
            if mode in special_stats:
                mean=stats_spec(data,mode,averaging)
            else:
                seasons=stats(data.groupby(data[time_name].dt.season),mode)
            for i,t in enumerate(mean[time_name]):
                s=t.dt.season.values
                mean.loc[{time_name:t}]=mean.isel({time_name:i})-seasons.sel(season=s).values

        elif diff_modus=='diff_rel':
            if mode in special_stats:
                mean=stats_spec(data,mode,averaging)
            else:
                seasons=stats(data.groupby(data[time_name].dt.season),mode)
            for i,t in enumerate(mean[time_name]):
                s=t.dt.season.values
                mean.loc[{time_name:t}]=(mean.isel({time_name:i})-seasons.sel(season=s).values)/seasons.sel(season=s).values*100
            
    elif averaging=='yearly':
        if mode in special_stats:
            mean=stats_spec(data,mode,averaging)
        else:    
            mean=stats(data.resample({time_name:'YS'}),mode)
        label=mean[time_name].dt.year.values

        if diff_modus=='diff':            
            alls=stats(data,mode)
            for i,t in enumerate(mean[time_name]):
                #import pdb
                #pdb.set_trace()
                mean.loc[{time_name:t}]=mean.isel({time_name:i})-alls.values

        elif diff_modus=='diff_rel':
            alls=stats(data,mode)
            for i,t in enumerate(mean[time_name]):
                #import pdb
                #pdb.set_trace()
                mean.loc[{time_name:t}]=(mean.isel({time_name:i})-alls.values)/alls.values*100
    
    elif averaging=='monthly':
        if mode in special_stats:
            mean=stats_spec(data,mode,averaging)
        else:    
            mean=stats(data.groupby(data[time_name].dt.month),mode)#.mean()
        label=mean['month'].values
        mean=mean.rename({'month':time_name})
    elif averaging=='monthly_t':
        if mode in special_stats:
            mean=stats_spec(data,mode,averaging)
        else:    
            mean=stats(data.resample({time_name:'MS'}),mode)#.mean()
        months=mean['time_counter'].dt.month
        years=mean['time_counter'].dt.year
        label=[str(m.values)+'-'+str(months[i].values) for i,m in enumerate(years)]

        if diff_modus=='diff':
            if mode in special_stats:
                mean=stats_spec(data,mode,averaging)
            else:    
                monthly=stats(data.groupby(data[time_name].dt.month),mode)
            for i,t in enumerate(mean[time_name]):
                s=t.dt.month.values
                mean.loc[{time_name:t}]=mean.isel({time_name:i})-monthly.sel(month=s).values
        
        elif diff_modus=='diff_rel':
            if mode in special_stats:
                monthly=stats_spec(data,mode,averaging)
            else:    
                monthly=stats(data.groupby(data[time_name].dt.month),mode)
            for i,t in enumerate(mean[time_name]):
                s=t.dt.month.values
                mean.loc[{time_name:t}]=(mean.isel({time_name:i})-monthly.sel(month=s).values)/monthly.sel(month=s).values*100

    else:
        raise NameError("Temporal mode unknown. Select from 'all','seasonal','seasonal_t','yearly','monthly','monthly_t")
    
    
    if diff_modus=='diff_all':
        if averaging=='all':
            mean=xr.full_like(mean,np.nan)
        else:
            if mode in special_stats:
                vals=special_stats(data,mode,'all')#.values
            else:
                vals=stats(data,mode)
            mean=mean-vals
    elif diff_modus=='diff_all_rel':
        if averaging=='all':
            mean=xr.full_like(mean,np.nan)
        else:
            if mode in special_stats:
                s=special_stats(data,mode,'all')
            else:
                s=stats(data,mode)
            mean=(mean-s)/s*100
    elif diff_modus=='diff_rel':
        pass
    elif diff_modus=='diff':
        pass
    elif diff_modus==None:
        pass
    else:
        raise NameError("Difference modus unknown")

    #select specific time after the difference computation !
    if time_name in mean.dims:        
        if isinstance(specific_time,int):
            mean=mean.isel({time_name:specific_time})
            lab_t=True
        elif isinstance(specific_time,str):
            mean=mean.sel({time_name:specific_time},method='nearest')
            lab_t=True
        else:
            lab_t=False
    else:
       lab_t=False
        
    #add labels after time selection!
    if averaging=='seasonal_t':
        seasons=mean[time_name].dt.season
        years=mean[time_name].dt.year
        label=[str(m.values)+'-'+str(seasons[i].values) for i,m in enumerate(years)]
    elif averaging=='yearly':
        label=mean[time_name].dt.year.values
    elif averaging=='monthly_t':
        months=mean['time_counter'].dt.month
        years=mean['time_counter'].dt.year
        label=[str(m.values)+'-'+str(months[i].values) for i,m in enumerate(years)]
    elif averaging=='seasonal' or averaging=='monthly':
        label=mean[time_name].values
    if averaging=='all':
        label=['Average']

    #trigger to avoid 0-dim label
    if lab_t: label=[label.item()]
    
    mean=mean.assign_attrs({'time_label':label})    

    if mode in special_stats:
        if 'ls_mask' in globals():
            mean=mask_data(mean,ls_mask)
    return mean

def spatial_mean(data,area=None,domain=None,depth_var='deptht',grid='T'):
    """
    Spatial mean both 2d and 3d data
    """
    if depth_var in data.dims:
        mean_ax=('lat','lon',depth_var)
        if area is not None:
            if depth_var not in area.dims:
                raise TypeError('Area needs to contain depth')
    else:
        mean_ax=('lat','lon')
    
    if domain is not None:
        #compute area

        if depth_var in data.dims:
            area=cell_volume(domain,grid=grid)
        else:
            area=area_2d(domain,grid=grid)
        
    if area is not None:
        data=data.weighted(area).mean(mean_ax,skipna=True)

    else:
        data=data.mean(mean_ax,skipna=True)
        
    return data


def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT. Usual implementations in scipy don't include the periodic options.

    x and y must be real sequences with the same length.

    By Stack overflow user Warren Wackesser: https://stackoverflow.com/questions/28284257/circular-cross-correlation-python
    
    See also: https://www.ocean.washington.edu/courses/ess522/lectures/08_xcorr.pdf
    """
    cov=np.fft.ifft(np.fft.fft(x).conj() * np.fft.fft(y)).real
    denom=np.sqrt(np.fft.fft(x*x)[0]*np.fft.fft(y*y)[0])
    
    corr=(cov/denom).real
    lag=np.argmax(corr)
    max_corr=corr[lag]
    
    return lag,max_corr
    
"""
PLOTTING FUNCTIONS

long list of arguments due to some important options to improve look of plot. for first impressions you can throw away most of them and use default config.

PLOT LIST.

SPATIAL MAP PLOTS

0. Show mask to visualize how you masked your data.
1. Single variable with multiple statistics in time
2. Multiple variables in Time
3. Compare 2 Model runs for a variable in time
4. Compare 2 Model runs for multipl

TIME SERIES PLOTS

1. 2D/3D Spatial average im time (mask data before to get a specific region) of multiple variables.
Includes options to plot river outflow, compare different model runs and variables visually and also quantitatively via (cross) correlations.

"""

def show_mask(da,time_name='time_counter',ls_mask=None,domain=None,title='',depth_var='deptht',ret=False,cartopy=True,cbar_kwargs={'label': 'mask','shrink':0.5},cmap='GnBu',
             lat_step=2,lon_step=4,bathymetry=True):

    if depth_var in da.dims:
        #just select first level ...
        da=da.isel({depth_var:0})
    
    if type(da)==xr.core.dataset.Dataset:
        some_var=list(da.data_vars)[0]
        da=da[some_var]
    elif type(da)==xr.core.dataarray.DataArray:
        pass
    else:
        raise TypeError('please provide xarray object for <da>')
    if ls_mask is None:
        raise TypeError('provide land sea mask "ls_mask"')
    if time_name in da.dims: 
        is_nan_axis = ~np.all(np.isnan(da.transpose(time_name,...)), axis=0)
    else: 
        is_nan_axis = ~np.isnan(da)

    if cartopy:
        #mask and show rivers
        projection=ccrs.PlateCarree()
        fig,ax=plt.subplots(subplot_kw={'projection':projection})
        is_nan_axis=mask_data(is_nan_axis,ls_mask)
        is_nan_axis.plot(ax=ax,cbar_kwargs=cbar_kwargs,cmap=cmap,vmin=0)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.RIVERS,edgecolor='blue',linewidth=1.5)
        #ax.add_feature(cfeature.COASTLINE,linewidth=1.5 # coatline can look a bit weird because it doesn't match
        ax.set_title(title)

        gl = ax.gridlines(crs=projection, draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False; gl.ylabels_right = False
    
        #FIX GRIDLINE SPACING
        min_lat=np.round(da['lat'].min().values); max_lat=np.round(da['lat'].max().values)
        min_lon=np.round(da['lon'].min().values); max_lon=np.round(da['lon'].max().values)
        lats=np.arange(min_lat,max_lat+lat_step,lat_step); lons=np.arange(min_lon,max_lon+lon_step,lon_step)
    
        gl.xlocator = mticker.FixedLocator(lons); gl.xformatter = LONGITUDE_FORMATTER;
        gl.ylocator = mticker.FixedLocator(lats); gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}

        if bathymetry:
            cmap = mpl.cm.GnBu
            bounds=[0,50,100,500,1000,2000]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            CS = domain['bathy_metry'].plot.contour(kwargs=dict(inline=True),levels=bounds,
                                                    ax=ax,cmap=cmap,norm=norm,add_colorbar=False,alpha=0.9)
            ax.clabel(CS,colors='dimgray')
            #ax.clabel(CS)
        ax.set_title(title)
    
    else:
        is_nan_axis.plot(cbar_kwargs=cbar_kwargs,cmap=cmap,vmin=0)
        plt.title(title)
    
    if ret==True:
        ret=is_nan_axis
    else:
        ret=None

    #if bathymetry
    
    return ret

def plot1_singlevar(data,
                    avgs=['all','seasonal','seasonal_t','yearly','monthly','monthly_t'],
                    stats=['mean','sum','std','min','max','quant_5','quant_95','nan_count','zero_count','nan_count_percentage','zero_count_percentage'],
                    diff_modus=None, #can also be diff, diff_rel, diff_all, diff_all,rel #repeated for each stats entry.
                    #this way we can display normal mean, NOTE THAT THIS DOES NOT WORK when 'all' in avgs.
                    specific_time=None, #can be used to reduce temporal data to a specific time (e.g. a specific year and season for seasonal_t or for monthly_t'
                    #therefore provide a time string (e.g. "2012-02") and closest moment is selected. the selection is done after time resampling/subtracting time means
                    #else one could just restrict the data to the start. for selection after seasonal resampling remember to select the startmonth (e.g. december if you want a djf average, february would give you MAM because time has been resampled
                    time_name='time_counter',
                    cmap='Reds',
                    vmax=None,vmin=None,
                    title='Bottom Oxygen',
                    height=1.8, width=5.2, y=1, labelsize=None, titlesize=None, left_pos=-0.5,
                    grid_kwargs={'on':True,'top':False,'right':False,'bottom':True,'left':True,
    'draw_labels':True, 'lon_step':4,'lat_step':2,'size':10},
                    cartopy_kwargs={'land':True, 'coastline': False,'rivers': True,'linewidth': 0.5,},
                    colorbar_kwargs={ 'levels':11, #levels in the plot every second level has a plot
                    'nbins':5,  #only show every 'nth label. 
                    'shrink':0.9, 'pad':0.02},
                    statsbox_kwargs={'use':False,'stats':['mean','std'],'pos':[0.75,0.95],'prec': '.1f',#precision when printing numbers
                                     'fontsize':8},#'stats':['mean','std','median','min','max'
                    unit=None
                    ):
    """
    Plot panel:
        x-axis: statistics for single variable
        y-axis: time
        
    To select: One input variable
    """
    ##### DATA PREPARATION  ####
    if diff_modus is None: modus=[None for _ in range(len(stats))]
    if type(diff_modus)!=list: diff_modus=[diff_modus]
    if len(diff_modus)!=len(stats): diff_modus=[diff_modus[0] for _ in range(len(stats))]
    
    #make a list if not a list
    if type(avgs)!=list: avgs=[avgs]
    if type(stats)!=list: stats=[stats]

    time_labels=[];all_data=[]
    #store data in a double layered list/dictionary first layer is the statistics, second is the type of statistic

    for i,s in enumerate(stats):
        da={}
        for ii,a in enumerate(avgs):
            res=temp_resampling_stats(data,averaging=a,mode=s,diff_modus=diff_modus[i],time_name=time_name,specific_time=specific_time)
            if i==0: time_labels.append(res.attrs['time_label'])
            da[a]=res
        all_data.append(da)
    #unpack list of time labels. lentgth needed for setting up plot
    #try:
    #    #unpack list of time labels. lentgth needed for setting up plot
    time_labels=[tt for t in time_labels for tt in t]
    #except:
    #time_labels=[time_labels]
    
    
    ##### PLOTTING #####
    ck=SimpleNamespace(**colorbar_kwargs)
    sb_kw=SimpleNamespace(**statsbox_kwargs)
    rows=len(time_labels); cols=len(stats)

    projection=ccrs.PlateCarree()
    fig,ax=plt.subplots(ncols=cols,nrows=rows,figsize=(cols*width,rows*height),subplot_kw={'projection':projection})
    da_stats_avg=plot_data(ax,all_data,avgs, stats,rows, cmap, time_name, sb_kw, ck, diff_modus, fig, labelsize, title, titlesize, y,vmax,vmin,unit=unit)
    #import pdb; pdb.set_trace()
    ##### PLOT IMPROVEMENTS #####
    plot_with_grid(ax, stats, diff_modus, time_labels, grid_kwargs, projection, data, cartopy_kwargs, sb_kw, da_stats_avg, rows, cols, left_pos, labelsize)
    fig.suptitle(title,fontsize=titlesize+5,y=y,fontweight='bold')
    fig.tight_layout()
    return fig

def plot2_multivar(data,variabs=['DOX','POC'],
                    avgs=['all','seasonal'],
                    #avgs=['all','seasonal','seasonal_t','yearly','monthly','monthly_t'],
                    #stats=['mean','sum','std','min','max','quant_5','quant_95'],
                    stats='mean',
                    diff_modus=None, #can also be diff, diff_rel, diff_all, diff_all,rel #repeated for each stats entry.
                    #this way we can display normal mean, 
                    specific_time=None, #can be used to reduce temporal data to a specific time (e.g. a specific year and season for seasonal_t or for monthly_t'
                    #therefore provide a time string (e.g. "2012-02") and closest moment is selected. the selection is done after time resampling/subtracting time means
                    #else one could just restrict the data to the start. for selection after seasonal resampling remember to select the startmonth (e.g. december if you want a djf average, february would give you MAM because time has been resampled
                    time_name='time_counter',
                    cmap='Reds', 
                    vmax=None,vmin=None,
                    title='',
                    height=1.8, width=5.2, y=1, labelsize=None, titlesize=None, left_pos=-0.5,
                    grid_kwargs={'on':True,'top':False,'right':False,'bottom':True,'left':True, 'draw_labels':True, 'lon_step':4,'lat_step':2,'size':10},
                    cartopy_kwargs={'land':True, 'coastline': False,'rivers': True,'linewidth': 0.5,},
                    colorbar_kwargs={ 'levels':11, #levels in the plot every second level has a plot
                    'nbins':5,  #only show every 'nth label. 
                    'shrink':0.9, 'pad':0.02},
                    statsbox_kwargs={'use':True,'stats':['mean','std'],'pos':[0.75,0.95],'prec': '.1f',#precision when printing numbers
                                     'fontsize':8},#'stats':['mean','std','median','min','max'
                   unit=None
                    ):
    """
    Plot panel:
        x-axis: variables
        y-axis: time

    To set: The statistics you want to have (e.g. mean) 
    """
    ##### DATA PREPARATION  ####
    
    if diff_modus is None: modus=[None for _ in range(len(stats))]
    if type(diff_modus)!=list: diff_modus=[diff_modus]
    if len(diff_modus)!=len(variabs): diff_modus=[diff_modus[0] for _ in range(len(variabs))]
    
    #make a list if not a list
    if type(avgs)!=list: avgs=[avgs]

    time_labels=[];all_data=[]
    #store data in a double layered list/dictionary first layer is the statistics, second is the type of statistic

    for i,v in enumerate(variabs):
        da={}
        for ii,a in enumerate(avgs):
            res=temp_resampling_stats(data[v],averaging=a,mode=stats,diff_modus=diff_modus[i],time_name=time_name,specific_time=specific_time)
            if i==0: time_labels.append(res.attrs['time_label'])
            da[a]=res
        all_data.append(da)
    #unpack list of time labels. lentgth needed for setting up plot
    #try:
    #unpack list of time labels. lentgth needed for setting up plot
    time_labels=[tt for t in time_labels for tt in t]
    #except: time_labels=[time_labels]

    ##### PLOTTING #####
    ck=SimpleNamespace(**colorbar_kwargs)
    sb_kw=SimpleNamespace(**statsbox_kwargs)
    rows=len(time_labels); cols=len(variabs)
    
    projection=ccrs.PlateCarree()
    fig,ax=plt.subplots(ncols=cols,nrows=rows,figsize=(cols*width,rows*height),subplot_kw={'projection':projection})
    da_stats_avg=plot_data(ax,all_data,avgs, variabs, rows,cmap, time_name, sb_kw, ck, diff_modus, fig, labelsize, title, titlesize, y,vmax,vmin,unit=unit)
    
    ##### PLOT IMPROVEMENTS #####
    plot_with_grid(ax, variabs, diff_modus, time_labels, grid_kwargs, projection, data, cartopy_kwargs, sb_kw, da_stats_avg, rows, cols, left_pos, labelsize)
    fig.tight_layout()
    return fig




def plot4_cf_runs(data1,data2,
    variabs=['DOX','POC'], 
    data_names=['data1','data2'],
    comp_mode=[None],#diff_abs,'diff_rel' #comparison mode
    avg='all',#['all','seasonal','seasonal_t','yearly','monthly','monthly_t'],
    stats='mean',
    diff_modus=None, #can also be diff, diff_rel, diff_all, diff_all,rel #repeated for each data entry entry.
    #this way we can display normal mean, NOTE THAT THIS DOES NOT WORK when 'all' in avgs.
    specific_time=0, #has to be used to reduce temporal data to a specific time (e.g. a specific year and season for seasonal_t or for monthly_t')
    #either provide an index or a specific string (e.g. "2012-02") and closest moment is selected. the selection is done after time resampling/subtracting time means
    #else one could just restrict the data to the start. for selection after seasonal resampling remember to select the startmonth (e.g. december if you want a djf average, february would give you MAM because time has been resampled
    time_name='time_counter',
    cmap='Reds', #can also be length of data ['viridis','viridis']
    cmap_diffs='RdBu_r',
    vmax=None,vmin=None,
    title='Bottom Oxygen',
    height=3, width=6, y=1, labelsize=None, titlesize=None, left_pos=-0.5,
    grid_kwargs={'on':True,'top':False,'right':False,'bottom':True,'left':True, 'draw_labels':True, 'lon_step':4,'lat_step':2,'size':10},
    cartopy_kwargs={'land':True, 'coastline': False,'rivers': True,'linewidth': 0.5,},
    colorbar_kwargs={ 'levels':11, #levels in the plot every second level has a plot
    'nbins':5,  #only show every 'nth label. 
    'shrink':0.9, 'pad':0.02},
    statsbox_kwargs={'use':False,'stats':['mean','std'],'pos':[0.75,0.95],'prec': '.1f',#precision when printing numbers
                     'fontsize':8},#'stats':['mean','std','median','min','max'    
    unit=None
):
    """
    Plot panel:
        x-axis: Data from two runs (+absolute difference or relative difference)
        y-axis: variabs

    To set:
        - time_resampling, time, statistic
    """
    ##### DATA PREPARATION  ####
    if type(comp_mode)!=list: comp_mode=[comp_mode]
    if type(diff_modus)!=list: diff_modus=[diff_modus for _ in range(len(variabs))]

    time_labels=[];all_data=[]
    #store data in a double layered list/dictionary first layer is the statistics, second is the type of statistic
    
    for i,data in enumerate([data1,data2]):
        da={}
        for ii,a in enumerate(variabs):
            res=temp_resampling_stats(data[a],averaging=avg,mode=stats,diff_modus=diff_modus[i],time_name=time_name,specific_time=specific_time)        
            if i==0: time_labels.append(res.attrs['time_label'])
            da[a]=res
        all_data.append(da)

    #do the differences
    for i,m in enumerate(comp_mode):
        if m=='diff_abs': data_names.append('Difference (Abs.)')
        elif m=='diff_rel': data_names.append('Difference (Rel.)')
        else:pass
            
        da={}
        for ii,a in enumerate(variabs):
            empty=[]
            da0=all_data[0][a]
            da1=all_data[1][a]
            for ii, t in enumerate(da0.attrs['time_label']):
                if time_name in da0.dims: d0 = da0.isel({time_name: ii}); d1 =da1.isel({time_name: ii})
                else: d0 = da0; d1 = da1
                if m=='diff_abs': empty.append(d0-d1.values)
                elif m=='diff_rel': empty.append((d0-d1)/d0*100)
                else: passb/workspaces/auto-t
            da[a]=empty
        all_data.append(da)
    

    ##### PLOTTING #####
    ck=SimpleNamespace(**colorbar_kwargs)
    sb_kw=SimpleNamespace(**statsbox_kwargs)
    rows=len(time_labels);
    if None not in comp_mode:
        cols=2+len(comp_mode)
    else:
        cols=2
    
  
    fig,ax=plt.subplots(ncols=cols,nrows=rows,figsize=(cols*width,rows*height),subplot_kw={'projection':projection})
    
    da_stats_avg=plot_data(ax,all_data,variabs, [data1,data2], rows,cmap, time_name, sb_kw, ck, diff_modus, fig, labelsize, title, titlesize, y,vmax,vmin,same_cbar=True)

    if cols>2:
        #plot difference
        da_stats_avg2=plot_data(ax,all_data,variabs, [data1,data2], rows,cmap_diffs, time_name, sb_kw, ck, diff_modus, fig, labelsize, title, titlesize, y,vmax,vmin,x_offset=len(comp_mode),diff_modus2=comp_mode,unit=unit)
        #plot_data(ax,all_data,avgs, [data1,data2], rows,cmap, time_name, sb_kw, ck, diff_modus, fig, labelsize, title, titlesize, y,vmax,vmin)
        da_stats_avg=np.concatenate([da_stats_avg,da_stats_avg2])
    ##### PLOT IMPROVEMENTS #####
    plot_with_grid(ax, data_names, diff_modus, variabs, grid_kwargs, projection, data, cartopy_kwargs, sb_kw, da_stats_avg, rows, cols, left_pos, labelsize)
    
    return


def plot_spatial_mean(variables=['DOX','PHO'],
                    data1=None,data2=None,data3=None,
                    labels=['data1','data2','data3'],
                    area=None,domain=None,
                    second_vars=['PHO','DOX'], #will be plotted along first var in the same plot (second y axis)
                    leg_kw={'loc':'best','ncols':3,'fs':8},
                    river_discharge={'data':None,'names':['all'],'first':False,'interpol':'D','ax_pos':1.2}, #interpol keyword needed for having same temporal resolution
                    num_ticks=None,
                    fig_width=10,
                    fig_height=3,
                    grid='T',
                    sharex=False, #if xaxis of subplots shared
                    depth_var='deptht',
                    time_name='time_counter',
                    correlation=None,
                    cross_correlation=None,
                    time_format=None,
                    title='',
                    titlesize=None,y=1,
                    ):

                    #river_discharge=None
                    #correlation={'calc':['var1-var2','var-riv','data1-data2'],'leg2pos':[1.4,0.95],'leg2_fs':8,'mode':'corr'}
                    #time_format={'loc':md.MonthLocator(range(1,13,3), bymonthday = 1),'fmt':md.DateFormatter('%Y-%m')}
                
    """
    Data going in:
        - Important prestep is masking the data, such that only the region of relvance for you is kept
    
    
    
    Option:
        1. Second variable alongside (To-Do)
    
            Important tweak: the gridlines are aligned to match each other (else the figure is visual crap)
        
        2. Add river discharge alongside (To-Do)
        3. Modify spacing of time axis labels (To-Do)
        4. Calculate correlations: between different variables (1), correlation between runs (2), correlation to river discharge (3) (To-Do)
        5. Low pass Filter time (To-Do)
    """
    data1=spatial_mean(data1,area=area,domain=domain,depth_var=depth_var,grid=grid)
    
    if data2 is not None:
        assert(list(data1.dims)==list(data2.dims),f"data1 has dims {data1.dims}, but data2 has dims {data2.dims}")
        data2=spatial_mean(data2,area=area,domain=domain,depth_var=depth_var,grid=grid)

    if data3 is not None:
        assert(list(data1.dims)==list(data3.dims),f"data1 has dims {data1.dims}, but data3 has dims {data3.dims}")
        data3=spatial_mean(data3,area=area,domain=domain,depth_var=depth_var,grid=grid)
    
    fig,axes=plt.subplots(nrows=len(variables),ncols=1,figsize=(fig_width,fig_height*len(variables)),sharex=sharex)
    
    fig.suptitle(title,fontsize=titlesize,y=y,fontweight='bold')
    ax1=None;ax2=None;ax3=None #initialize
    
    #loop over variables
    data=[data for data in [data1,data2,data3] if data!=None]
    
    if second_vars is not None: num=len(data)*2
    else: num=len(data)
    if river_discharge is not None:
        if isinstance(river_discharge['names'],list):
            if len(river_discharge['names'])>1: num=num+len(river_discharge['names'])
    
    #color palette
    colors=sns.color_palette('colorblind',num)
    
    for i,v in enumerate(variables):
        if len(variables)>1: ax1=axes[i]
        else: ax1=axes
    
        #save lines to get a unified legend for both y axis
        lines=[]
        
        for j,d in enumerate(data):
            if len(data)>1: label=v+' ('+labels[j]+')'
            else: label=v
            p=d[v].plot(ax=ax1,label=label,c=colors[j])
            #if isinstance(p,list):
            p=p[0]
            lines.append(p)
        if 'units' in d[v].attrs: lab=d[v].attrs['units']
        else: lab=None
        #ax.set_ylabel(v+' ['+lab+']')
        ax1.set_ylabel(lab)
        ax1.set_title(v)
        ax1.set_xlabel('')
        
        #add second variable
        if second_vars is not None:
            v2=second_vars[i]
            if v2 is not None:
                ax2 = ax1.twinx()
                for j,d in enumerate(data):
                    if len(data)>1: label=v2+' ('+labels[j]+')'
                    else: label=v2            
                    p=d[v2].plot(ax=ax2,ls='--',c=colors[j+len(data)],label=label)
                    lines.append(p[0])
                if 'units' in d[v2].attrs:
                    lab=d[v2].attrs['units']
                else: lab=None
                ax2.set_ylabel(v2+' ['+lab+']')
                ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
                ax1.set_ylabel(v+' ['+lab+']')
                ax1.set_title('')       
    
        #add river discharge (option just on first plot)
        #river discharge will be black and dotted
        if river_discharge is not None:
            if ((river_discharge['first']==False) or (i==0)):
                ax3 = ax1.twinx()
                if ax2 is not None:
                    ax3.spines['right'].set_position(('axes',river_discharge['ax_pos']))
                    #displace the second y axis to have them distinguishable!
                
                rivers=river_discharge['data']
                names=river_discharge['names']
                if not isinstance(names,list):
                    names=[names]
                if len(names)==1: colors_r=['black']
                else: colors_r=colors[-len(names):None]
                for ii,k in enumerate(names): 
                    #NOTE THAT RIVER DATA HAS TO BE INTERPOLATED TO DAILY RESOLUTION for correlation calculation
                    #limit river data in time. subtract/add a day to have the outer values
                    river_d=rivers[k].sel(time=slice(d[v][time_name][0]- np.timedelta64(1, 'D'),d[v][time_name][-1] + np.timedelta64(1, 'D')))
                    if k=='all': k='all rivers'
                    elif k=='nws': k='NWS rivers'
                    else: pass
                    lines.append(river_d.plot(ax=ax3,ls=':',c=colors_r[ii],label=k,marker='*')[0])
                ax3.set_ylabel(river_d.unit)
                if (second_vars is not None):
                    if (v2 is not None):
                        ax3.set_title('')
                else:
                    ax3.set_title(v)
                    ax3.set_ylabel('['+lab+']')
                ax3.grid(False)    
        
        all_axes=[ax for ax in [ax1,ax2,ax3] if ax!=None]
        alignYaxes(all_axes, align_values=None,nticks=num_ticks)

        
        if time_format!=None:
            loc=time_format['loc']
            fmt=time_format['fmt']
            ax1.xaxis.set_major_locator(loc)
            ax1.xaxis.set_major_formatter(fmt)
        
        # added these three lines
        labs = [l.get_label() for l in lines]
        #choose the last axes object of twinx for the legend to be sure that framealpha=1 is respected (no lines plotted over legend)
        if ax3==None and ax2==None: leg_ax=ax1
        elif ax3==None and ax2!=None: leg_ax=ax2
        else: leg_ax=ax3
        leg_ax.legend(lines, labs, loc=leg_kw['loc'],ncols=leg_kw['ncols'],fontsize=leg_kw['fs'],facecolor='white', framealpha=1)      
        
        #calculate correlation     
        if correlation is not None:
            corr=SimpleNamespace(**correlation)
            correl_dic={}
            if not isinstance(corr.calc,list): corr.calc=[corr.calc]
            for c in corr.calc:
                for j,d in enumerate(data):
                    if c=='var1-var2':
                        if v2 is not None:
                            r=xr.corr(d[v],d[v2],dim=time_name).values.item()
                            sup=''
                            if len(data)>1: sup=' ('+labels[j]+')'
                            correl_dic[(v+'-'+v2+sup)]=r
                    elif c=='var-riv':
                        if river_discharge is None:
                            raise TypeError("Provide river discharge info! It's missing")
                        riv_all_d=rivers[names[0]].resample(time='D').interpolate('linear')
                        riv_all_d=riv_all_d.sel({'time':d[time_name]},method='nearest')
                        r=xr.corr(d[v],riv_all_d).values.item()
                        sup=''
                        if len(data)>1: sup=' ('+labels[j]+')'
                        correl_dic[(v+'-RD'+sup)]=r
                        
                        if v2 is not None:
                            r=xr.corr(d[v2],riv_all_d).values.item()
                            correl_dic[(v2+'-RD'+sup)]=r
                    elif c=='data1-data2':
                        break
                    else:
                        raise TypeError(f"Correlation mode '{c}' not known. Must be 'var1-var2','var-riv' or 'data1-data2'")  
                    
                if c=='data1-data2':
                    r=xr.corr(data1[v],data2[v],dim=time_name).values.item()
                    correl_dic[(v+' ('+labels[0]+'-'+labels[1]+')')]=r
                    if second_vars!=None:
                        v2=second_vars[i]
                        if v2 is not None:
                            correl_dic[(v2+' ('+labels[0]+'-'+labels[1]+')')]=xr.corr(data1[v2],data2[v2],dim=time_name).values.item()
            #add correl_dic text  
            textstr = '\n'.join(['Correlation:']+[f"{iii} = {x:{'.2f'}}" for iii, x in correl_dic.items()])

            props = dict(boxstyle='round', facecolor='white',alpha=1)
            newlines=textstr.count('\n')        
            leg_ax.text(corr.leg2pos[0],corr.leg2pos[1], textstr, transform=leg_ax.transAxes, fontsize=corr.leg2_fs,verticalalignment='top', bbox=props)

        #calculate correlation     
        if cross_correlation is not None:
            corr=SimpleNamespace(**cross_correlation)
            correl_dic={}
            if not isinstance(corr.calc,list): corr.calc=[corr.calc]
            for c in corr.calc:
                for j,d in enumerate(data):
                    if c=='var1-var2':
                        if v2 is not None:
                            (lag,r)=periodic_corr(d[v].values,d[v2].values)
                            sup=''
                            if len(data)>1: sup=' ('+labels[j]+')'
                            correl_dic[(v+'-'+v2+sup)]=r
                    elif c=='var-riv':
                        if river_discharge is None:
                            raise TypeError("Provide river discharge info! It's missing")
                        riv_all_d=rivers[names[0]].resample(time='D').interpolate('linear')
                        riv_all_d=riv_all_d.sel({'time':d[time_name]},method='nearest')
                        (lag,r)=periodic_corr(d[v].values,riv_all_d.values)
                        sup=''
                        if len(data)>1: sup=' ('+labels[j]+')'
                        correl_dic[(v+'-RD'+sup)]=str(np.round(r,2))+', '+str(lag)
                        
                        if v2 is not None:
                            (lag,r)=periodic_corr(d[v2].values,riv_all_d.values)
                            correl_dic[(v2+'-RD'+sup)]=str(np.round(r,2))+', '+str(lag)
                    elif c=='data1-data2':
                        break
                    else:
                        raise TypeError(f"Cross correlation mode '{c}' not known. Must be 'var1-var2','var-riv' or 'data1-data2'")  
                    
                if c=='data1-data2':
                    (lag,r)=periodic_corr(data1[v].values,data2[v].values)
                    correl_dic[(v+' ('+labels[0]+'-'+labels[1]+')')]=str(np.round(r,2))+','+str(lag)
                    if second_vars!=None:
                        v2=second_vars[i]
                        if v2 is not None:
                            (lag,r)=periodic_corr(data1[v2].values,data2[v2].values)
                            correl_dic[(v2+' ('+labels[0]+'-'+labels[1]+')')]=str(np.round(r,2))+','+str(lag)
            #add correl_dic text  
            textstr = '\n'.join(['Max. Cross Corr. (r,lag) :']+[f"{iii} = {x}" for iii, x in correl_dic.items()])

            props = dict(boxstyle='round', facecolor='white',alpha=1)
            newlines=textstr.count('\n')        
            leg_ax.text(corr.leg2pos[0],corr.leg2pos[1], textstr, transform=leg_ax.transAxes, fontsize=corr.leg2_fs,verticalalignment='top', bbox=props)

        
        #reset axes
        ax1=None;ax2=None;ax3=None;
    
    fig.tight_layout()

    return fig


def plot_data(ax,all_data, y_axis, x_axis,rows, cmap, time_name, sb_kw, ck, diff_modus, fig, labelsize, title, titlesize, y,vmax_i,vmin_i,
              x_offset=0,diff_modus2=None,unit=None,same_cbar=False):

        """
        x_offset meaning: needed for difference plots in plot3/plot4, where plot_data is
        called once for regular plots, and then for diffplots. index for accesing subplots
        needs to continue running, x_offset and iter[0]/iter[1] is mainly about that.
        """
        idx = 0
        da_stats_avg = [[] for _ in range(len(x_axis))]
    
        for ii, a in enumerate(y_axis):
            if x_offset==0:
                iteri=[0,len(x_axis)]
            else:
                iteri=[len(x_axis),len(x_axis)+len(diff_modus2)]
            
            if same_cbar:
                #to enforce same colorbar for plot 3/4
                d0=all_data[0][a]
                d1=all_data[1][a]
                #make sure right order
                d1=d1.transpose(*d0.dims)
                data_=np.concatenate([d0,d1])
                
            for j in range(iteri[0],iteri[1]):
                if isinstance(cmap, list):
                    if j-x_offset>=len(cmap): cm = cmap[-1] #in case user forgot to make colormap list long enough repeat last entry
                    else: cm = cmap[j-x_offset]
                else: cm = cmap
                data = all_data[j][a]
                if (not same_cbar): # or (same_cbar and j>=2):
                    data_=data
                if vmax_i is None: vmax = np.nanpercentile(data_, 95)
                else: vmax=vmax_i
                if vmin_i is None: vmin = np.nanpercentile(data_, 5)
                else: vmin=vmin_i
                if (vmax==vmin) or np.isnan(vmax) or np.isnan(vmin):
    
                    if 'count_percentage' in x_axis[j-x_offset]:
                        vmax=100; vmin=0
                    elif 'count' in x_axis[j-x_offset]:
                        vmax=1; vmin=0
                    else:
                        vmax=1;vmin=-1

                #enforce symmetrc colorbar for difference plots
                if diff_modus2 is not None:
                    if vmax>0:
                        if vmax>np.abs(vmin):
                            vmin=-vmax
                        else:
                            if vmin<0:
                                vmax=-vmin
                    else:
                        vmax=-vmin
                
                    
                jj = idx
                
                if isinstance(data,xr.core.dataarray.DataArray):
                    if time_name in data.dims:
                       rang=len(data[time_name])
                    else:
                       rang=1
                else:
                    rang=len(data)
                for i in range(rang):
                    if (rows>1) & (len(x_axis)>1):
                        axis = ax[jj, j]
                    elif (rows==1) & (len(x_axis)==1):
                        axis = ax
                    elif rows==1:
                        axis = ax[j]
                    elif len(x_axis)==1:
                        axis = ax[jj]
                    
                    if x_offset==0:
                        if time_name in data.dims:
                            d = data.isel({time_name: i})
                        else:
                            if isinstance(data,xr.core.dataarray.DataArray):
                                d=data
                            else:
                                d = data[i]
                    else:
                        d=data[i]
    
                    if sb_kw.use:
                        stat_sg_fig = spatial_stat(d, sb_kw.stats)
                        da_stats_avg[j-x_offset].append(stat_sg_fig)

                    plo = d.plot(ax=axis, vmax=vmax, vmin=vmin, cmap=cm, add_colorbar=False, levels=ck.levels)
                    if type(diff_modus2)==list:
                         if diff_modus2[j-x_offset]=='diff_rel':
                            add_colorbar(fig, plo, vmin, vmax, ck, 'rel', d,labelsize,stat=x_axis[j-x_offset],unit='%')
                         else:
                             add_colorbar(fig, plo, vmin, vmax, ck, diff_modus[j], d, labelsize,stat=x_axis[j-x_offset],unit=unit)
                    else: add_colorbar(fig, plo, vmin, vmax, ck, diff_modus, d, labelsize,stat=x_axis[j],unit=unit)
                    jj += 1
    
                if vmax_i is None:
                    vmax = None
                if vmin_i is None:
                    vmin = None
            idx = jj
    
        fig.suptitle(title, fontsize=titlesize, y=y, fontweight='bold')

        return da_stats_avg

def add_colorbar(fig, plo, vmin, vmax, ck, diff_modus, d, labelsize,stat=None,unit=None):
    tick_labels = np.linspace(vmin, vmax, ck.levels, endpoint=True)
    #debug()
    cb = fig.colorbar(plo, shrink=ck.shrink, extend='both', pad=ck.pad, ticks=tick_labels)
    cb.ax.tick_params(labelsize=10, width=1, length=4)
    if ((diff_modus == None)) and (('percentage' not in stat) and ('count' not in stat)):
        if 'units' in d.attrs:
            u=d.attrs['units']
            if u is not None:
                cb.set_label(u, fontsize=labelsize)
        if 'units' not in d.attrs:
            if isinstance(unit,str):
                cb.set_label(unit, fontsize=labelsize)
    if (('count' in stat) and ('percentage' not in stat)):
        cb.set_label('', fontsize=labelsize)
    elif ('percentage' in stat) or diff_modus=='rel':
        cb.set_label('%', fontsize=labelsize)
    else:
        pass
        #cb.set_label('%', fontsize=labelsize)
        
    # set that only a certain number of ticks is shown
    #for i, label in enumerate(cb.ax.yaxis.get_ticklabels()):
    for i, label in enumerate(cb.ax.yaxis.get_major_ticks()):
        if i % ck.nbins != 0:
            #label.set_visible(False)
            label.draw = lambda *args:None #https://stackoverflow.com/a/19499496/18783204

def plot_with_grid(ax, stats, diff_modus, time_labels, grid_kwargs, projection, data, cartopy_kwargs, sb_kw, da_stats_avg, rows, cols, left_pos, labelsize):
    for i in range(rows):
        for j in range(cols):
            if (rows>1) & (cols>1):
                axis = ax[i, j]
            elif (rows==1) & (cols==1):
                axis = ax
            elif rows==1:
                axis = ax[j]
            elif cols==1:
                axis = ax[i]

            axis.set_title('')
            if i==0:
                if diff_modus[j]!=None: text=stats[j]+' ('+diff_modus[j]+')'
                else: text=stats[j]

                #sometimes in a list for some reason
                if isinstance(text,list): text=text[0]

                axis.set_title(text,fontweight='bold',fontsize=labelsize)            
            if j==0:
                if isinstance(time_labels[i],list): time_lab=time_labels[i][0]
                else: time_lab=time_labels[i]
                axis.text(left_pos,0.5,time_lab,transform=axis.transAxes,ha='center',fontweight='bold',fontsize=labelsize)
                
            #TURN GRID/ON OFF
            gw=SimpleNamespace(**grid_kwargs)
            
            if gw.on:
                 gl = axis.gridlines(crs=projection, draw_labels=gw.draw_labels,linewidth=1, color='gray', alpha=0.5, linestyle='--')
            #TURN X/Y LABELS ON/OFF
            if gw.draw_labels:
                gl.xlabels_top = gw.top; gl.ylabels_right = gw.right; gl.ylabels_left = gw.left; gl.xlabels_bottom = gw.bottom
    
            #FIX GRIDLINE SPACING
            min_lat=np.round(data['lat'].min().values); max_lat=np.round(data['lat'].max().values)
            min_lon=np.round(data['lon'].min().values); max_lon=np.round(data['lon'].max().values)
            lats=np.arange(min_lat,max_lat+gw.lat_step,gw.lat_step); lons=np.arange(min_lon,max_lon+gw.lon_step,gw.lon_step)
    
            gl.xlocator = mticker.FixedLocator(lons); gl.xformatter = LONGITUDE_FORMATTER; gl.ylocator = mticker.FixedLocator(lats); gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': gw.size}; gl.ylabel_style = {'size': gw.size}
            
            #CARTOPY ADD FEATURES
            if cartopy_kwargs['land']==True: axis.add_feature(cfeature.LAND)
            if cartopy_kwargs['rivers']==True: axis.add_feature(cfeature.RIVERS,edgecolor='blue',linewidth=cartopy_kwargs['linewidth'])
            if cartopy_kwargs['coastline']==True: axis.add_feature(cfeature.COASTLINE,linewidth=cartopy_kwargs['linewidth'])


            #SPATIAL MEAN/STATISTICS IN A BOX
            if sb_kw.use:
                try:
                    box_stats=np.array(da_stats_avg)[j,i]
                except:
                    try:
                        box_stats=np.array(da_stats_avg)[i]
                    except:
                        box_stats=np.array(da_stats_avg)[j]
               
                textstr = '\n'.join([f"{sb_kw.stats[iii]} = {x:{sb_kw.prec}}" for iii, x in enumerate(box_stats)])
                props = dict(boxstyle='round', facecolor='white',alpha=1)
                # place a text box in upper right in axes coords
                axis.text(sb_kw.pos[0], sb_kw.pos[1], textstr, transform=axis.transAxes, fontsize=sb_kw.fontsize,verticalalignment='top', bbox=props)


def plot3_cf_runs(
    data1,data2,
    data_names=['data1','data2'],
    comp_mode=[None],#diff_abs,'diff_rel' #comparison mode
    avgs=['all','seasonal','seasonal_t','yearly','monthly','monthly_t'],
    stats='mean',
    diff_modus=None, #can also be diff, diff_rel, diff_all, diff_all,rel #repeated for each data entry entry.
    #this way we can display normal mean, NOTE THAT THIS DOES NOT WORK when 'all' in avgs.
    specific_time=None, #can be used to reduce temporal data to a specific time (e.g. a specific year and season for seasonal_t or for monthly_t'
    #therefore provide a time string (e.g. "2012-02") and closest moment is selected. the selection is done after time resampling/subtracting time means
    #else one could just restrict the data to the start. for selection after seasonal resampling remember to select the startmonth (e.g. december if you want a djf average, february would give you MAM because time has been resampled
    time_name='time_counter',
    cmap='Reds', #can also be length of data ['viridis','viridis']
    cmap_diffs='RdBu_r',
    vmax=None,vmin=None,
    title='Bottom Oxygen',
    height=1.8, width=5.2, y=1, labelsize=None, titlesize=None, left_pos=-0.5,
    grid_kwargs={'on':True,'top':False,'right':False,'bottom':True,'left':True,
'draw_labels':True, 'lon_step':4,'lat_step':2,'size':10},
    cartopy_kwargs={'land':True, 'coastline': False,'rivers': True,'linewidth': 0.5,},
    colorbar_kwargs={ 'levels':11, #levels in the plot every second level has a plot
    'nbins':5,  #only show every 'nth label. 
    'shrink':0.9, 'pad':0.02},
    statsbox_kwargs={'use':False,'stats':['mean','std'],'pos':[0.75,0.95],'prec': '.1f',#precision when printing numbers
                     'fontsize':8},#'stats':['mean','std','median','min','max'
    unit=None
):
    """
    Plot panel:
        x-axis: Data from two runs (+absolute difference or relative difference)
        y-axis: time

    To set: The statistics you want to have (e.g. mean)
    """
    ##### DATA PREPARATION  ####
    if type(comp_mode)!=list: comp_mode=[comp_mode]
    if type(diff_modus)!=list: diff_modus=[diff_modus]
    if len(diff_modus): diff_modus=[diff_modus[0] for _ in range(len(data_names)+len(comp_mode))]
    
    #make a list if not a list
    if type(avgs)!=list: avgs=[avgs]

    time_labels=[];all_data=[]
    #store data in a double layered list/dictionary first layer is the statistics, second is the type of statistic
    
    for i,data in enumerate([data1,data2]):
        da={}
        for ii,a in enumerate(avgs):
            res=temp_resampling_stats(data,averaging=a,mode=stats,diff_modus=diff_modus[i],time_name=time_name,specific_time=specific_time)
            if i==0: time_labels.append(res.attrs['time_label'])
            da[a]=res
        all_data.append(da)

    #do the differences
    for i,m in enumerate(comp_mode):
        if m=='diff_abs': data_names.append('Difference (Abs.)')
        elif m=='diff_rel': data_names.append('Difference (Rel.)')
        else:pass
            
        da={}
        for ii, a in enumerate(avgs):
            empty=[]
            da0=all_data[0][a]
            da1=all_data[1][a]
            for ii, t in enumerate(da0.attrs['time_label']):
                if time_name in da0.dims: d0 = da0.isel({time_name: ii}); d1 =da1.isel({time_name: ii})
                else: d0 = da0; d1 = da1
                if m=='diff_abs': empty.append(d0-d1.values)
                elif m=='diff_rel': empty.append((d0-d1)/d0*100)
                else: pass
            da[a]=empty
        all_data.append(da)

    #try:
    #unpack list of time labels. lentgth needed for setting up plot
    time_labels=[tt for t in time_labels for tt in t]
    #except: time_labels=[time_labels]

    ##### PLOTTING #####
    ck=SimpleNamespace(**colorbar_kwargs)
    sb_kw=SimpleNamespace(**statsbox_kwargs)
    rows=len(time_labels);
    if None not in comp_mode:
        cols=2+len(comp_mode)
    else:
        cols=2

    projection=ccrs.PlateCarree()
    fig,ax=plt.subplots(ncols=cols,nrows=rows,figsize=(cols*width,rows*height),subplot_kw={'projection':projection})
    #plot data from two model runs
    da_stats_avg=plot_data(ax,all_data,avgs, [data1,data2], rows,cmap, time_name, sb_kw, ck, diff_modus, fig, labelsize, title, titlesize, y,vmax,vmin,unit=unit,same_cbar=True)

    if cols>2:
        #plot difference
        da_stats_avg2=plot_data(ax,all_data,avgs, [data1,data2], rows,cmap_diffs, time_name, sb_kw, ck, diff_modus, fig, labelsize, title, titlesize, y,vmax,vmin,x_offset=2,diff_modus2=comp_mode,unit=unit,
                               same_cbar=False)
        da_stats_avg=np.concatenate([da_stats_avg,da_stats_avg2])
    
    ##### PLOT IMPROVEMENTS #####
    plot_with_grid(ax, data_names, diff_modus, time_labels, grid_kwargs, projection, data, cartopy_kwargs, sb_kw, da_stats_avg, rows, cols, left_pos, labelsize)
    fig.tight_layout()
    return fig

sel=lambda ds :ds.sel(lat=slice(44,47),lon=slice(28.5,33.8))
