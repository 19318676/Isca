import xarray as xar
#import matplotlib.pyplot as plt
import numpy as np
import prediction as pred
import random

def create_input_file_from_ANN(exp_name, month_num):
    data_dir = '/emmy-noether/home/jc1420/isca_data'
    file_list = [f'{data_dir}/{exp_name}/run{month_num:04d}/atmos_monthly.nc']

    dataset = xar.open_mfdataset(file_list, decode_times = False)
    grid_latitudes = dataset['lat']
    grid_longitudes = dataset['lon']

    DATA = np.loadtxt('base_isca_array.csv', delimiter = ',', dtype = float)
    m = 0


    for i in range(len(grid_latitudes)):
        for j in range(len(grid_longitudes)):

            T_array = np.array([dataset['temp'].sel(lat=grid_latitudes[i], lon=grid_longitudes[j])])
            T_array = np.flip(T_array)
            q_array = np.array([dataset['sphum'].sel(lat=grid_latitudes[i], lon=grid_longitudes[j])])
            q_array = np.flip(q_array)

            for k in range(len(T_array[0,0,:])):
                DATA[k+2, m] = T_array[0,0,k]
                DATA[k+42,m] = q_array[0,0,k]

            
            m += 1

    DATA2 = np.copy(DATA)
    DATA3 = np.copy(DATA2)
    T_output = pred.predict(DATA[2:,:], 'sigma_T_3_512', 'T')
    q_output = pred.predict(DATA2[2:,:], 'sigma_q_2_448', 'q')

    T_SD_array = np.zeros((40, 64, 128))
    for k in range(40):
        for j in range(len(grid_latitudes)):
            for i in range(len(grid_longitudes)):
                T_SD_array[k,j,i] = T_output[k,i+(j*128)]

    q_SD_array = np.zeros((40, 64, 128))
    for k in range(40):
        for j in range(len(grid_latitudes)):
            for i in range(len(grid_longitudes)):
                q_SD_array[k,j,i] = q_output[k,i+(j*128)]

    return T_SD_array, q_SD_array

    
'''
exp_name = 'realistic_continents_fixed_sst_test_experiment'

#lm_data_dir = '/emmy-noether/home/jc1420/Isca/input/land_masks'
#lm_file_list = [f'{lm_data_dir}/era_land_t42.nc']
#orog_sd_data_dir = '/emmy-noether/home/jc1420/code/'
#orog_sd_file_list = [f'{orog_sd_data_dir}/era5_subgrid_orog_t42.nc']

dataset = xar.open_mfdataset(file_list, decode_times = False)
#lm_dataset = xar.open_mfdataset(lm_file_list, decode_times = False)
#orog_sd_dataset = xar.open_mfdataset(orog_sd_file_list, decode_times = False)
grid_latitudes = dataset['lat']
grid_longitudes = dataset['lon']

DATA = np.loadtxt('base_isca_array.csv', delimiter = ',', dtype = float)
m = 0

for i in range(len(grid_latitudes)):
    for j in range(len(grid_longitudes)):
        #DATA[0,m] = grid_latitudes[i]
        #DATA[1,m] = grid_longitudes[j]
        T_array = np.array([dataset['temp'].sel(lat=grid_latitudes[i], lon=grid_longitudes[j])])
        T_array = np.flip(T_array)
        q_array = np.array([dataset['sphum'].sel(lat=grid_latitudes[i], lon=grid_longitudes[j])])
        q_array = np.flip(q_array)
        #lm_array = np.array([lm_dataset['land_mask'].sel(lat=grid_latitudes[i], lon=grid_longitudes[j])])
        #orog_array = np.array([lm_dataset['zsurf'].sel(lat=grid_latitudes[i], lon=grid_longitudes[j])])
        #sd_orog_array = np.array([orog_sd_dataset['sdor'].sel(lat=grid_latitudes[i], lon=grid_longitudes[j], method = 'nearest')])
        for k in range(len(T_array[0,0,:])):
            DATA[k+2, m] = T_array[0,0,k]
            DATA[k+42,m] = q_array[0,0,k]
            #DATA[-2,m] = lm_array
            #DATA[-3,m] = sd_orog_array
            #DATA[-4,m] = orog_array
            #DATA [-1,m] = 1
        
        m += 1
mini = np.amin(DATA)
maxi = np.amax(DATA)
print(f'The min is {mini} and the max is {maxi}')
DATA2 = np.copy(DATA)
DATA3 = np.copy(DATA2)
T_output = pred.predict(DATA[2:,:], 'sigma_T_3_512', 'T')
q_output = pred.predict(DATA2[2:,:], 'sigma_q_2_448', 'q')
def test_graph():
    col = random.randint(0,len(DATA[0,:]))
    T = DATA3[2:42,col]
    q = DATA3[42:82,col]
    lat, lon = DATA3[0,col], DATA3[1,col]
    T_SD = T_output[:,col]
    q_SD = q_output[:,col]

    plt.figure()
    plt.xlabel('temperature (K)')
    plt.ylabel('Sigma value')
    plt.title(f'test graph for latitude {lat} and longitude {lon}')
    y = np.loadtxt('ISCA_sigma.csv', delimiter = ',', dtype = float)
    plt.plot(T, y, label = 'average', color= 'blue')
    plt.plot(T+T_SD, y, label = 'average '+r'$\pm$'+' 1 SD', color = 'red',linestyle='dashed')
    plt.plot(T-T_SD, y, color = 'red',  linestyle='dashed')
    plt.plot(T+(2*T_SD), y, label = 'average '+r'$\pm$'+' 2 SD', color = 'green',linestyle='dashed')
    plt.plot(T-(2*T_SD), y, color = 'green',  linestyle='dashed')
    plt.plot(T+(3*T_SD), y, label = 'average '+r'$\pm$'+' 3 SD', color = 'purple',linestyle='dashed')
    plt.plot(T-(3*T_SD), y, color = 'purple',  linestyle='dashed')
    plt.ylim(1.02, -0.02)
    plt.legend(loc = 'upper right')
    plt.show()

    plt.figure()
    plt.xlabel('q')
    plt.ylabel('Sigma value')
    plt.title(f'test graph for latitude {lat} and longitude {lon}')
    y = np.loadtxt('ISCA_sigma.csv', delimiter = ',', dtype = float)
    plt.plot(q, y, label = 'average', color= 'blue')
    plt.plot(q+q_SD, y, label = 'average '+r'$\pm$'+' 1 SD', color = 'red',linestyle='dashed')
    plt.plot(q-q_SD, y, color = 'red',  linestyle='dashed')
    plt.plot(q+(2*q_SD), y, label = 'average '+r'$\pm$'+' 2 SD', color = 'green',linestyle='dashed')
    plt.plot(q-(2*q_SD), y, color = 'green',  linestyle='dashed')
    plt.plot(q+(3*q_SD), y, label = 'average '+r'$\pm$'+' 3 SD', color = 'purple',linestyle='dashed')
    plt.plot(q-(3*q_SD), y, color = 'purple',  linestyle='dashed')
    plt.ylim(1.02, -0.02)
    plt.legend(loc = 'upper right')
    plt.show()
#plt.show()
#plt.show()

# reshape data to a 40 by 64 by 128 array
T_SD_array = np.zeros((40, 64, 128))
for k in range(40):
    for j in range(len(grid_latitudes)):
        for i in range(len(grid_longitudes)):
            T_SD_array[k,j,i] = T_output[k,i+(j*128)]

q_SD_array = np.zeros((40, 64, 128))
for k in range(40):
    for j in range(len(grid_latitudes)):
        for i in range(len(grid_longitudes)):
            q_SD_array[k,j,i] = q_output[k,i+(j*128)]

def col_map():
    plt.figure()
    plt.ylim(0,64)
    plt.imshow(T_SD_array[0,:,:])
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.ylim(0,64)
    plt.imshow(q_SD_array[0,:,:])
    plt.colorbar()
    plt.show()

'''