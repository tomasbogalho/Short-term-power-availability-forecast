img_dir = '/Images/'
model_dir = '/models/'
checkpoint_dir = '/checkpoints/'
training_dir = '/training/'
os.makedirs('./outputs' + img_dir, exist_ok=True)
os.makedirs('./outputs' + model_dir, exist_ok=True)
os.makedirs('./outputs' + checkpoint_dir, exist_ok=True)
os.makedirs('./outputs' + training_dir, exist_ok=True)



''' NOTE :

Whenever you want to access database, add this to the beginning of the python file

from  Database import  *

this will import all metadata, tables, engine and session maker.
'''
##################################################################################################
#                                       LIBRARY IMPORTS                                          #
##################################################################################################

import sqlalchemy as sa
from sqlalchemy import Table, Column, Boolean, Integer, String, DateTime,Float, MetaData, create_engine,exc, func, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from tqdm import tqdm
from colorama import Fore, Style
##################################################################################################
#                                         CONNECTION                                             #
##################################################################################################

#   TO DO:
#       -Make this in a CONFIG file

user     = ''
host     = ''
password = ''
port     = ''
sslmode  = ''
database = ''
schema   = ''

conn_string=("postgresql://" + user +":" + password + "@" + host + ":" + port + "/" + database  )


##################################################################################################
#                             ENGINE , SESSION , BASE ,  META                                    #
##################################################################################################

engine  = create_engine(conn_string , echo=True)
Session = sessionmaker(bind=engine)
Base    = declarative_base()
meta    = MetaData(schema='')


##################################################################################################
#                                     Log                                              #
##################################################################################################
Log = Table("Log",meta,

    Column('eventId' ,Integer   , autoincrement=True, primary_key=True, unique=True ),
    Column('Message' ,String(500), default=None),
    Column('Type', String(50),default=None)

)

##################################################################################################
#                                     TRANSACTIONS                                               #
##################################################################################################
Transactions = Table("Transactions",meta,

    Column('TransactionId' ,Integer   , autoincrement=True, primary_key=True, unique=True ),
    Column('ChargePointId' ,String(50), default=None),
    Column('ConnectorId'   ,Integer   , default=None),
    Column('OCPPTagId'     ,String(50), default=None),
    Column('StartTime'     ,String(50), default=None),
    Column('StartValue'    ,Integer   , default=0   ),
    Column('StopTime'      ,String(50), default=None),
    Column('StopValue'     ,Integer   , default=0   ),
    Column('StopReason'    ,String(50), default=None),
    Column('ReservationID' ,Integer   , default=0   ),

)

##################################################################################################
#                                     Monitoring                                                 #
##################################################################################################
Monitoring = Table("Monitoring",meta,

    Column('EventId'                          ,Integer   , autoincrement=True, primary_key=True, unique=True ),
    Column('UUID'                             ,String(50), default=None),
    Column('ChargePointId'                    ,String(50), default=None),
    Column('ConnectorId'                      ,Integer   , default=None),
    Column('TransactionId'                    ,Integer   , default=None),
    Column('Timestamp'                        ,DateTime  , default=None),
    Column('Context'                          ,String(50), default=None),
    Column('Format'                           ,String(50), default=None),

    Column('Energy_Active_Import_Register'    ,Float, default=0),
    Column('Energy_Reactive_Import_Register'  ,Float, default=0),
    Column('Energy_Active_Export_Register'    ,Float, default=0),
    Column('Energy_Reactive_Export_Register'  ,Float, default=0),

    Column('Energy_Active_Import_Interval'    ,Float, default=0),
    Column('Energy_Reactive_Import_Interval'  ,Float, default=0),
    Column('Energy_Active_Export_Interval'    ,Float, default=0),
    Column('Energy_Reactive_Export_Interval'  ,Float, default=0),

    Column('Power_Active_Import'              ,Float, default=0),
    Column('Power_Active_Export'              ,Float, default=0),
    Column('Power_Reactive_Import'            ,Float, default=0),
    Column('Power_Reactive_Export'            ,Float, default=0),
    Column('Power_Offered'                    ,Float, default=0),
    Column('Power_Factor'                     ,Float, default=0),

    Column('Current_Import_L1'                ,Float, default=0),
    Column('Current_Import_L2'                ,Float, default=0),
    Column('Current_Import_L3'                ,Float, default=0),

    Column('Voltage_L1'                       ,Float, default=0),
    Column('Voltage_L2'                       ,Float, default=0),
    Column('Voltage_L3'                       ,Float, default=0),

    Column('Frequency'                        ,Float, default=0),
    Column('Soc'                              ,Float, default=0),
    Column('RPM'                              ,Float, default=0),

)


##################################################################################################
#                                        Events                                                  #
##################################################################################################

Events = Table("Events",meta,
    Column('EventId'  ,Integer ,autoincrement=True, primary_key=True, unique=True ),
    Column('Timestamp',DateTime  , default=None ),
    Column('MessageId',String    ,default='None'),
    Column('ChargePoint'     ,String    ,default='None'),
    Column('MessageType'     ,String    ,default='None'),
    Column('Action'   ,String    ,default='None'),
    Column('Json'     ,JSON      ,default='None'),
    )

##################################################################################################
#                                    CHARGEPOINTS                                                #
##################################################################################################


Chargepoints = Table("ChargePoints",meta,

    Column('name'                          ,String(200), default=None , primary_key=True , unique=True),
    Column('response_timeout'              ,String(200), default=None),
    Column('charge_point_vendor'           ,String(200), default=None),
    Column('charge_point_model'            ,String(200), default=None),
    Column('charge_point_serial_number'    ,String(200), default=None),
    Column('firmware_version'              ,String(200), default=None),
    Column('iccid'                         ,String(200), default=None),
    Column('imsi'                          ,String(200), default=None),
    Column('meter_type'                    ,String(200), default=None),
    Column('meter_serial_number'           ,String(200), default=None),

    Column('LastHB'                        ,DateTime  , default=None),

    Column('Connector0Status'              ,String(200), default=None),
    Column('Connector0Error_code'          ,String(200), default=None),
    Column('Connector0Info'                ,String(200), default=None),
    Column('Connector0Timestamp'           ,DateTime  , default=None),
    Column('Connector0Vendor_id'           ,String(200), default=None),
    Column('Connector0Vendor_error_code'   ,String(200), default=None),

    Column('Connector1Status'              ,String(200), default=None),
    Column('Connector1Error_code'          ,String(200), default=None),
    Column('Connector1Info'                ,String(200), default=None),
    Column('Connector1Timestamp'           ,DateTime  , default=None),
    Column('Connector1Vendor_id'           ,String(200), default=None),
    Column('Connector1Vendor_error_code'   ,String(200), default=None),

    Column('Connector2Status'              ,String(200), default=None),
    Column('Connector2Error_code'          ,String(200), default=None),
    Column('Connector2Info'                ,String(200), default=None),
    Column('Connector2Timestamp'           ,DateTime  , default=None),
    Column('Connector2Vendor_id'           ,String(200), default=None),
    Column('Connector2Vendor_error_code'   ,String(200), default=None),


)

##################################################################################################
#                                         OCPP TAGS                                              #
##################################################################################################
Tags = Table("OCPPTags",meta,

    Column('OCPPTagId'           ,String(50), default=None  , primary_key=True,unique=True),
    Column('ParentIdTag'         ,String(50), default=None ),
    Column('ExpityDate'          ,String(50), default=None ),
    Column('InTransactionStatus' ,Boolean   , default=False),
    Column('BlockedStatus'       ,Boolean   , default=False),
    Column('KnownTag'            ,Boolean   , default=False),
)


##################################################################################################
#                                        Buildings                                               #
##################################################################################################
Buildings = Table("Buildings",meta,

    Column('BuildingId'      ,Integer    , autoincrement=True, primary_key=True, unique=True),
    Column('Name'            ,String(50) , default=None),
    Column('Address'         ,String(500), default=None),
    Column('Type'            ,String(500), default=None),
)

##################################################################################################
#                                           LOG                                                  #
##################################################################################################
Logs = Table("Logs",meta,

    Column('MessageId'          ,Integer    , autoincrement=True, primary_key=True, unique=True),
    Column('Message'            ,String(500), default=None),
)

##################################################################################################
#                                    PV                                                          #
##################################################################################################

PV = Table("PV",meta,
    Column('Timestamp'     ,DateTime, primary_key=True, unique=True ),
    Column('I1'            ,Float   ,  default=0),
    Column('I2'            ,Float   ,  default=0),
    Column('I3'            ,Float   ,  default=0),
    Column('V1'            ,Float   ,  default=0),
    Column('V2'            ,Float   ,  default=0),
    Column('V3'            ,Float   ,  default=0),
    Column('ActPwr'        ,Float   ,  default=0),
)

tenminutes_pv = Table("tenminutes_pv",meta,
    Column('Timestamp'     ,DateTime, primary_key=True, unique=True ),
    Column('I1'            ,Float   ,  default=0),
    Column('I2'            ,Float   ,  default=0),
    Column('I3'            ,Float   ,  default=0),
    Column('V1'            ,Float   ,  default=0),
    Column('V2'            ,Float   ,  default=0),
    Column('V3'            ,Float   ,  default=0),
    Column('ActPwr'        ,Float   ,  default=0),
)

onehour_pv = Table("onehour_pv",meta,
    Column('Timestamp'     ,DateTime, primary_key=True, unique=True ),
    Column('I1'            ,Float   ,  default=0),
    Column('I2'            ,Float   ,  default=0),
    Column('I3'            ,Float   ,  default=0),
    Column('V1'            ,Float   ,  default=0),
    Column('V2'            ,Float   ,  default=0),
    Column('V3'            ,Float   ,  default=0),
    Column('ActPwr'        ,Float   ,  default=0),
)

oneday_pv = Table("oneday_pv",meta,
    Column('Timestamp'     ,DateTime, primary_key=True, unique=True ),
    Column('I1'            ,Float   ,  default=0),
    Column('I2'            ,Float   ,  default=0),
    Column('I3'            ,Float   ,  default=0),
    Column('V1'            ,Float   ,  default=0),
    Column('V2'            ,Float   ,  default=0),
    Column('V3'            ,Float   ,  default=0),
    Column('ActPwr'        ,Float   ,  default=0),
)
##################################################################################################
#                                    CONSUMPTION                                                 #
##################################################################################################

consumption = Table("consumption",meta,
    Column('Timestamp'      ,DateTime, primary_key=True, unique=True ,default=None),
    Column('Ir'             ,Float   ,  default=0),
    Column('Is'             ,Float   ,  default=0),
    Column('It'             ,Float   ,  default=0),
    Column('Vrs'            ,Float   ,  default=0),
    Column('Vst'            ,Float   ,  default=0),
    Column('Vtr'            ,Float   ,  default=0),
    Column('P'              ,Float   ,  default=0),
    Column('S'              ,Float   ,  default=0),
)

tenminutes = Table("tenminutes",meta,
    Column('Timestamp'      ,DateTime, primary_key=True, unique=True ,default=None),
    Column('Ir'             ,Float   ,  default=0),
    Column('Is'             ,Float   ,  default=0),
    Column('It'             ,Float   ,  default=0),
    Column('Vrs'            ,Float   ,  default=0),
    Column('Vst'            ,Float   ,  default=0),
    Column('Vtr'            ,Float   ,  default=0),
    Column('P'              ,Float   ,  default=0),
    Column('S'              ,Float   ,  default=0),
)

onehour = Table("onehour",meta,
    Column('Timestamp'      ,DateTime, primary_key=True, unique=True ,default=None),
    Column('Ir'             ,Float   ,  default=0),
    Column('Is'             ,Float   ,  default=0),
    Column('It'             ,Float   ,  default=0),
    Column('Vrs'            ,Float   ,  default=0),
    Column('Vst'            ,Float   ,  default=0),
    Column('Vtr'            ,Float   ,  default=0),
    Column('P'              ,Float   ,  default=0),
    Column('S'              ,Float   ,  default=0),
)

oneday = Table("oneday",meta,
    Column('Timestamp'      ,DateTime, primary_key=True, unique=True ,default=None),
    Column('Ir'             ,Float   ,  default=0),
    Column('Is'             ,Float   ,  default=0),
    Column('It'             ,Float   ,  default=0),
    Column('Vrs'            ,Float   ,  default=0),
    Column('Vst'            ,Float   ,  default=0),
    Column('Vtr'            ,Float   ,  default=0),
    Column('P'              ,Float   ,  default=0),
    Column('S'              ,Float   ,  default=0),
)







shift_steps_5 = 5
shift_steps_10 = 10
shift_steps_15 = 15



EPSILON = 1e-10


# In[5]:


def Average(lst): 
    return sum(lst) / len(lst) 


# In[ ]:


def _error(actual, predicted):
    """ Simple error """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    return actual - predicted


# In[ ]:


def _percentage_error(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


# In[ ]:


def mape(actual, predicted):
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


# In[ ]:


def mse(actual, predicted):
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


# In[ ]:


def rmse(actual, predicted):
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


# In[ ]:


def nrmse(actual, predicted):
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())


# In[ ]:


def mae(actual, predicted):
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


# In[ ]:


def fnc(year, month, day, hour, minute):
    panel.set_datetime(datetime(year, month, day, hour, minute))
    return panel.power()


# In[ ]:


def s(row):
    if row['Interpolation'] == '-1000':
        return row['Theoretical Value']  
    else:
        return row['Interpolation']


# In[ ]:


def holiday(row):
    if row['day_of_month'] == 1 and row['month'] == 1:
        return 1  
    if row['day_of_month'] == 25 and row['month'] == 2:
        return 1
    if row['day_of_month'] == 10 and row['month'] == 4:
        return 1
    if row['day_of_month'] == 12 and row['month'] == 4:
        return 1  
    if row['day_of_month'] == 25 and row['month'] == 4:
        return 1
    if row['day_of_month'] == 1 and row['month'] == 5:
        return 1  
    if row['day_of_month'] == 10 and row['month'] == 6:
        return 1  
    if row['day_of_month'] == 11 and row['month'] == 6:
        return 1
    if row['day_of_month'] == 15 and row['month'] == 8:
        return 1
    if row['day_of_month'] == 5 and row['month'] == 10:
        return 1  
    if row['day_of_month'] == 1 and row['month'] == 11:
        return 1
    if row['day_of_month'] == 1 and row['month'] == 12:
        return 1
    if row['day_of_month'] == 8 and row['month'] == 12:
        return 1  
    if row['day_of_month'] == 25 and row['month'] == 12:        
        return 1
    else:
        return 0


# In[ ]:


def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]


# In[ ]:


def f(row):
    if row['Avg_GHI'] <= 1:
        return 0
    else:
        return row['ActPwr']


# In[11]:


def collectFromDatabase(db):
    
    if db == True:
        
        # Ler base de dados EDP e Atualizar a base de dados local
        this_session = Session()
        df_consumption_ = this_session.query(consumption).all()
        df_pv_ = this_session.query(PV).all()
        this_session.close()
        df_consumption_ = pd.DataFrame(df_consumption_)
        df_pv_ = pd.DataFrame(df_pv_)
        df_consumption_.to_csv("consumption.csv")
        df_pv_.to_csv("PV.csv")
        
    elif db == False:
        
        # Ler base de dados local
        df_consumption_ = pd.read_csv("consumption.csv")
        df_pv_ = pd.read_csv("PV.csv")
        df_consumption_ = df_consumption_.drop(columns=['Unnamed: 0'])
        df_pv_ = df_pv_.drop(columns=['Unnamed: 0'])
        df_consumption_.Timestamp =  pd.to_datetime(df_consumption_.Timestamp)
        df_pv_.Timestamp =  pd.to_datetime(df_pv_.Timestamp)
        
    return df_pv_, df_consumption_


# In[6]:


def getData(local):
    
    if local == True:
    
        data = pd.read_csv("dfinal.csv")
        data = data.set_index('Timestamp')
        data = data.astype('float64')
    
    else:
    
        dataset1 = np.genfromtxt('dados__Rad.dat', delimiter=',', skip_header=3, missing_values='Missing',names=True,dtype=None)
        dataset2 = np.genfromtxt('dados__Met.dat', delimiter=',', skip_header=3, missing_values='Missing',names=True,dtype=None)

        df1 = pd.DataFrame(data=dataset1)
        df2 = pd.DataFrame(data=dataset2)

        df1.columns = ["Timestamp","RECORD","Avg_GHI","Avg_DHI","Avg_POA","Avg_DNI","Avg_cosDNI","Avg_closGHI","Std_GHI","Std_DHI","Std_POA","Std_DNI","Std_cosDNI","Std_closGHI","Min_GHI","Min_DHI","Min_POA","Min_DNI","Min_cosDNI","Min_closGHI","Max_GHI","Max_DHI","Max_POA","Max_DNI","Max_cosDNI","Max_closGHI","phi","theta","SensorT"]
        df2.columns = ["Timestamp","RECORD","T_amb_min","T_amb_max","T_amb_avg","T_dp_avg","RH_min","RH_max","RH_avg","AH_min","AH_max","AH_avg","p_amb_min","p_amb_max","p_amb_avg","rho_act","v_min","v_max","v_avg","v_vectavg","v_dir_min","v_dir_max","v_dir_vectavg"]

        df1['Timestamp'] = pd.to_datetime(df1['Timestamp'].str.decode("utf-8"), format='"%Y-%m-%d %H:%M:%S"')
        df2['Timestamp'] = pd.to_datetime(df2['Timestamp'].str.decode("utf-8"), format='"%Y-%m-%d %H:%M:%S"')
        df3 = pd.merge(df1,df2, how='inner', left_index=True, right_index=True)

        PV = df_pv.copy()
        consumption = df_consumption.copy()

        consumption = consumption.set_index('Timestamp')
        df3.rename(columns={"Timestamp_x": "Timestamp", "RECORD_x": "Record"}, inplace=True)

        PV['Timestamp'] = PV['Timestamp'].dt.round('min')
        PV = PV.groupby('Timestamp').mean().reset_index()
        PV = PV.set_index('Timestamp')

        idx = pd.period_range(PV.index[0], PV.index[-1], freq='min')
        idy = pd.period_range(consumption.index[0], consumption.index[-1], freq='min')

        PV = PV.reset_index()
        PV = PV.set_index('Timestamp').resample("min").first().reset_index().reindex(columns=PV.columns)
        cols = PV.columns.difference(['I1', 'I2', 'I3', 'V1', 'V2', 'V3', 'ActPwr'])
        PV[cols] = PV[cols].ffill()
        PV['ActPwr'] = PV['ActPwr']*1000

        consumption = consumption.reset_index()
        consumption = consumption.set_index('Timestamp').resample("min").first().reset_index().reindex(columns=consumption.columns)
        cols = consumption.columns.difference(['Ir','Is','It','Vrs','Vst','Vtr','P','S'])
        consumption[cols] = consumption[cols].ffill()

        consumption = consumption.set_index('Timestamp')
        df3 = df3.set_index('Timestamp')

        PV_original = PV.copy()
        PV = PV_original[['Timestamp', 'ActPwr']].copy()

        consumption_original = consumption.copy()
        consumption = consumption_original[['P']]

        # Sun Irradiation Theoretical Calculation

        panel = solar_panel(500, 0.15, id_name='EDP')  # surface, efficiency and name
        panel.set_orientation(array([0, 0, -1]))  # upwards
        panel.set_position(38.707089, -9.148882, 0)  # LISBON latitude, longitude, altitude

        PV = PV.reset_index()
        PV['Theoretical Value'] = PV['Timestamp'].apply(lambda x: fnc(x.year, x.month, x.day, x.hour, x.minute))# year, month, day, hour, minute
        PV = PV.set_index('Timestamp')

        PV['Theoretical Value'] = PV['Theoretical Value'].shift(60, axis = 0) #passar tudo para UTC
        PV = PV.replace(np.nan, '-1')

        # Aqui começa a parte dos dados meteo

        # Juntar os dados meteorológicos e PV num só sítio

        PREV = df4 = pd.merge(df3, PV, left_index=True, right_index=True)

        PREV = PREV.drop(columns=['Timestamp_y', 'RECORD_y'])

        PREV = PREV[['Avg_DHI', 'Avg_GHI', 'Avg_DNI', 'Avg_POA',                      'T_amb_avg','RH_avg', 'AH_avg', 'p_amb_avg', 'rho_act', 'v_avg', 'v_dir_vectavg',                         'Theoretical Value', 'ActPwr']].copy()

        PREV = PREV.replace('-1', np.nan)

        # Adição de dados e Interpolações


        # Adicionar dados da noite 0's à noite

        NOITE = PREV.copy()
        NOITE['ActPwr_noite'] = NOITE.apply(f, axis=1)

        dataaux=NOITE.copy()

        # Interpolação para falhas de valores inferiores a 30 minutos

        INT = NOITE.copy()
        INT['Interpolation'] = INT['ActPwr_noite'].interpolate(method='polynomial', limit=30, order=2)

        dataaux['Interpolation'] = INT['Interpolation'].copy()

        INT['Interpolation'] = INT['Interpolation'].replace(np.nan, '-1000')
        INT['Interpolation_Final'] = INT.apply(s, axis=1)
        INT['Interpolation'] = INT['Interpolation'].replace('-1000',np.nan)

        # Dfinal -> Juntar PV e Consumption

        INT = INT.drop(['ActPwr', 'ActPwr_noite', 'Interpolation'], axis=1)
        INT = INT.rename(columns={"Interpolation_Final": "ActPwr"})

        consumption = consumption.interpolate(method='polynomial', limit=30, order=2)

        dfinal = pd.concat([INT, consumption], axis=1, sort = False, join = 'inner')
        dfinal = dfinal.rename(columns={"Interpolation_Final": "ActPwr"})
        dfinal = dfinal.reset_index()
        dfinal = dfinal.set_index('Timestamp')

        dfinal['hour'] = dfinal.index.hour
        dfinal['day_of_month'] = dfinal.index.day
        dfinal['day_of_week'] = dfinal.index.dayofweek
        dfinal['month'] = dfinal.index.month
        dfinal['holiday'] = dfinal.apply(holiday, axis=1)
        dfinal['AvailablePower'] = 1200000 - dfinal['P'] + dfinal['ActPwr']

        dfinal = dfinal.astype('float64')

        dfinal.to_csv("dfinal.csv")
        
        data = dfinal.copy()
    
    
    return data


# get a list of models to evaluate
def get_models():
    
    Finalist_1 = keras.models.load_model('Finalist1_trained_d4')
    Finalist_2 = keras.models.load_model('Finalist2_trained_d4')
    Finalist_3 = keras.models.load_model('Finalist3_trained_d4')
    
    modelsVec = []
    modelsVec.append(Finalist_1)
    modelsVec.append(Finalist_2)
    modelsVec.append(Finalist_3)
    
    return modelsVec


# In[9]:


def get_models_to_compare():
    
    Finalist_1 = keras.models.load_model('Finalist1_trained_d6')
    Finalist_2 = keras.models.load_model('Finalist2_trained_d6')
    Finalist_3 = keras.models.load_model('Finalist3_trained_d6')
    
    modelsVec = []
    modelsVec.append(Finalist_1)
    modelsVec.append(Finalist_2)
    modelsVec.append(Finalist_3)
    
    return modelsVec


# In[12]:


def predVectorsComputation(test_dataset, models, x, y):
    
    aux =  pd.DataFrame(index=test_dataset.index[:-15])
    #aux =  test_dataset.copy()
    
    #aux.drop(aux.tail(15).index,inplace=True)
    
    
    for z in range(len(models)):

        m = models[z]

        real, predicted, predicted_normalized= prediction(x=x, y_true=y, end_idx=len(aux), model=m)   

        real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,        predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15          = extractPredictionVectors(real, predicted)
        
       
        if z == 2:

            aux['Pred_(t+5)_Model_'+ str(z+1)] = predicted_AvailablePower_5
            aux['Pred_(t+10)_Model_'+ str(z+1)] = predicted_AvailablePower_10
            aux['Pred_(t+15)_Model_'+ str(z+1)] = predicted_AvailablePower_15

            aux['Real_(t+5)'] = real_AvailablePower_5
            aux['Real_(t+10)'] = real_AvailablePower_10
            aux['Real_(t+15)'] = real_AvailablePower_15

        else:
            aux['Pred_(t+5)_Model_'+ str(z+1)] = predicted_AvailablePower_5
            aux['Pred_(t+10)_Model_'+ str(z+1)] = predicted_AvailablePower_10
            aux['Pred_(t+15)_Model_'+ str(z+1)] = predicted_AvailablePower_15
            
    return aux


# In[4]:


def extractPredictionVectors(real, predicted):
    
    rows_real = len(real)
    rows_predicted = len(predicted)


    real_AvailablePower_5 = []
    predicted_AvailablePower_5= []
    real_AvailablePower_10 = []
    predicted_AvailablePower_10 = []
    real_AvailablePower_15 = []
    predicted_AvailablePower_15 = []


    for i in range(rows_real):
        real_AvailablePower_5.append(real[i][0])
        real_AvailablePower_10.append(real[i][1])
        real_AvailablePower_15.append(real[i][2])


    for i in range(rows_predicted):    
        predicted_AvailablePower_5.append(predicted[i][0])
        predicted_AvailablePower_10.append(predicted[i][1])
        predicted_AvailablePower_15.append(predicted[i][2])
        
    return real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15, predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15


# In[10]:


def printTrainErrors(real, predicted):

    real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,    predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15      = extractPredictionVectors(real, predicted)

    train_rmse_5 = rmse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Train Available Power RMSE for 5 minutes prediction: %.2f' % train_rmse_5)

    train_rmse_10 = rmse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Train Available Power RMSE for 10 minutes prediction: %.2f' % train_rmse_10)

    train_rmse_15 = rmse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Train Available Power RMSE for 15 minutes prediction: %.2f' % train_rmse_15)


    #####

    train_mse_5 = mse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Train Available Power MSE for 5 minutes prediction: %.2f' % train_mse_5)

    train_mse_10 = mse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Train Available Power MSE for 10 minutes prediction: %.2f' % train_mse_10)

    train_mse_15 = mse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Train Available Power MSE for 15 minutes prediction: %.2f' % train_mse_15)


    ####

    train_mae_5 = mae(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Train Available Power MAE in 5 minutes: %.2f' % train_mae_5)

    train_mae_10 = mae(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Train Available Power MAE in 10 minutes: %.2f' % train_mae_10)

    train_mae_15 = mae(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Train Available Power MAE in 15 minutes: %.2f' % train_mae_15)
    
    return train_rmse_5, train_rmse_10, train_rmse_15,            train_mse_5, train_mse_10, train_mse_15,            train_mae_5, train_mae_10, train_mae_15
            


# In[ ]:


def printTrainErrorsNormalized(real, predicted):

    real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,    predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15      = extractPredictionVectors(real, predicted)

    train_rmse_5 = rmse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Train Available Power RMSE for 5 minutes prediction: %.6f' % train_rmse_5)

    train_rmse_10 = rmse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Train Available Power RMSE for 10 minutes prediction: %.6f' % train_rmse_10)

    train_rmse_15 = rmse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Train Available Power RMSE for 15 minutes prediction: %.6f' % train_rmse_15)


    #####

    train_mse_5 = mse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Train Available Power MSE for 5 minutes prediction: %.6f' % train_mse_5)

    train_mse_10 = mse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Train Available Power MSE for 10 minutes prediction: %.6f' % train_mse_10)

    train_mse_15 = mse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Train Available Power MSE for 15 minutes prediction: %.6f' % train_mse_15)


    ####

    train_mae_5 = mae(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Train Available Power MAE in 5 minutes: %.6f' % train_mae_5)

    train_mae_10 = mae(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Train Available Power MAE in 10 minutes: %.6f' % train_mae_10)

    train_mae_15 = mae(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Train Available Power MAE in 15 minutes: %.6f' % train_mae_15)
    
    return train_rmse_5, train_rmse_10, train_rmse_15,            train_mse_5, train_mse_10, train_mse_15,            train_mae_5, train_mae_10, train_mae_15
            


# In[14]:


def printAverageErrors(df):
    
    train_rmse_5 = rmse(df['Real_(t+5)'], df['avg_(t+5)'])
    print('Train Available Power RMSE for 5 minutes prediction: %.2f' % train_rmse_5)

    train_rmse_10 = rmse(df['Real_(t+10)'], df['avg_(t+10)'])
    print('Train Available Power RMSE for 10 minutes prediction: %.2f' % train_rmse_10)

    train_rmse_15 = rmse(df['Real_(t+15)'], df['avg_(t+15)'])
    print('Train Available Power RMSE for 15 minutes prediction: %.2f' % train_rmse_15)


    #####

    train_mse_5 = mse(df['Real_(t+5)'], df['avg_(t+5)'])
    print('Train Available Power MSE for 5 minutes prediction: %.2f' % train_mse_5)

    train_mse_10 = mse(df['Real_(t+10)'], df['avg_(t+10)'])
    print('Train Available Power MSE for 10 minutes prediction: %.2f' % train_mse_10)

    train_mse_15 = mse(df['Real_(t+15)'], df['avg_(t+15)'])
    print('Train Available Power MSE for 15 minutes prediction: %.2f' % train_mse_15)


    ####

    train_mae_5 = mae(df['Real_(t+5)'], df['avg_(t+5)'])
    print('Train Available Power MAE in 5 minutes: %.2f' % train_mae_5)

    train_mae_10 = mae(df['Real_(t+10)'], df['avg_(t+10)'])
    print('Train Available Power MAE in 10 minutes: %.2f' % train_mae_10)

    train_mae_15 = mae(df['Real_(t+15)'], df['avg_(t+15)'])
    print('Train Available Power MAE in 15 minutes: %.2f' % train_mae_15)


# In[13]:


def printValidationErrors(real, predicted):

    real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,    predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15      = extractPredictionVectors(real, predicted)

    validation_rmse_5 = rmse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Validation Available Power RMSE for 5 minutes prediction: %.2f' % validation_rmse_5)

    validation_rmse_10 = rmse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Validation Available Power RMSE for 10 minutes prediction: %.2f' % validation_rmse_10)

    validation_rmse_15 = rmse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Validation Available Power RMSE for 15 minutes prediction: %.2f' % validation_rmse_15)


    #####

    validation_mse_5 = mse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Validation Available Power MSE for 5 minutes prediction: %.2f' % validation_mse_5)

    validation_mse_10 = mse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Validation Available Power MSE for 10 minutes prediction: %.2f' % validation_mse_10)

    validation_mse_15 = mse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Validation Available Power MSE for 15 minutes prediction: %.2f' % validation_mse_15)



    ####

    validation_mae_5 = mae(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Validation Available Power MAE in 5 minutes: %.2f' % validation_mae_5)

    validation_mae_10 = mae(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Validation Available Power MAE in 10 minutes: %.2f' % validation_mae_10)

    validation_mae_15 = mae(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Validation Available Power MAE in 15 minutes: %.2f' % validation_mae_15)
    
    return validation_rmse_5, validation_rmse_10, validation_rmse_15,            validation_mse_5, validation_mse_10, validation_mse_15,            validation_mae_5, validation_mae_10, validation_mae_15


# In[ ]:


def printValidationErrorsNormalized(real, predicted):

    real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,    predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15      = extractPredictionVectors(real, predicted)

    validation_rmse_5 = rmse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Validation Available Power RMSE for 5 minutes prediction: %.6f' % validation_rmse_5)

    validation_rmse_10 = rmse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Validation Available Power RMSE for 10 minutes prediction: %.6f' % validation_rmse_10)

    validation_rmse_15 = rmse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Validation Available Power RMSE for 15 minutes prediction: %.6f' % validation_rmse_15)


    #####

    validation_mse_5 = mse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Validation Available Power MSE for 5 minutes prediction: %.6f' % validation_mse_5)

    validation_mse_10 = mse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Validation Available Power MSE for 10 minutes prediction: %.6f' % validation_mse_10)

    validation_mse_15 = mse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Validation Available Power MSE for 15 minutes prediction: %.6f' % validation_mse_15)



    ####

    validation_mae_5 = mae(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Validation Available Power MAE in 5 minutes: %.6f' % validation_mae_5)

    validation_mae_10 = mae(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Validation Available Power MAE in 10 minutes: %.6f' % validation_mae_10)

    validation_mae_15 = mae(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Validation Available Power MAE in 15 minutes: %.6f' % validation_mae_15)
    
    return validation_rmse_5, validation_rmse_10, validation_rmse_15,            validation_mse_5, validation_mse_10, validation_mse_15,            validation_mae_5, validation_mae_10, validation_mae_15


# In[3]:


def printTestErrors(real, predicted):

    real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,    predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15      = extractPredictionVectors(real, predicted)

    test_rmse_5 = rmse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Test Available Power RMSE for 5 minutes prediction: %.2f' % test_rmse_5)

    test_rmse_10 = rmse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Test Available Power RMSE for 10 minutes prediction: %.2f' % test_rmse_10)

    test_rmse_15 = rmse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Test Available Power RMSE for 15 minutes prediction: %.2f' % test_rmse_15)


    #####

    test_mse_5 = mse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Test Available Power MSE for 5 minutes prediction: %.2f' % test_mse_5)

    test_mse_10 = mse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Test Available Power MSE for 10 minutes prediction: %.2f' % test_mse_10)

    test_mse_15 = mse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Test Available Power MSE for 15 minutes prediction: %.2f' % test_mse_15)



    ####

    test_mae_5 = mae(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Test Available Power MAE in 5 minutes: %.2f' % test_mae_5)

    test_mae_10 = mae(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Test Available Power MAE in 10 minutes: %.2f' % test_mae_10)

    test_mae_15 = mae(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Test Available Power MAE in 15 minutes: %.2f' % test_mae_15)
    
    
    return test_rmse_5, test_rmse_10, test_rmse_15, test_mse_5, test_mse_10, test_mse_15, test_mae_5, test_mae_10, test_mae_15


# In[2]:


def printTestErrorsNormalized(real, predicted):

    real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,    predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15      = extractPredictionVectors(real, predicted)

    test_rmse_5 = rmse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Test Available Power RMSE for 5 minutes prediction: %.6f' % test_rmse_5)

    test_rmse_10 = rmse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Test Available Power RMSE for 10 minutes prediction: %.6f' % test_rmse_10)

    test_rmse_15 = rmse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Test Available Power RMSE for 15 minutes prediction: %.6f' % test_rmse_15)


    #####

    test_mse_5 = mse(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Test Available Power MSE for 5 minutes prediction: %.6f' % test_mse_5)

    test_mse_10 = mse(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Test Available Power MSE for 10 minutes prediction: %.6f' % test_mse_10)

    test_mse_15 = mse(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Test Available Power MSE for 15 minutes prediction: %.6f' % test_mse_15)



    ####

    test_mae_5 = mae(real_AvailablePower_5, predicted_AvailablePower_5)
    print('Test Available Power MAE in 5 minutes: %.6f' % test_mae_5)

    test_mae_10 = mae(real_AvailablePower_10, predicted_AvailablePower_10)
    print('Test Available Power MAE in 10 minutes: %.6f' % test_mae_10)

    test_mae_15 = mae(real_AvailablePower_15, predicted_AvailablePower_15)
    print('Test Available Power MAE in 15 minutes: %.6f' % test_mae_15)
    
    
    return test_rmse_5, test_rmse_10, test_rmse_15, test_mse_5, test_mse_10, test_mse_15, test_mae_5, test_mae_10, test_mae_15


# In[15]:


def plotLoss(history, model, dataset):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'],label='Train MSE')
    plt.plot(history.history['val_loss'],label='Validation MSE')
    plt.title('Model Loss')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('Images/' + model + '_Loss_' + dataset)
    plt.show()


# In[8]:


def plot_comparison(x_scaled, y, y_scaler, length, model, train):
    
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    start_idx = 0    
    x = x_scaled
    y_true = y
        
    #if train == 0:
    x = np.expand_dims(x, axis=0)
        
             
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
   
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0]) 
    
    # For each output-signal.
    #for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
    #    signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
    #    signal_true = y_true[:, signal]

    
    return y_true, y_pred_rescaled, y_pred[0]


# In[ ]:


def plot_comparison_second_level(x_scaled, y, length, model, train):
    
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    start_idx = 0    
    x = x_scaled
    y_true = y
        
    #if train == 0:
    x = np.expand_dims(x, axis=0)
        
             
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
   
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_second_level_scaler.inverse_transform(y_pred[0]) 
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

    
    return y_true, y_pred_rescaled, y_pred[0]


# In[5]:


def prediction(x, y_true, end_idx, model):

    # End-index for the sequences.
    start_idx = 0
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0]) 

    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]

        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]
        
        return y_true, y_pred_rescaled, y_pred[0]


# In[ ]:


def prediction_level_2(x, y_true, end_idx, model):

    # End-index for the sequences.
    start_idx = 0
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_second_level_scaler.inverse_transform(y_pred[0]) 

    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]

        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]
        
        return y_true, y_pred_rescaled, y_pred[0]


# In[2]:


def plotTrain(real, predicted, model_name, time_steps, validation_step, original_dataset, save):

    shift_steps_5 = 5
    shift_steps_10 = 10
    shift_steps_15 = 15

    real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,    predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15      = extractPredictionVectors(real, predicted)


    plt.figure(figsize=(20,5))
    font = {'weight' : 'normal','size'   : 12}
    plt.rc('font', **font)
    plt.grid()
    
    if time_steps == 5:

        plt.plot(original_dataset.index.shift(shift_steps_5, freq='min')[:validation_step], real_AvailablePower_5[:validation_step], linestyle="-", label='Measured Available Power')
        plt.plot(original_dataset.index.shift(shift_steps_5, freq='min')[:validation_step], predicted_AvailablePower_5[:validation_step], linestyle="-", label='ANN output forecast')
        
        plt.title('Train Data Available Power - 5 minutes forecast')
        
    if time_steps == 10:
        
        plt.plot(original_dataset.index.shift(shift_steps_10, freq='min')[:validation_step], real_AvailablePower_10[:validation_step][:validation_step], linestyle="-", label='Measured Available Power')
        plt.plot(original_dataset.index.shift(shift_steps_10, freq='min')[:validation_step], predicted_AvailablePower_10[:validation_step], linestyle="-", label='ANN output forecast')
        plt.title('Train Data Available Power - 10 minutes forecast')

    if time_steps == 15:
        
        plt.plot(original_dataset.index.shift(shift_steps_15, freq='min')[:validation_step], real_AvailablePower_15[:validation_step], linestyle="-", label='Measured Available Power')
        plt.plot(original_dataset.index.shift(shift_steps_15, freq='min')[:validation_step], predicted_AvailablePower_15[:validation_step], linestyle="-", label='ANN output forecast')
        plt.title('Train Data Available Power - 15 minutes forecast')
        
        
        
    plt.ylabel('Available Power [W]')
    plt.xlabel('Date')

    plt.legend()
    if save == 1 and time_steps == 5:
        plt.savefig('Images/' + model + '_' + dataset + '_' + 'Train Data Available Power - 5 minutes forecast')
        
    if save == 1 and time_steps == 10:
        plt.savefig('Images/' + model + '_' + dataset + '_' + 'Train Data Available Power - 10 minutes forecast')
        
    if save == 1 and time_steps == 15:
        plt.savefig('Images/' + model + '_' + dataset + '_' + 'Train Data Available Power - 15 minutes forecast')
        
    plt.show()


# In[17]:


def plotValidation(real, predicted, model_name, time_steps, validation_step, original_dataset, save):

    shift_steps_5 = 5
    shift_steps_10 = 10
    shift_steps_15 = 15

    real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,    predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15      = extractPredictionVectors(real, predicted)


    plt.figure(figsize=(20,5))
    font = {'weight' : 'normal','size'   : 12}
    plt.rc('font', **font)
    plt.grid()
    
    if time_steps == 5:

        plt.plot(original_dataset.index.shift(shift_steps_5, freq='min')[validation_step:-15], real_AvailablePower_5, linestyle="-", label='Measured Available Power')
        plt.plot(original_dataset.index.shift(shift_steps_5, freq='min')[validation_step:-15], predicted_AvailablePower_5, linestyle="-", label='ANN output forecast')
        plt.title('Validation Data Available Power - 5 minutes forecast')
        
    if time_steps == 10:
        
        plt.plot(original_dataset.index.shift(shift_steps_10, freq='min')[validation_step:-15], real_AvailablePower_10, linestyle="-", label='Measured Available Power')
        plt.plot(original_dataset.index.shift(shift_steps_10, freq='min')[validation_step:-15], predicted_AvailablePower_10, linestyle="-", label='ANN output forecast')
        plt.title('Validation Data Available Power - 10 minutes forecast')

    if time_steps == 15:
        
        plt.plot(original_dataset.index.shift(shift_steps_15, freq='min')[validation_step:-15], real_AvailablePower_15, linestyle="-", label='Measured Available Power')
        plt.plot(original_dataset.index.shift(shift_steps_15, freq='min')[validation_step:-15], predicted_AvailablePower_15, linestyle="-", label='ANN output forecast')
        plt.title('Validation Data Available Power - 15 minutes forecast')
        
        
        
    plt.ylabel('Available Power [W]')
    plt.xlabel('Date')

    plt.legend()
    if save == 1 and time_steps == 5:
        plt.savefig('Images/' + model + '_' + dataset + '_' + 'Validation Data Available Power - 5 minutes forecast')
        
    if save == 1 and time_steps == 10:
        plt.savefig('Images/' + model + '_' + dataset + '_' + 'Validation Data Available Power - 10 minutes forecast')
        
    if save == 1 and time_steps == 15:
        plt.savefig('Images/' + model + '_' + dataset + '_' + 'Validation Data Available Power - 15 minutes forecast')
        
    plt.show()


# In[ ]:


def plotPrediction(window, start_prediction, real, predicted, x_dataset, y_dataset):
    
    real_AvailablePower_5, real_AvailablePower_10, real_AvailablePower_15,    predicted_AvailablePower_5, predicted_AvailablePower_10, predicted_AvailablePower_15      = extractPredictionVectors(real, predicted)
    
    end_real = start_prediction
    initial_real = end_real - window

    end_real_2 = end_real + 11
    initial_forcast = end_real-5
    end_forcast = initial_forcast + 1


    auxVec = [predicted_AvailablePower_5[initial_forcast:end_forcast],
              predicted_AvailablePower_10[initial_forcast:end_forcast],
              predicted_AvailablePower_15[initial_forcast:end_forcast]]

    aux2Vec = [x_dataset.index.shift(shift_steps_5, freq='min')[initial_forcast:end_forcast],
               x_dataset.index.shift(shift_steps_10, freq='min')[initial_forcast:end_forcast],
               x_dataset.index.shift(shift_steps_15, freq='min')[initial_forcast:end_forcast]]


    plt.figure(figsize=(20,5))
    plt.grid()
    plt.plot(y_dataset.index[initial_real:end_real], real_AvailablePower_5[initial_real:end_real], linestyle="-", label='Measured Available Power')
    plt.plot(y_dataset.index[end_real:end_real_2], real_AvailablePower_5[end_real:end_real_2], linestyle="--", label='Measured Available Power')

    plt.plot(aux2Vec, auxVec, 'go-',markersize=10, label='ANN forecast')

    #plt.plot(x_test_dataset.index.shift(shift_steps_5, freq='min')[initial_forcast:end_forcast], predicted_AvailablePower_5[initial_forcast:end_forcast], 'o',markersize=10, label='ANN output 5 minutes forecast')
    #plt.plot(x_test_dataset.index.shift(shift_steps_10, freq='min')[initial_forcast:end_forcast], predicted_AvailablePower_10[initial_forcast:end_forcast],'o' ,markersize=10, label='ANN output 10 minutes forecast')
    #plt.plot(x_test_dataset.index.shift(shift_steps_15, freq='min')[initial_forcast:end_forcast], predicted_AvailablePower_15[initial_forcast:end_forcast],'o' ,markersize=10, label='ANN output 15 minutes forecast')


    plt.ylabel('Available Power [W]')
    plt.xlabel('Date')
    plt.title('Test Data - Available Power - Forecast')
    plt.legend()
    #plt.savefig('Images/' + model + '_' + dataset + '_' + 'Test Data Available Power - 5 minutes forecast')
    plt.show()


# In[1]:


def batch_generator(batch_size, sequence_length, random_inicialize, num_train, x_train_scaled, y_train_scaled):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, 13)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, 3)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
        

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            if random_inicialize == True:
                idx = np.random.randint(num_train - sequence_length)
            else:
                idx = 0
            # Copy the sequences of data starting at this index.
            
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]           
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]       
        
        
        yield (x_batch, y_batch)


# In[ ]:


def plotExtraCharts(dt):
    
    plt.figure(figsize=(25, 5) )

    font = {'weight' : 'normal','size'   : 21}
    plt.rc('font', **font)
    plt.xlim(737456.5, 737484.5)

    plt.axhline(y=1200000, xmin=0.02, xmax=0.98, color='r', label='Grid')
    plt.plot(dt.index, dt.P, label='Consumption')
    plt.plot(dt.index, dt.ActPwr, label='Production')
    plt.xlabel('Date')
    plt.ylabel('P [W]')
    plt.title('Power trends')
    plt.legend(loc = 'lower right')
    plt.grid()
    #plt.xticks(rotation=90)
    axes = plt.gca()
    plt.savefig("Max_cons_prod")
    
    plt.figure(figsize=(25, 5) )

    font = {'weight' : 'normal','size'   : 21}
    plt.rc('font', **font)
    plt.xlim(737456.5, 737484.5)
    plt.plot(dt.index, dt.AvailablePower, label='Available power')
    plt.xlabel('Date')
    plt.ylabel('P [W]')

    plt.title('Power trends')

    plt.legend()
    plt.grid()
    axes = plt.gca()
    plt.savefig("Available_power")
    
    plt.figure(figsize=(25, 5) )

    font = {'weight' : 'normal','size'   : 21}
    plt.rc('font', **font)
    plt.xlim(737456.5, 737484.5)
    plt.plot(dt.index, dt.P, label='Power consumption')
    plt.xlabel('Date')
    plt.ylabel('P [W]')

    plt.title('Consumption')

    plt.legend()
    plt.grid()
    axes = plt.gca()
    plt.savefig("power_consumption")
    
    plt.figure(figsize=(25, 5) )

    font = {'weight' : 'normal','size'   : 21}
    plt.rc('font', **font)
    plt.xlim(737456.5, 737484.5)
    plt.plot(dt.index, dt.ActPwr, label='Solar production')
    plt.xlabel('Date')
    plt.ylabel('P [W]')

    plt.title('Solar production')

    plt.legend()
    plt.grid()
    axes = plt.gca()
    plt.savefig("power_production")


# In[8]:


def train_model(model, trainX, trainY, valX, valY, batch, epoch_count, cb, val):
    
    if val == True:    
        history = model.fit(trainX, trainY, batch_size=batch, epochs=epoch_count, \
                        callbacks=cb, validation_data = (valX, valY))
    else:    
        history = model.fit(trainX, trainY, batch_size=batch, epochs=epoch_count)
    return history

# In[7]:


def create_model(steps_before, steps_after, cnn, feature_count, units, layer, ft, intervals):
    """ 
        creates, compiles and returns a RNN model 
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """       
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras import backend as K

    
    
    if intervals == True:
    
        if layer == 'GRU':

            if cnn == True:
                model = Sequential()

                model = Sequential()
                model.add(Conv1D(filters=ft, kernel_size=2, activation='relu', input_shape=(steps_before, feature_count),padding = 'causal'))
                model.add(Conv1D(filters=ft, kernel_size=2, activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
                model.add(RepeatVector(steps_after))
                model.add(GRU(units, activation='relu', return_sequences=True))
                model.add(Lambda(lambda x: K.dropout(x, level=0.2)))
                model.add(TimeDistributed(Dense(1)))

            else:

                model = Sequential()
                model.add(GRU(ft, activation='relu', input_shape=(steps_before, feature_count)))
                model.add(RepeatVector(steps_after))
                model.add(GRU(units, activation='relu', return_sequences=True))
                model.add(Lambda(lambda x: K.dropout(x, level=0.2)))
                model.add(TimeDistributed(Dense(1)))
                


        elif layer == 'LSTM':

            if cnn == True:

                model = Sequential()
                model.add(Conv1D(filters=ft, kernel_size=2, activation='relu', input_shape=(steps_before, feature_count),padding = 'causal'))
                model.add(Conv1D(filters=ft, kernel_size=2, activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
                model.add(RepeatVector(steps_after))
                model.add(LSTM(units, activation='relu', return_sequences=True))
                model.add(Lambda(lambda x: K.dropout(x, level=0.2)))
                model.add(TimeDistributed(Dense(1)))

            else:

                model = Sequential()
                model.add(LSTM(ft, activation='relu', input_shape=(steps_before, feature_count)))
                model.add(RepeatVector(steps_after))
                model.add(LSTM(units, activation='relu', return_sequences=True))
                model.add(Lambda(lambda x: K.dropout(x, level=0.2)))
                model.add(TimeDistributed(Dense(1)))
        else:
            print('Error: Type of layer not defined')
       
    else:
    
        if layer == 'GRU':

            if cnn == True:
                model = Sequential()

                model = Sequential()
                model.add(Conv1D(filters=ft, kernel_size=2, activation='relu', input_shape=(steps_before, feature_count),padding = 'causal'))
                model.add(Conv1D(filters=ft, kernel_size=2, activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
                model.add(RepeatVector(steps_after))
                model.add(GRU(units, activation='relu', return_sequences=True))
                model.add(Dropout(0.2))
                model.add(TimeDistributed(Dense(1)))

            else:

                model = Sequential()
                model.add(GRU(ft, activation='relu', input_shape=(steps_before, feature_count)))
                model.add(RepeatVector(steps_after))
                model.add(GRU(units, activation='relu', return_sequences=True))
                model.add(Dropout(0.2))
                model.add(TimeDistributed(Dense(1)))

        elif layer == 'LSTM':

            if cnn == True:

                model = Sequential()
                model.add(Conv1D(filters=ft, kernel_size=2, activation='relu', input_shape=(steps_before, feature_count),padding = 'causal'))
                model.add(Conv1D(filters=ft, kernel_size=2, activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
                model.add(RepeatVector(steps_after))
                model.add(LSTM(units, activation='relu', return_sequences=True))
                model.add(Dropout(0.2))
                model.add(TimeDistributed(Dense(1)))

            else:

                model = Sequential()
                model.add(LSTM(ft, activation='relu', input_shape=(steps_before, feature_count)))
                model.add(RepeatVector(steps_after))
                model.add(LSTM(units, activation='relu', return_sequences=True))
                model.add(Dropout(0.2))
                model.add(TimeDistributed(Dense(1)))
        else:
            print('Error: Type of layer not defined')        
    
    model.compile(loss='mse', optimizer='adam', metrics = ['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()])
    
    model.summary()
    return model


# In[2]:

def create_model_vanilla(steps_before, feature_count, units, layer, intervals):
    """ 
        creates, compiles and returns a RNN model 
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    
        
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras import backend as K

    
    if intervals == False:
    
        if layer == 'GRU':

            model = keras.Sequential()
            model.add(GRU(units=units, input_shape=(steps_before, feature_count)))
            #model.add(Lambda(lambda x: K.dropout(x, level=0.1)))
            model.add(Dropout(0.2))
            model.add(Dense(3))

        if layer == 'LSTM':

            model = keras.Sequential()
            model.add(LSTM(units=units, input_shape=(steps_before, feature_count)))
            #model.add(Lambda(lambda x: K.dropout(x, level=0.1)))
            model.add(Dropout(0.2))
            model.add(Dense(3))

    if intervals == True: 
        
        if layer == 'GRU':

            model = keras.Sequential()
            model.add(GRU(units=units, input_shape=(steps_before, feature_count)))
            model.add(Lambda(lambda x: K.dropout(x, level=0.2)))
            model.add(Dense(3))

        if layer == 'LSTM':

            model = keras.Sequential()
            model.add(LSTM(units=units, input_shape=(steps_before, feature_count)))
            model.add(Lambda(lambda x: K.dropout(x, level=0.2)))
            model.add(Dense(3))

    
    model.compile(loss='mse', optimizer='adam', metrics = ['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()])
    
    model.summary()
    return model


def callbacksFunction(model_name, dataset):
    
    path_checkpoint = './outputs/checkpoints/checkpoint_' + model_name + '_' + dataset + '.keras'
    
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)

    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    callback_tensorboard = TensorBoard(log_dir='./logs/',
                                       histogram_freq=0,
                                       write_graph=False, profile_batch = 100000000)

    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-4,
                                           patience=1,
                                           verbose=1)

    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    time_callback = TimeHistory()

    csv_logger = CSVLogger('./outputs/training/training_' + model_name + '_' + dataset + '.log', separator=',', append=False)

    callbacks = [callback_early_stopping,
                 callback_checkpoint,
                 callback_tensorboard,
                 callback_reduce_lr,
                 time_callback,
                 csv_logger]
    
    return callbacks, path_checkpoint


# In[9]:


def data_gen():
    while True:
        x = np.random.rand(512, 15, 12)  # batch x time x features
        yield x, x[:, :, 0] * x[:, :, 1] < 0.25


# In[ ]:



def var_importance(model):
    g = data_gen()
    x = np.concatenate([next(g)[0] for _ in range(50)]) # Get a sample of data
    orig_out = model.predict(x)
    for i in range(12):  # iterate over the three features
        new_x = x.copy()
        perturbation = np.random.normal(0.0, 0.2, size=new_x.shape[:2])
        new_x[:, :, i] = new_x[:, :, i] + perturbation
        perturbed_out = model.predict(new_x)
        effect = ((orig_out - perturbed_out) ** 2).mean() ** 0.5
        print(f'Variable {i+1}, perturbation effect: {effect:.4f}')


# In[ ]:


def features():
    features = ['Avg_GHI',
                'RH_avg',
                'T_amb_avg',
                'hour', 
                'month', 
                'holiday',
                'day_of_week',
                'AvailablePower']
    return features


# In[ ]:


def printReconstruction(dt):

    plt.rcParams.update({'figure.figsize':(20,10), 'figure.dpi':300})
    font = {'weight' : 'normal','size'   : 28}
    plt.rc('font', **font)
    pyplot.figure()

    dt['ActPwr'].plot(label='Phase 5', linewidth=4, color='red')
    dt['Theoretical Value'].plot(label='Phase 4', linewidth=1, color='blue',)

    plt.title('Production Active Power')
    plt.xlim('02/24/2020 06:30', '02/24/2020 19:00')
    plt.ylim(-5,65000 )
    plt.xlabel('Date')
    plt.ylabel('[W]')
    plt.legend()
    axes = plt.gca()
    plt.grid()
    plt.savefig("int0")
    pyplot.show()


# In[ ]:


def getDatasets(dfinal):
    
    a1 = dfinal[1440:102179].copy()
    v1 = dfinal[102179:122339].copy()
    a2 = dfinal[1440:122339].copy()
    v2 = dfinal[122339:142499].copy()
    a3 = dfinal[1440:142499].copy()
    v3 = dfinal[142499:162659].copy()
    a4 = dfinal[1440:162659].copy()
    v4 = dfinal[162659:182819].copy()
    a5 = dfinal[1440:182819].copy()
    t5 = dfinal[182819:202979].copy()
    d6 = dfinal[1440:202979].copy()
    
    return a1, a2, a3, a4, a5, v1, v2, v3, v4, t5, d6


# In[3]:


def alarm():
    from pygame import mixer 
    mixer.init() 
    mixer.music.load("alarm.mp3") 
    mixer.music.set_volume(1) 
    mixer.music.play() 







import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

from pylab import rcParams
import seaborn as sns
import math
import os
from datetime import datetime,  timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
 

from matplotlib import pyplot

from solarpy import irradiance_on_plane
from solarpy import solar_panel
from numpy import array

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D, RepeatVector, Lambda, TimeDistributed, Embedding, TimeDistributed, BatchNormalization, Reshape, concatenate, Permute
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.backend import square, mean


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.utils import to_categorical



plt.rcParams.update({'figure.figsize':(20,5), 'figure.dpi':300})


#df_pv, df_consumption = collectFromDatabase(True)


dfinal = getData(local = True)


a1, a2, a3, a4, a5, v1, v2, v3, v4, t5, d6 = getDatasets(dfinal)


dtr = [a1, a2, a3, a4]
dv = [v1, v2, v3, v4]
dtt = [v2, v3, v4, t5]



def Vanilla(layer, dtr, dv, dtt, cnn):
       
    array = [16, 64, 256]
    sl = [60]
    
    for l in range (0, len(sl)):

        for d in range (0, len(dtr)):

            for i in range (0, len(array)):

                model_name = layer + '-' + str(array[i]) + '-'+ str(d+1) + '-'+ str(sl[l])


                U = int(array[i])
                
                dt = dtr[d].copy()
                
                dataset = str(d+1)

                print(model_name, dataset)
                
                finalfeatures = features()

                dt = dt[finalfeatures].copy()

                dt_val = dv[d].copy()

                dt_val = dt_val[finalfeatures].copy()

                dt_test = dtt[d].copy()

                dt_test = dt_test[finalfeatures].copy()

                train = dt.values
                val = dt_val.values
                test = dt_test.values

                n_pre = int(sl[l])
                n_post = 15

                dX, dY = [], []
                for i in range(len(train)-n_pre-n_post):
                    dX.append(train[i:i+n_pre])
                trainX = np.array(dX)


                for i in range(len(train)-n_pre-n_post):

                    ar = np.array(([row[7] for row in train[i+n_pre:i+n_pre+n_post]][4], [row[7] for row in train[i+n_pre:i+n_pre+n_post]][9], [row[7] for row in train[i+n_pre:i+n_pre+n_post]][14]))
                    B = ar.reshape(-1, len(ar))
                    dY.append(B)
                    
                trainY = np.array(dY)


                vX, vY = [], []
                for i in range(len(val)-n_pre-n_post):
                    vX.append(val[i:i+n_pre])
                valX = np.array(vX)

                for i in range(len(val)-n_pre-n_post):

                    ar = np.array(([row[7] for row in val[i+n_pre:i+n_pre+n_post]][4], [row[7] for row in val[i+n_pre:i+n_pre+n_post]][9], [row[7] for row in val[i+n_pre:i+n_pre+n_post]][14]))
                    B = ar.reshape(-1, len(ar))
                    vY.append(B)

                valY = np.array(vY)

                tX, tY = [], []
                for i in range(len(test)-n_pre-n_post):
                    tX.append(test[i:i+n_pre])
                testX = np.array(tX)

                for i in range(len(test)-n_pre-n_post):

                    ar = np.array(([row[7] for row in test[i+n_pre:i+n_pre+n_post]][4], [row[7] for row in test[i+n_pre:i+n_pre+n_post]][9], [row[7] for row in test[i+n_pre:i+n_pre+n_post]][14]))
                    B = ar.reshape(-1, len(ar))
                    tY.append(B)

                testY = np.array(tY)

                trainX_original = trainX.copy()
                trainY_original = trainY.copy()
                valX_original = valX.copy()
                valY_original = valY.copy()
                testX_original = testX.copy()
                testY_original = testY.copy()

                trainY = trainY.reshape(trainY.shape[0], trainY.shape[2])
                valY = valY.reshape(valY.shape[0], valY.shape[2])
                testY = testY.reshape(testY.shape[0], testY.shape[2])

                scalersX = {}

                for i in range(trainX.shape[2]):
                    scalersX[i] = MinMaxScaler(feature_range=(0, 1))
                    trainX[:, :, i] = scalersX[i].fit_transform(trainX[:, :, i]) 

                for i in range(valX.shape[2]):
                    valX[:, :, i] = scalersX[i].transform(valX[:, :, i]) 
                    
                for i in range(testX.shape[2]):
                    testX[:, :, i] = scalersX[i].transform(testX[:, :, i]) 

                scalerY = MinMaxScaler(feature_range=(0, 1))
                trainY = scalerY.fit_transform(trainY) 
                valY = scalerY.transform(valY)
                testY = scalerY.transform(testY)

                print('creating model...')
                model = create_model_vanilla(steps_before = n_pre, feature_count = 8, units = U, layer = layer, intervals = True)
                callbacks, path_checkpoint = callbacksFunction(model_name, dataset)
                history = train_model(model, trainX, trainY, valX, valY, 512, 100, callbacks, True)

                try:
                    model.load_weights(path_checkpoint)
                except Exception as error:
                    print("Error trying to load checkpoint.")
                    print(error)

                n_epochs = len(history.history['loss'])
                times = callbacks[4].times 
                Time_Epoch = Average(times) 
                Total_Time = sum(times) 

                Train_mae = min(history.history['mae'])
                Train_mse = min(history.history['mse'])
                Train_rmse = min(history.history['root_mean_squared_error'])

                Validation_mae = min(history.history['val_mae'])
                Validation_mse = min(history.history['val_mse'])
                Validation_rmse = min(history.history['val_root_mean_squared_error'])

                nan_array = np.empty((n_pre - 1))
                nan_array.fill(np.nan)
                nan_array2 = np.empty(n_post)
                nan_array2.fill(np.nan)
                ind = np.arange(n_pre + n_post)

                GlobalPredictions = []
                GlobalPredictions_normalized = []

                for j in range (0,5):
                    
                    print(j, end= ' ')
                    predict_ = model.predict(testX)
                    predict_original_ = predict_.copy()

                    predict = scalerY.inverse_transform(predict_)
                    
                    GlobalPredictions.append(predict)
                    GlobalPredictions_normalized.append(predict_)
                    
                GlobalPredictions = np.array(GlobalPredictions)
                GlobalPredictions_normalized = np.array(GlobalPredictions_normalized)


                mean = np.mean(GlobalPredictions, axis=0)

                ci_1 = 0.80
                lower_lim_1 = np.quantile(GlobalPredictions, 0.5-ci_1/2, axis=0)
                upper_lim_1 = np.quantile(GlobalPredictions, 0.5+ci_1/2, axis=0)

                ci_2 = 0.95
                lower_lim_2 = np.quantile(GlobalPredictions, 0.5-ci_2/2, axis=0)
                upper_lim_2 = np.quantile(GlobalPredictions, 0.5+ci_2/2, axis=0)


                aux = np.full([valX_original.shape[0],1,15], np.nan)

                aux2 = np.full([mean.shape[0],1,15], np.nan)

                aux3 = np.full([lower_lim_1.shape[0],1,15], np.nan)

                aux4 = np.full([upper_lim_1.shape[0],1,15], np.nan)

                aux5 = np.full([lower_lim_2.shape[0],1,15], np.nan)

                aux6 = np.full([upper_lim_2.shape[0],1,15], np.nan)

                fig, ax = plt.subplots(figsize=(10, 3.5))
                for i in range(200, valX.shape[0], valX.shape[0]):
                    
                    aux[i, 0, 4] = valY_original[i, 0, 0]
                    aux[i, 0, 9] = valY_original[i, 0, 1]
                    aux[i, 0, 14] = valY_original[i, 0, 2]
                    
                    aux2[i, 0, 4] = mean[i, 0]
                    aux2[i, 0, 9] = mean[i, 1]
                    aux2[i, 0, 14] = mean[i, 2]
                    
                    aux3[i, 0, 4] = lower_lim_1[i, 0]
                    aux3[i, 0, 9] = lower_lim_1[i, 1]
                    aux3[i, 0, 14] = lower_lim_1[i, 2]
                    
                    aux4[i, 0, 4] = upper_lim_1[i, 0]
                    aux4[i, 0, 9] = upper_lim_1[i, 1]
                    aux4[i, 0, 14] = upper_lim_1[i, 2]
                    
                    aux5[i, 0, 4] = lower_lim_2[i, 0]
                    aux5[i, 0, 9] = lower_lim_2[i, 1]
                    aux5[i, 0, 14] = lower_lim_2[i, 2]
                    
                    aux6[i, 0, 4] = upper_lim_2[i, 0]
                    aux6[i, 0, 9] = upper_lim_2[i, 1]
                    aux6[i, 0, 14] = upper_lim_2[i, 2]
                    
                    forecasts = np.concatenate((nan_array, valX_original[i, -1:, 7], aux2[i, 0, :]))
                    
                    lower_lim_1 = np.concatenate((nan_array, valX_original[i, -1:, 7], aux3[i, 0, :]))
                    upper_lim_1 = np.concatenate((nan_array, valX_original[i, -1:, 7], aux4[i, 0, :]))
                    lower_lim_2 = np.concatenate((nan_array, valX_original[i, -1:, 7], aux5[i, 0, :]))
                    upper_lim_2 = np.concatenate((nan_array, valX_original[i, -1:, 7], aux6[i, 0, :]))
                    
                    ground_truth = np.concatenate((nan_array, valX_original[i, -1:, 7], aux[i, 0, :]))
                    network_input = np.concatenate((valX_original[i, :, 7], nan_array2))

                    plt.xticks(rotation=45)
                    
                    SMALLER_SIZE = 18
                    SMALL_SIZE = 19
                    MEDIUM_SIZE = 23
                    BIGGER_SIZE = 25

                    #plt.('test title', fontsize=BIGGER_SIZE)
                    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
                    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
                    plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
                    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                    plt.rc('legend', fontsize=SMALLER_SIZE)  # legend fontsize
                    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


                    
                    plt.fill_between(ind, lower_lim_2, upper_lim_2, color='orange', linewidth=2, label = str(int(ci_2 *100)) + '% CI')
                        
                    ax.plot(ind, lower_lim_2, '-o', color='orange', markersize=20, marker='_')
                    ax.plot(ind, upper_lim_2, '-o', color='orange', markersize=20, marker='_')
                    
                    ax.plot(ind, ground_truth, 'r-o', markersize=10, markeredgecolor='black', linewidth=2, label='Av. Power')
                    
                    ax.plot(ind, forecasts, 'go--', markersize=10, linewidth=2, marker='h', markerfacecolor='lightgreen', \
                             markeredgewidth=2, label='Forecast')
                            
                    ax.plot(ind[40:], network_input[40:], '-o', markersize=10, markeredgecolor='black', linewidth=2, label='Av. Power')
                    
                    plt.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
                    plt.xlabel('Date')
                    plt.ylabel('Available Power [W]')
                    plt.title('Model 2 - Available Power - Forecast')
                    plt.legend(loc='lower left')
                    #plt.savefig('Images/' + model_name , bbox_inches = 'tight')
                    plt.savefig('./outputs/Images/' + model_name + '_op2', bbox_inches = 'tight')


                predicted_normalized = []
                predicted_mean_normalized = []
                original_normalized = []

                predicted = []
                predicted_mean = []
                original = []

                mean = np.mean(GlobalPredictions, axis=0)

                mean_normalized = np.mean(GlobalPredictions_normalized, axis=0)

                aux = np.full([testY_original.shape[0],1,15], np.nan)

                aux5 = np.full([testY.shape[0],1,15], np.nan)

                aux1 = np.full([predict.shape[0],1,15], np.nan)

                aux2 = np.full([predict.shape[0],1,15], np.nan)

                aux3 = np.full([mean.shape[0],1,15], np.nan)

                aux4 = np.full([mean_normalized.shape[0],1,15], np.nan)


                for j in range (0, testX.shape[0]):
                    for i in range(j, testX.shape[0], testX.shape[0]):
                        
                        aux[i, 0, 4] = testY_original[i, 0, 0]
                        aux[i, 0, 9] = testY_original[i, 0, 1]
                        aux[i, 0, 14] = testY_original[i, 0, 2]
                        
                        aux3[i, 0, 4] = mean[i, 0]
                        aux3[i, 0, 9] = mean[i, 1]
                        aux3[i, 0, 14] = mean[i, 2]
                        
                        aux4[i, 0, 4] = mean_normalized[i, 0]
                        aux4[i, 0, 9] = mean_normalized[i, 1]
                        aux4[i, 0, 14] = mean_normalized[i, 2]
                        
                        aux5[i, 0, 4] = testY[i, 0]
                        aux5[i, 0, 9] = testY[i, 1]
                        aux5[i, 0, 14] = testY[i, 2]
                        
                        forecasts_mean_normalized = np.concatenate((nan_array, testX[i, -1:, 7], aux4[i, 0, :]))
                        forecasts_mean = np.concatenate((nan_array, testX_original[i, -1:, 7], aux3[i, 0, :]))
                        
                        ground_truth_normalized = np.concatenate((nan_array, testX[i, -1:, 7], aux5[i, 0, :]))
                        ground_truth = np.concatenate((nan_array, testX_original[i, -1:, 7], aux[i, 0, :]))
                        
                        predicted_mean_normalized.append((forecasts_mean_normalized[n_pre+4], forecasts_mean_normalized[n_pre+9], forecasts_mean_normalized[n_pre+14]))
                        predicted_mean.append((forecasts_mean[n_pre+4], forecasts_mean[n_pre+9], forecasts_mean[n_pre+14]))
                        
                        original_normalized.append((ground_truth_normalized[n_pre+4], ground_truth_normalized[n_pre+9], ground_truth_normalized[n_pre+14]))
                        original.append((ground_truth[n_pre+4], ground_truth[n_pre+9], ground_truth[n_pre+14]))




                test_rmse_5, test_rmse_10, test_rmse_15, test_mse_5, test_mse_10, test_mse_15, test_mae_5, test_mae_10, test_mae_15 = printTestErrors(original, predicted_mean)

                test_rmse_5_n, test_rmse_10_n, test_rmse_15_n, test_mse_5_n, test_mse_10_n, test_mse_15_n, test_mae_5_n, test_mae_10_n, test_mae_15_n = printTestErrorsNormalized(original_normalized, predicted_mean_normalized)

                result = model.evaluate(x=testX, y=testY)
                print("loss (test-set):", result)

                with open('./outputs/Test_' + dataset + '.csv', 'a', newline='') as file:
                    fieldnames = ['Model', 
                                  'Validation_MSE', 'Validation_RMSE', 'Validation_MAE',
                                  'Test_MSE_n', 'Test_RMSE_n', 'Test_MAE_n',                  
                                  'Test_5_RMSE_n', 'Test_10_RMSE_n', 'Test_15_RMSE_n', 
                                  'Test_5_MSE_n', 'Test_10_MSE_n', 'Test_15_MSE_n', 
                                  'Test_5_MAE_n', 'Test_10_MAE_n', 'Test_15_MAE_n',
                                  
                                  'Test_5_RMSE', 'Test_10_RMSE', 'Test_15_RMSE', 
                                  'Test_5_MSE', 'Test_10_MSE', 'Test_15_MSE', 
                                  'Test_5_MAE', 'Test_10_MAE', 'Test_15_MAE']
                    
                    writer = csv.DictWriter(file, fieldnames=fieldnames)

                    writer.writeheader()

                with open('./outputs/Test_' + dataset + '.csv', 'a', newline='') as file:
                    fieldnames = ['Model',
                                  'Validation_MSE', 'Validation_RMSE', 'Validation_MAE',
                                  'Test_MSE_n', 'Test_RMSE_n', 'Test_MAE_n',                  
                                  'Test_5_RMSE_n', 'Test_10_RMSE_n', 'Test_15_RMSE_n', 
                                  'Test_5_MSE_n', 'Test_10_MSE_n', 'Test_15_MSE_n', 
                                  'Test_5_MAE_n', 'Test_10_MAE_n', 'Test_15_MAE_n',
                                
                                  'Test_5_RMSE', 'Test_10_RMSE', 'Test_15_RMSE', 
                                  'Test_5_MSE', 'Test_10_MSE', 'Test_15_MSE', 
                                  'Test_5_MAE', 'Test_10_MAE', 'Test_15_MAE']
                    
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                       
                    writer.writerow({'Model': model_name, 

                                     'Validation_MSE': Validation_mse, 
                                     'Validation_RMSE': Validation_rmse, 
                                     'Validation_MAE':Validation_mae , 

                                     
                                     'Test_MSE_n': result[2], 
                                     'Test_RMSE_n': result[3],
                                     'Test_MAE_n': result[1], 
                                     
                                     'Test_5_RMSE_n': test_rmse_5_n,
                                     'Test_10_RMSE_n': test_rmse_10_n,
                                     'Test_15_RMSE_n': test_rmse_15_n, 
                                     'Test_5_MSE_n': test_mse_5_n,
                                     'Test_10_MSE_n': test_mse_10_n,
                                     'Test_15_MSE_n': test_mse_15_n, 
                                     'Test_5_MAE_n': test_mae_5_n,
                                     'Test_10_MAE_n': test_mae_10_n,
                                     'Test_15_MAE_n': test_mae_15_n,
                                                     
                                     'Test_5_RMSE': test_rmse_5,
                                     'Test_10_RMSE': test_rmse_10,
                                     'Test_15_RMSE': test_rmse_15, 
                                     'Test_5_MSE': test_mse_5,
                                     'Test_10_MSE': test_mse_10,
                                     'Test_15_MSE': test_mse_15, 
                                     'Test_5_MAE': test_mae_5,
                                     'Test_10_MAE': test_mae_10,
                                     'Test_15_MAE': test_mae_15                 
                                                        })




def encoder_decoder(layer, dtr, dv, dtt, cnn):
       
    array = [16, 64, 256]
    filters = [16, 64, 256]
    sl = [60]
    
    for l in range (0, len(sl)):

        for d in range (0, len(dtr)):

            for f in range (0, len(filters)):
        
                for i in range (0, len(array)):

                    model_name = 'ED-' + layer + '-' + layer + '-' + str(filters[f]) + '-' + str(array[i]) + '-'+ str(d+1) + '-'+ str(sl[l])

                    U = int(array[i])
                    
                    fil = int(filters[f])

                    dt = dtr[d].copy()
                    
                    dataset = str(d+1)

                    print(model_name, dataset)
                    
                    finalfeatures = features()

                    dt = dt[finalfeatures].copy()

                    dt_val = dv[d].copy()

                    dt_val = dt_val[finalfeatures].copy()

                    dt_test = dtt[d].copy()
                    dt_test = dt_test[finalfeatures].copy()


                    train = dt.values
                    val = dt_val.values
                    test = dt_test.values

                    n_pre = int(sl[l])
                    n_post = 15

                    dX, dY = [], []
                    for i in range(len(train)-n_pre-n_post):
                        dX.append(train[i:i+n_pre])
                    trainX = np.array(dX)

                    for i in range(len(train)-n_pre-n_post):

                        ar = np.array([row[7] for row in train[i+n_pre:i+n_pre+n_post]])
                        B = ar.reshape(len(ar),-1)
                        dY.append(B)
                        
                    trainY = np.array(dY)

                    vX, vY = [], []
                    for i in range(len(val)-n_pre-n_post):
                        vX.append(val[i:i+n_pre])
                    valX = np.array(vX)

                    for i in range(len(val)-n_pre-n_post):

                        ar = np.array([row[7] for row in val[i+n_pre:i+n_pre+n_post]])
                        B = ar.reshape(len(ar),-1)
                        vY.append(B)

                    valY = np.array(vY)

                    tX, tY = [], []
                    for i in range(len(test)-n_pre-n_post):
                        tX.append(test[i:i+n_pre])
                    testX = np.array(tX)

                    for i in range(len(test)-n_pre-n_post):

                        ar = np.array([row[7] for row in test[i+n_pre:i+n_pre+n_post]])
                        B = ar.reshape(len(ar),-1)
                        tY.append(B)

                    testY = np.array(tY)

                    trainX_original = trainX.copy()
                    trainY_original = trainY.copy()
                    valX_original = valX.copy()
                    valY_original = valY.copy()
                    testX_original = testX.copy()
                    testY_original = testY.copy()

                    scalersX = {}

                    for i in range(trainX.shape[2]):
                        scalersX[i] = MinMaxScaler(feature_range=(0, 1))
                        trainX[:, :, i] = scalersX[i].fit_transform(trainX[:, :, i]) 

                    for i in range(valX.shape[2]):
                        valX[:, :, i] = scalersX[i].transform(valX[:, :, i]) 
                        
                    for i in range(testX.shape[2]):
                        testX[:, :, i] = scalersX[i].transform(testX[:, :, i])

                    scalersY = {}

                    for i in range(trainY.shape[2]):
                        scalersY[i] = MinMaxScaler(feature_range=(0, 1))
                        trainY[:, :, i] = scalersY[i].fit_transform(trainY[:, :, i]) 

                    for i in range(valY.shape[2]):
                        valY[:, :, i] = scalersY[i].transform(valY[:, :, i]) 
                        
                    for i in range(testY.shape[2]):
                        testY[:, :, i] = scalersY[i].transform(testY[:, :, i]) 

                    print('creating model...')
                    model = create_model(steps_before = n_pre, steps_after = n_post, cnn = cnn, feature_count = 8, units = U, layer = layer, ft = fil, intervals=True)

                    callbacks, path_checkpoint = callbacksFunction(model_name, dataset)
                    history = train_model(model, trainX, trainY, valX, valY, 512, 100, callbacks, True)

                    try:
                        model.load_weights(path_checkpoint)
                    except Exception as error:
                        print("Error trying to load checkpoint.")
                        print(error)

                    n_epochs = len(history.history['loss'])
                    times = callbacks[4].times 
                    Time_Epoch = Average(times) 
                    Total_Time = sum(times) 

                    Train_mae = min(history.history['mae'])
                    Train_mse = min(history.history['mse'])
                    Train_rmse = min(history.history['root_mean_squared_error'])

                    Validation_mae = min(history.history['val_mae'])
                    Validation_mse = min(history.history['val_mse'])
                    Validation_rmse = min(history.history['val_root_mean_squared_error'])

                    GlobalPredictions = []
                    GlobalPredictions_normalized = []


                    nan_array = np.empty((n_pre - 1))
                    nan_array.fill(np.nan)
                    nan_array2 = np.empty(n_post)
                    nan_array2.fill(np.nan)
                    ind = np.arange(n_pre + n_post)

                    for j in range (0,5):
    
                        print(j, end= ' ')
                        predict_ = model.predict(testX)
                        predict_original_ = predict_.copy()

                        for i in range(valY.shape[2]):
                            predict_[:, :, i] = scalersY[i].inverse_transform(predict_[:, :, i])
    
                        GlobalPredictions.append(predict_)
                        GlobalPredictions_normalized.append(predict_original_)
    
                    GlobalPredictions = np.array(GlobalPredictions)
                    GlobalPredictions_normalized = np.array(GlobalPredictions_normalized)


                    mean = np.mean(GlobalPredictions, axis=0)

                    predicted_mean_normalized = []
                    original_normalized = []

                    predicted_mean = []
                    original = []


                    for j in range (0, testX.shape[0]):
                        for i in range(j, testX.shape[0], testX.shape[0]):
                                    
                            forecasts_mean_normalized = np.concatenate((nan_array, testX[i, -1:, 7], predict_original_[i, :, 0]))
                            forecasts_mean = np.concatenate((nan_array, testX_original[i, -1:, 7], predict_[i, :, 0]))
                            
                            ground_truth_normalized = np.concatenate((nan_array, testX[i, -1:, 7], testY[i, :, 0]))
                            ground_truth = np.concatenate((nan_array, testX_original[i, -1:, 7], testY_original[i, :, 0]))
                            
                            predicted_mean_normalized.append((forecasts_mean_normalized[n_pre+4], forecasts_mean_normalized[n_pre+9], forecasts_mean_normalized[n_pre+14]))
                            predicted_mean.append((forecasts_mean[n_pre+4], forecasts_mean[n_pre+9], forecasts_mean[n_pre+14]))
                            
                            original_normalized.append((ground_truth_normalized[n_pre+4], ground_truth_normalized[n_pre+9], ground_truth_normalized[n_pre+14]))
                            original.append((ground_truth[n_pre+4], ground_truth[n_pre+9], ground_truth[n_pre+14]))




                    test_rmse_5, test_rmse_10, test_rmse_15, test_mse_5, test_mse_10, test_mse_15, test_mae_5, test_mae_10, test_mae_15 = printTestErrors(original, predicted_mean)
                    test_rmse_5_n, test_rmse_10_n, test_rmse_15_n, test_mse_5_n, test_mse_10_n, test_mse_15_n, test_mae_5_n, test_mae_10_n, test_mae_15_n = printTestErrorsNormalized(original_normalized, predicted_mean_normalized)

                    result = model.evaluate(x=testX, y=testY)
                    print("loss (test-set):", result)

                    with open('./outputs/Test_' + dataset + '.csv', 'a', newline='') as file:
                        fieldnames = ['Model', 
                                  'Validation_MSE', 'Validation_RMSE', 'Validation_MAE',
                                  'Test_MSE_n', 'Test_RMSE_n', 'Test_MAE_n',                  
                                  'Test_5_RMSE_n', 'Test_10_RMSE_n', 'Test_15_RMSE_n', 
                                  'Test_5_MSE_n', 'Test_10_MSE_n', 'Test_15_MSE_n', 
                                  'Test_5_MAE_n', 'Test_10_MAE_n', 'Test_15_MAE_n',
                                  
                                  'Test_5_RMSE', 'Test_10_RMSE', 'Test_15_RMSE', 
                                  'Test_5_MSE', 'Test_10_MSE', 'Test_15_MSE', 
                                  'Test_5_MAE', 'Test_10_MAE', 'Test_15_MAE']
                    
                        writer = csv.DictWriter(file, fieldnames=fieldnames)

                        writer.writeheader()

                    with open('./outputs/Test_' + dataset + '.csv', 'a', newline='') as file:
                        fieldnames = ['Model',
                                      'Validation_MSE', 'Validation_RMSE', 'Validation_MAE',
                                      'Test_MSE_n', 'Test_RMSE_n', 'Test_MAE_n',                  
                                      'Test_5_RMSE_n', 'Test_10_RMSE_n', 'Test_15_RMSE_n', 
                                      'Test_5_MSE_n', 'Test_10_MSE_n', 'Test_15_MSE_n', 
                                      'Test_5_MAE_n', 'Test_10_MAE_n', 'Test_15_MAE_n',
                                    
                                      'Test_5_RMSE', 'Test_10_RMSE', 'Test_15_RMSE', 
                                      'Test_5_MSE', 'Test_10_MSE', 'Test_15_MSE', 
                                      'Test_5_MAE', 'Test_10_MAE', 'Test_15_MAE']
                        
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                           
                        writer.writerow({'Model': model_name, 

                                         'Validation_MSE': Validation_mse, 
                                         'Validation_RMSE': Validation_rmse, 
                                         'Validation_MAE':Validation_mae , 

                                         
                                         'Test_MSE_n': result[2], 
                                         'Test_RMSE_n': result[3],
                                         'Test_MAE_n': result[1], 
                                         
                                         'Test_5_RMSE_n': test_rmse_5_n,
                                         'Test_10_RMSE_n': test_rmse_10_n,
                                         'Test_15_RMSE_n': test_rmse_15_n, 
                                         'Test_5_MSE_n': test_mse_5_n,
                                         'Test_10_MSE_n': test_mse_10_n,
                                         'Test_15_MSE_n': test_mse_15_n, 
                                         'Test_5_MAE_n': test_mae_5_n,
                                         'Test_10_MAE_n': test_mae_10_n,
                                         'Test_15_MAE_n': test_mae_15_n,
                                                         
                                         'Test_5_RMSE': test_rmse_5,
                                         'Test_10_RMSE': test_rmse_10,
                                         'Test_15_RMSE': test_rmse_15, 
                                         'Test_5_MSE': test_mse_5,
                                         'Test_10_MSE': test_mse_10,
                                         'Test_15_MSE': test_mse_15, 
                                         'Test_5_MAE': test_mae_5,
                                         'Test_10_MAE': test_mae_10,
                                         'Test_15_MAE': test_mae_15                 
                                                            })



Vanilla('GRU', dtr, dv, dtt, False)
Vanilla('LSTM', dtr, dv, dtt, False)

encoder_decoder('GRU', dtr, dv, dtt, False)
encoder_decoder('LSTM', dtr, dv, dtt, False)

encoder_decoder('GRU', dtr, dv, dtt, True)
encoder_decoder('LSTM', dtr, dv, dtt, True)







