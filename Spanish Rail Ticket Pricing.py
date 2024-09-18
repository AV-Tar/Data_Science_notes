#!/usr/bin/env python
# coding: utf-8

# # Пояснение к файлу

#         1. Файл создан в личных целях, чтобы собрать в одном месте наиболее часто используемый код в моих повседневных рабочих задачах
#         2. Периодически буду обновлять его по мере появления в моем личном багаже новых инструментов и методов работы с данными
#         3. Данный код не претендует на "идеальность", я не пытался "наводить красоту", здесь мне нужны просто рабочие методы, а уж под конкретный проект/задачу можно и красоту навести :)
#         4. Закомментированный код оставлен для того, чтобы показать разные способы реализации задачи
#         
# **P.S. опубликованный документ "как есть" лучше, чем "красота и гениальность", лежащая в "дальнем ящике", о которой никто не знает**

# Я периодически забываю синтаксис написания кода/метода, поэтому хочу сделать для себя один файл-шпаргалку, в который буду постепенно собирать весь необходимый для меня, чтобы не искать по разным файлам в разных проектах.
# 
# Для подобного документа-шпаргалки есть несколько полезных ресурсов с готовыми **Jupyter Notebook** для **Data Science**:
# 
# 1. Data Science Handbook (by Jake VanderPlas) — это книга с примерами кода и набором методов:
# - GitHub Jake VanderPlas
# https://github.com/jakevdp/PythonDataScienceHandbook
# 
# 2. Awesome Data Science Notebooks — репозиторий с набором Jupyter Notebook для различных аспектов Data Science:
# - Awesome DS Notebooks
# https://github.com/jupyter-naas/awesome-notebooks
# 
# 3. Автор многих книг по машинному обучению, больше 10 лет выкладывает код в репозитории
# https://github.com/rasbt/python-machine-learning-book/

# # Внешнее оформление тетрадки

# ## Темы Jupyter Notebook

# In[1]:


# !pip install jupyterthemes

## популярная темная тема
# jt -t chesterish

## Восстановить основную тему можно через
# jt -r


# ## Комментирование, выделение рамками

# Синий фон ячейки
# ```
# <div class = "alert alert-info" style="border-left: 7px solid blue">
# <b>Пояснение</b>
#     
# Текст ячейки
# </div>
# ```

# Зеленый фон ячейки
# ```
# <div style="border:solid Green 2px; padding: 40px">
# <div class="alert alert-success">
# <h2> Комментарий №_<a class="tocSkip"> </h2>
# 
# <font color='green'><b> Принято </b>  </font>
# 
# Текст
# ```

# Оранжевый фон ячейки
# 
# ```
# <div style="border:solid Orange 2px; padding: 40px">
# <div class="alert alert-warning">
#     <h2> Комментарий №_ <a class="tocSkip"> </h2>
#     
# <font color='Orange'><b>Некоторые рекомендации 💡:</b> </font>
#         
# Текст ячейки
# ```

# Красный фон ячейки
# ```
# <br/>
# <div style="border:solid Red 2px; padding: 40px">
#     <div class="alert alert-block alert-danger">
# <h2> Комментарий №_  <a class="tocSkip"></h2>
# 
# <font color='red'><b>На доработку ❌:</b>  </font>
#     
# Текст замечания
#     
#     
# </div>
# ```

# Варианты оформления рамок вокруг ячейки
# 
# ```
# голубая тонкая:
# <div style='border: 3px solid #67a0f549; padding: 20px'>
# 
# зеленая утолщенная:
# <div style='border:solid green 5px; padding: 20px'>
# ```

# ## Прогрессбар

# In[2]:


# # прогрессбар
# import time
# from tqdm import tqdm

# mylist = [1,2,3,4,5]

# for i in tqdm(mylist):
#     time.sleep(1)


# ## Добавление в тетрадку картинок по ссылке

# In[3]:


from IPython.display import Image # Библиотека для отображения картинок

display(Image(url='https://storage.googleapis.com/kaggle-datasets-images/188635/421394/be1365b1ef651ba7dc49f33588f008dc/dataset-cover.jpg?t=2019-05-09-21-51-02', 
              width = '') # width = 200 параметр задает ширину картинки, пустота - значение по умолчанию
       ) 
# display(Image(filename='Python_royal_35.JPG', width = 200)) # Локальный файл


# # Датасет Spanish Rail Tickets Pricing - Renfe

# Исходная ссылка на Кагл:
# 
# https://www.kaggle.com/datasets/thegurusteam/spanish-high-speed-rail-system-ticket-pricing/data

# ## Сведения об исходных данных

# Датасет содержит данные о стоимости билетов на поезда высокой скорости испанской железнодорожной системы **Renfe**. Он включает следующие параметры:
# 
# - Дата и время поездки: Указаны даты и время отправления и прибытия.
# - Информация о маршруте: Содержит станции отправления и назначения.
# - Класс обслуживания: Например, эконом или первый класс.
# - Длительность поездки: Время в пути.
# - Цена билета: Стоимость каждого билета, которая может варьироваться в зависимости от даты, времени и типа обслуживания.
# - Наличие мест: Указано, сколько мест было доступно на момент бронирования.
# 
# Этот датасет можно использовать для предсказания цен на билеты с помощью моделей машинного обучения, таких как DNN (глубокие нейронные сети) или XGBoost.

# - id - идентификатор
# - company - компания
# - origin - страна происхождения
# - destination - место назначения (страна прибытия)
# - departure - дата и время отправления
# - arrival - дата и время прибытия
# - duration - продолжительность поездки
# - vehicle_type - тип транспортного средства
# - vehicle_class - класс транспортного средства
# - price - стоимость билета

# ## Импорт библиотек

# Установка нехватающих библиотек

# In[4]:


# !pip install missingno


# In[5]:


# !pip install chart-studio


# In[6]:


# !pip install pykalman 


# In[7]:


# !pip install filterpy


# In[8]:


# !pip install phik


# Импорт установленных библиотек для дальнейшего анализа

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import seaborn as sns
import datetime


# прогрессбар
from tqdm import tqdm

from chart_studio import plotly
import plotly.graph_objs as go
import plotly.graph_objs as go
from plotly.offline import iplot
from chart_studio import plotly as py
from plotly.offline import init_notebook_mode, iplot

import phik
from phik.report import plot_correlation_matrix
from phik import report

# для анализа временных рядов
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import adfuller

# филтр Калмана
from filterpy.kalman import KalmanFilter

# игнорирование всплывающих уведомлений
import warnings
warnings.filterwarnings("ignore")


# # Чтение файла

# In[10]:


# путь к расположению файла
path = 'D://MyProjects//Spanish Rail Ticket Pricing - Renfe//thegurus-opendata-renfe-trips.csv'


# ## Стандартная загрузка всего датафрейма

# In[11]:


# # Загрузка всего датафрейма без прогрессбара
# df = pd.read_csv(path, index_col=0)


# ## Загрузка всего датафрейма с отображением прогрессбара

# In[12]:


# # Устанавливаем для tqdm режим автоматического обновления
# tqdm.pandas()

# # Чтение файла с отображением прогресса
# df = pd.read_csv(path, index_col=0, iterator=True, chunksize=10000000)
# # chunksize=10000000 - разделение на блоки при чтении

# # Прогрессбар при сборке в DataFrame
# df = pd.concat([chunk.progress_apply(lambda x: x) for chunk in tqdm(df, total=38)], ignore_index=True)


# ## Загрузка части датафрейма для быстрой отладки кода

# In[13]:


# Загрузка первых nrows строк датафрейма
df = pd.read_csv(path, index_col=0, nrows=1e6)

# index_col=0 - первый столбец будет исполтзоваться в качестве индекса
# nrows=10000 - чтение только первых 10000 строк


# ## Загрузка случайных 10 000 строк из датафрейма

# In[14]:


# # Определение количества строк в файле
# total_rows = sum(1 for _ in open(path)) - 1  # -1, так как есть строка заголовка

# # Случайным образом выбираются индексы строк, которые будуд загружаться
# skip_rows = sorted(np.random.choice(range(1, total_rows + 1), total_rows - 10000, replace=False))

# # Чтение файла, пропуская выбранные строки
# df = pd.read_csv(path, index_col=0, skiprows=skip_rows)


# In[15]:


df = df.sort_values(by='departure')


# In[16]:


# первые строки датафрейма
df.head(3)


# # Первичный анализ, предобработка данных

# ## Первичный анализ датафреймма

# In[17]:


# общая информация о датафрейме
df.info()


# In[18]:


# размер датафрейма (строк, столбцов)
df.shape


# ### Функция для вывода общей информации по датафрейму

# In[19]:


def df_info(df):
    '''
    функция для вывода основных показателей датафреймов
    
    '''
    print("-"*100)
    print('Общая информамия о датафрейме:')
    print("-"*100)
    print(df.info())
    
    print("-"*100)
    print('Первые 5 строк таблицы')
    print("-"*100)
    display(df.head(5))
    
    print("-"*100)
    print('Количество пропусков')
    print("-"*100)
    print(df.isnull().sum().sort_values(ascending=False))
    
    print("-"*100)
    print('Статистические данные датафрейма')
    print("-"*100)
    display(np.round(df.describe(), 2).T)
    
    print("-"*100)
    print('Количество дубликатов:')
    print("-"*100)
    display(df.duplicated().sum())


# In[20]:


df_info(df)


# ## Количество пропусков в датафрейме, заполнение их

# In[21]:


# количество пропусков в столбцах
df.isnull().sum()


# In[22]:


# Процент пропусков по столбцу
(df.isnull().sum() / df.shape[0]) * 100


# ### Визуализация пропусков

# In[23]:


# визуализация первых 1000 строк пропущенных данных в датафрейме
sns.heatmap(df[:1000].isna().transpose(), cmap="crest", cbar_kws={'label': 'Пропуски в данных'})
plt.show()


# ## Уникальные значения в столбце

# In[24]:


df['origin'].unique()


# ## Удаление столбцов (признаков) с пропусками или ненужных столбцов

# In[25]:


# Удаление ненужных столбцов
df = df.drop(columns=['seats','meta', 'insert_date'])


# In[26]:


df.head(2)


# ## Удаление пропусков

# In[27]:


# удаление пропусков в столбце price
df.dropna(subset=['price'], inplace=True)


# Можно заполнить пропуски средним значением по столбцу

# In[28]:


# # заполнение пропусков средним значением
# df['price'].fillna(df['price'].mean(), inplace=True)


# In[29]:


# # заполнение модой нескольких столбцов
# columns = ['train_class','fare']
# for c in columns:
#     df[c].fillna(df[c].mode()[0], inplace=True)


# Или полностью удалить пропуски во всем датафрейме "одним махом"

# In[30]:


# # полное удаление всех пропусков
# df.dropna(inplace=True)


# In[31]:


# # сброс индекса
# df = df.reset_index()


# In[32]:


# проверка количества пропусков в столбцах
df.isnull().sum()


# ## Переименование названий столбцов

# In[33]:


# словарь для переиенования стобцов в виде "старое название - новое название"
column_dict = {
    "departure" : "start_date",
    "arrival" : "end_date",
    "vehicle_type" : "train_type",
    "vehicle_class" : "train_class"
}


# In[34]:


# df = df.rename(columns = column_dict)


# ## Создание дополнительных столбцов (признаков)

# Создадание столбца времени отправления из СТРОКОВЫХ значений столбца **departure**

# In[35]:


# здесь для каждой строки берутся значения с 11 по 13 элеммент (т.к. именно они являются часами отправки)
#  из столбца departure, результат записывается в столбец departure_hour
df['departure_hour'] = df.apply(lambda x: int(x['departure'][11:13]), axis=1)


# In[36]:


# преобразование значений столбца в формат datetime
datetimeFormat = '%Y-%m-%d %H:%M:%S'
def fun(a,b):
    diff = datetime.datetime.strptime(b, datetimeFormat)- datetime.datetime.strptime(a, datetimeFormat)
    return(diff.seconds/3600.0)


# In[37]:


df['travel_time_in_hrs'] = df.apply(lambda x: fun(x['departure'], x['arrival']), axis=1) 


# In[38]:


# просмотр результата
df.head(2)


# ## Преобразование значений столбцов в формат даты (datetime)

# In[39]:


# Изменение столбцов даты на столбцы даты и времени
for i in ['departure','arrival']:
    df[i] = pd.to_datetime(df[i])


# In[40]:


# проверка результата преобразования
df.info()


# ## Выделение отдельных признаков из столбца с датой

# Поскольку дата указана в строковом формате, ее можно разложить на разные столбцы: год, месяц, число и день недели:

# In[41]:


for col in ['departure', 'arrival']:
    date_col = pd.to_datetime(df[col])
    df[col + '_hour'] = date_col.dt.hour
    df[col + '_minute'] = date_col.dt.minute
    df[col + '_second'] = date_col.dt.second
    df[col + '_weekday'] = date_col.dt.day_of_week
    df[col + '_day'] = date_col.dt.day
    df[col + '_month'] = date_col.dt.month
    df[col + '_year'] = date_col.dt.year
    
#     del df[col]


# In[42]:


df.head(2)


# ## Изменение типа данных в столбцах

# In[43]:


# список столбцов для преобразования в числовой тип данных из строкового
list_columns_num = ['duration', 'price']

# список столбцов для преобразования тип данных - datetime из строкового
list_columns_date = ['departure', 'arrival']


# In[44]:


def data_convert (dataframe, list_columns_num, list_columns_date):
    '''
    функция преобразования типа данных в числовые и формат даты
    
    '''
    for i in list_columns_num:
        dataframe[i] = dataframe[i].astype(float)
        
    for j in list_columns_date:
        dataframe[j] = pd.to_datetime(dataframe[j])
       
    return dataframe


# In[45]:


# # вызов функции для преобразования типов данных в столбцах
# df = data_convert(df, list_columns_num, list_columns_date)


# ## Выборка нужных столбцов из исходных таблиц

# In[46]:


# список названий столбцов в удобном порядке
list_columns = ['origin',
                'departure',
                'price'
               ]


# In[47]:


def df_select_columns (dataframe, list_columns):
    '''
    функция выбирает столбцы списка list_columns из dataframe,
    возвращается датафрейм с выбранными столбцами
    
    '''  
    dataframe = dataframe[list_columns]
    
    return dataframe


# In[48]:


# вызов функции для выборки столбцов по листу ХХХ
df_select = df_select_columns (df, list_columns)


# In[49]:


# просмотр результата выборки
df_select.head(2)


# ## Формирование выборки по нескольким признакам

# In[50]:


# таблица - выборка по двум признакам:
# выбираются все строки со значением MADRID столбца origin и значениями SEVILLA столбца destination
df1 = df[(df['origin']=="MADRID") & (df['destination']=="SEVILLA")]
df1.head(3)


# In[51]:


f, ax = plt.subplots(figsize=(12, 2))
ax = sns.barplot(x="vehicle_type", y="price", data=df1)
plt.show()


# In[52]:


f, ax = plt.subplots(figsize=(12, 4))
ax = sns.barplot(x="vehicle_type", y="price", data=df1, palette="coolwarm")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel('Тип транспортного средства')
ax.set_ylabel('Цена')
ax.set_title('Цены на транспортные средства по типам')

# Отображение значений на барах
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Ограничение диапазона Y (по желанию)
ax.set_ylim(0, df1['price'].max() * 1.1)

plt.tight_layout()
plt.show()


# ## Функция выборки таблицы с нужными значениями

# In[53]:


def data_selection(dataframe, column, name_unique):
    '''
    функция для выборки данных из исходного датафрейма
    по уникальным товарам name_unique в столбце column,
    возвращает сформированную выборку в новый датафрейм df_unique
    и название введенного уникального наименования name_unique  
    
    '''
    df_unique = dataframe[(dataframe[column] == name_unique)]
    
    return df_unique


# In[54]:


# вызов функции выборки по уникальному значению
column_select = 'vehicle_type'
name_unique = 'AVE'

df_uniq = data_selection(df, column_select, name_unique)


# In[55]:


df_uniq.head(2)


# ## Преобразование значений столбца в другие единицы измерения

# In[56]:


df_uniq['price'] = df_uniq['price'] / 1000


# In[57]:


df_uniq.head(2)


# ## Сортировка таблицы по значениям столбца

# In[58]:


# сортировка таблицы по столбцу 'price'
df_sorted = df_uniq.sort_values(by='price')


# In[59]:


df_sorted.head()


# In[60]:


# гистограмма распределения значений
df_sorted['price'].hist(figsize=(10, 2), color='green', alpha=0.3, label='price') 
plt.title('Распределение записей в таблице в зависимости от цены price')
plt.xlabel('price')
plt.ylabel('количество строк')
plt.legend()
plt.grid(True)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
plt.show()


# ## Группировка по странам origin, суммирование price

# In[61]:


df_groupby = df.groupby('origin')['price'].sum()


# In[62]:


df_groupby


# # EDA - исследовательский анализ данных

# ## Числовая аналитика

# ### Уникальные значения в столбце

# In[63]:


# количество уникальных значений в столбце vehicle_type
print(len(df['vehicle_type'].unique()))


# In[64]:


# первые 5 уникальных значений столбца vehicle_type
unique_vehicle_type = df['vehicle_type'].unique()[:5]
unique_vehicle_type


# In[65]:


# количество записей в столбце destination
df['destination'].count()


# ### Количество уникальных значений в каждом столбце

# In[66]:


for col in df.columns:
    print(col, ":", df[col].unique().shape[0])


# ### Дата начала и конца датафрейма

# Проверка наличия минимальных и максимальных дат в столбцах даты и времени

# In[67]:


print(f" started date minimum value {df.departure.min()}")
print(f" started date maximum value {df.departure.max()}")


# ### Статистические показатели

# In[68]:


# всего датафрейма
print("Основные статистические показатели:\n")
df.describe()


# In[69]:


# отдельного столбца price
stat_column = 'price'
print(f'Статистические показатели столбца {stat_column}:')
df[stat_column].describe()


# ## Графическая аналитика

# ### Распределение значений выходной / рабочий день

# In[70]:


def make_features(data, MAX_LAG, ROLLING_MEAN_SIZE, column):
    '''функция генерации новых признаков'''
    
    num_col = []
    
    # день недели/месяц
    data['день_недели'] = data.index.dayofweek
    data['месяц'] = data.index.month
    
    # бинарный признак для выходных дней (суббота и воскресенье)
    data['выходной'] = data.index.dayofweek.isin([5, 6]).astype(int)
    
    # Определение времени года по месяцам
    data['сезон'] = data['месяц'].apply(lambda x: 'Весна' if x in [2, 3, 4, 5] else
                                                  'Лето' if x in [6, 7, 8] else
                                                  'Осень' if x in [9, 10, 11] else
                                                  'Зима')
    
    # Создание словаря для замены чисел на названия дней недели
    days_mapping = {
        0: 'пн',
        1: 'вт',
        2: 'ср',
        3: 'чт',
        4: 'пт',
        5: 'сб',
        6: 'вс'
    }

    # Применение замены к столбцу 'день_недели'
    data['день_недели'] = data['день_недели'].map(days_mapping)

    if ROLLING_MEAN_SIZE > 0:     
        data['rolling_mean'] = data[column].shift().rolling(ROLLING_MEAN_SIZE).mean()
        num_col.append('rolling_mean')
        
    if MAX_LAG > 0:
        for lag in range(1, MAX_LAG + 1):
            data['lag_{}'.format(lag)] = data[column].shift(lag)
            num_col.append('lag_{}'.format(lag))            
   
    data.dropna(inplace=True)
    
    return data, num_col


# **выходной** - флаг на выходной день:
# 
#  - 1 - выходной
#  - 0 - будни
# 
# **rolling_mean** - скользящее среднее
# 
# **lag** - сдвиг на количество дней

# In[71]:


# ширина окна сдвига, дней
MAX_LAG = 10

# ширина окна скользящего среднего, дней
ROLLING_MEAN_SIZE = 10


# In[72]:


# # дополнительные фичи
# df_features_Up = df_Unique_product.copy(deep=True)
# df_features_Up, NUM_FEATURES = make_features(df_Unique_product, MAX_LAG, ROLLING_MEAN_SIZE, 'количество')


# In[73]:


# # Подготовка данных для диаграммы
# labels = df_features_Up['выходной'].value_counts().index
# sizes = df_features_Up['выходной'].value_counts().values
# labels = df_features_Up['выходной'].value_counts()


# In[74]:


# # Создание круговой диаграммы
# plt.figure(figsize=(4, 4))
# plt.pie(df_features_Up.groupby('выходной')['количество'].sum(), labels=labels, autopct='%1.1f%%', startangle=45)
# plt.title('Распределение выходных')
# plt.show()


# In[75]:


# sum_weekend = df_features_Up.groupby('выходной')['количество'].sum()
# sum_weekend


# In[76]:


df['origin'].unique()


# In[77]:


df_filtered = df[df['origin'].isin(["MADRID", "SEVILLA"])]


# In[78]:


# список цветов
colors = ["#FF5733", "#33C4FF", "#FF33B8", "#3380FF"]

# Настройка графика
plt.figure(figsize=(12, 6), facecolor="#f0f0f0")

# Построение временного ряда
time_series = sns.lineplot(
    x='departure', 
    y='price', 
    data = df_filtered, 
    hue='origin', 
    palette=colors
)

# Настройка формата оси X для отображения дат (если departure содержит даты)
time_series.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
time_series.set_title("Изменение цены за билет по времени и странам", fontsize=16)
time_series.set_ylabel("Цена", fontsize=12)
time_series.set_xlabel("Дата вылета", fontsize=12)

# Перемещение легенды за пределы графика
time_series.legend(title='Страна отправления', loc='upper left', bbox_to_anchor=(1, 1))

# поворот отображения подписей
plt.xticks(rotation=45)
plt.tight_layout()

# Отображение сетки и графика
plt.grid()
plt.show()


# In[79]:


# Создание фигуры и оси
fig, ax = plt.subplots(figsize=(12, 5))

# Построение боксплота на оси `ax`
df[:10000].boxplot(by='destination', column=['price'], grid=False, ax=ax, patch_artist=True)

# Добавление сетки
ax.grid(True, linestyle='--', linewidth=0.5)

# Добавление заголовков и подписей осей
ax.set_title('Распределение цен на билеты по пунктам назначения', fontsize=14)
ax.set_xlabel('Пункт назначения', fontsize=12)
ax.set_ylabel('Цена билета', fontsize=12)

# Поворот меток на оси X для лучшей читаемости
plt.xticks(rotation=45)

# Улучшенное размещение графика
plt.tight_layout()

# Отображение графика
plt.show()


# ### Гистограммы распределения значений

# In[80]:


# гистограмма распределения значений
df['price'].hist(figsize=(10, 2), color='green', alpha=0.5, label='price') 
plt.title('Распределение записей в таблице в зависимости от цены price')
plt.xlabel('price')
plt.ylabel('количество строк')
plt.legend()
plt.grid(True)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
plt.show()


# In[81]:


# # Уменьшаем разрешение графика (DPI)
# fig, ax = plt.subplots(figsize=(12, 5), dpi=200)

# ax.plot(df.index, df['price'], label='значение', color='green', alpha=0.5)
# ax.set_title('Динамика изменения price')
# ax.set_xlabel('дата')
# ax.set_ylabel('цена')

# # Устанавливаем формат и частоту меток оси X
# locator = AutoDateLocator()
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

# plt.xticks(rotation=90)
# ax.legend()
# plt.grid(True)
# plt.minorticks_on()
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
# plt.tight_layout()
# plt.show()


# ### !!! Диаграммы

# In[82]:


plt.figure(figsize=(11, 1))
sns.boxplot(x=df['price'])
# plt.xlim(0, 0.8)
plt.grid(True)
plt.show()


# In[83]:


cnt_ = df['vehicle_class'].value_counts()
cnt_ = cnt_.sort_index() 
fig = {
  "data": [
    {
      "values": cnt_.values,
      "labels": cnt_.index,
      "domain": {"x": [0, .5]},
      "name": "Percentage of journeys started and ended on same date",
      "hoverinfo":"label+percent+name",
      "hole": .5, # процент внутреннего круга
      "type": "pie"
    },],
  "layout": {
        "title":"Percentage of journeys started and ended on same date",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
             "text": "Pie Chart",
                "x": 0.50,
                "y": 1
            },
        ]
    }
}
iplot(fig)
cnt_


# In[84]:


cnt_srs = df['fare'].value_counts()
trace1 = go.Bar(
                x = cnt_srs.index,
                y = cnt_srs.values,
                marker = dict(color = 'rgba(0, 255, 200, 0.8)',
                             line=dict(color='rgb(0,0,0)',width=0.2)),
                text = cnt_srs.index)

data = [trace1]
layout = go.Layout(title = 'Распределение Fare')
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# С логарифмической шкалой

# In[85]:


cnt_srs = df['fare'].value_counts()
trace1 = go.Bar(
                x=cnt_srs.index,
                y=cnt_srs.values,
                marker=dict(color='rgba(0, 200, 150, 0.8)',  # Более мягкий цвет
                            line=dict(color='rgb(0,0,0)', width=0.5)),
                text=cnt_srs.values,
                hoverinfo='x+y+text',
                width=0.8  # Увеличение ширины столбцов
)

data = [trace1]
layout = go.Layout(
    title='Распределение Fare',
    xaxis_title='Fare',
    yaxis_title='Количество',
    yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.5)'),
    yaxis_type='log'  # Логарифмическая шкала
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[86]:


cnt_srs = df['fare'].value_counts()

trace1 = go.Bar(
    x=cnt_srs.values,
    y=cnt_srs.index,
    orientation='h',
    marker=dict(color='rgba(155, 0, 100, 0.8)',
                line=dict(color='rgb(0,0,0)', width=0.2)),
    text=cnt_srs.index
)

data = [trace1]

layout = go.Layout(
    title='Название диаграммы',
    width=1000,   # Устанавливаем ширину
    height=400    # Устанавливаем высоту
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[87]:


cnt_srs = df['duration'].value_counts()

# Основной график с маркерами
trace1 = go.Scatter(
    x=cnt_srs.index,
    y=cnt_srs.values,
    mode="markers",
    marker=dict(size=10, color='rgba(100, 35, 55, 0.8)', line=dict(width=2, color='DarkSlateGrey')),
    text=cnt_srs.values,  # Отображение значений при наведении
    hoverinfo='x+y+text'
)

# Данные для графика
data = [trace1]

# Настройка оформления
layout = dict(
    title='Распределение продолжительности поездок',
    xaxis=dict(title='Продолжительность (часы)', ticklen=5, zeroline=False, showgrid=True),
    yaxis=dict(title='Количество поездок', showgrid=True)
)

fig = dict(data = data, layout = layout)
iplot(fig)


# In[88]:


trace1 = go.Box(
    y=df.price[:10000,],
    name = 'Box plot of average travelling time in minutes only 50k observations',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
data = [trace1]
iplot(data)


# In[89]:


df.groupby(['origin','vehicle_type'])['vehicle_type'].count()


# In[90]:


df.head(2)


# In[91]:


cnt_srs = df.groupby('vehicle_type')['price'].agg(['mean'])
cnt_srs.columns = ["mean"]
cnt_srs['vehicle_type'] = cnt_srs.index

data = [
    {
        'x': cnt_srs['vehicle_type'],
        'y': cnt_srs['mean'],
        'mode': 'markers+text',
        'text' : cnt_srs['vehicle_type'],
        'textposition' : 'bottom center',
        'marker': {
            'color': "#f27da6",
            'size': 15,
            'opacity': 0.9
        }
    }
]

layout = go.Layout(title="Average fare prices according to Train type", 
                   xaxis=dict(title='Train type'),
                   yaxis=dict(title='Average price(Euros)')
                  )
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename='scatter0')


# Количество людей, выходящих на посадку с разных станций

# In[92]:


fig, ax = plt.subplots(figsize=(12, 3))
ax = sns.countplot(df['origin'])
plt.show()


# Распределение цен на билеты

# In[93]:


f, ax = plt.subplots(figsize=(12, 4))
ax = sns.distplot(df['price'][500:1000], rug=True)
plt.show();


# In[94]:


import seaborn as sns
import matplotlib.pyplot as plt

# Создаем фигуру и ось
f, ax = plt.subplots(figsize=(12, 4))

# Строим гистограмму с KDE
sns.histplot(
    df['price'][500:1000], kde=True, bins=30, ax=ax,
    color='orange', kde_kws={'bw_adjust': 0.5}
)

# Отдельно строим rugplot
sns.rugplot(
    df['price'][500:1000], ax=ax, color='green'
)

# Добавляем заголовок, подписи осей и сетку
ax.set_title('Распределение цен (для строк с 500 до 1000)')
ax.set_xlabel('Цена')
ax.set_ylabel('Частота')
ax.grid(True)

# Отображаем график
plt.show()


# In[95]:


f, ax = plt.subplots(figsize=(12, 3))
ax = sns.boxplot(x='vehicle_class', y='price', data=df)
plt.show()


# In[96]:


plt.figure(figsize=(12, 4))
sns.scatterplot(data=df[df['destination'] != 'MADRID'].sample(1000), x='duration',
y='price', hue='destination');
plt.title('График цен и времени отправления с указанием пункта назначения (за исключением Мадрида)');
plt.xticks(range(24))
plt.show()


# В этом случае, если мы можем наблюдать, как постоянно, независимо от времени суток, пункт назначения с самым высоким тарифом-Барселона, при этом пиковые значения сохраняются в вечерние часы, в то время как, напротив, самые низкие значения обычно наблюдаются для поездов, следующих в Валенсию.

# In[97]:


# Гистограмма с настройками
sns.displot(
    df[df['destination'] == 'SEVILLA']['price'].dropna(), 
    bins=64, 
    kde=True,  # Включение плотности KDE
    color='skyblue',  # Установка цвета
    height=5, aspect=1.9 # Размер графика
)

# Добавление заголовка и подписей осей
plt.title('Гистограмма цен на билеты в Севилью', fontsize=16)
plt.xlabel('Цена билета', fontsize=12)
plt.ylabel('Количество билетов', fontsize=12)

# Отображение сетки и графика
plt.grid(True)
plt.show()


# На приведенной выше диаграмме мы видим, что самый обычный тариф на поезда, следующие в Севилью, составляет чуть менее 80 евро, а большинство билетов, как правило, составляют 45-75 евро, при этом тарифы за пределами этих диапазонов встречаются гораздо реже.

# In[98]:


print(df['destination'].value_counts())


# In[99]:


# Размер фигуры
plt.figure(figsize=(10, 5))

# Строим барчарт с горизонтальными столбцами, отсортированный по убыванию
df['destination'].value_counts().sort_values().plot(kind='barh', color='skyblue')

# Добавляем заголовок и подписи осей
plt.title('Частота билетов в зависимости от пункта назначения', fontsize=16)
plt.xlabel('Количество билетов', fontsize=12)
plt.ylabel('Пункт назначения', fontsize=12)

# Добавляем сетку для лучшей читаемости
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Отображаем график
plt.show()


# In[100]:


# Размер фигуры
plt.figure(figsize=(11, 5))

# Строим countplot, отсортированный по убыванию
sns.countplot(
    data=df, 
    y='destination',  # Используем горизонтальный график
    order=df['destination'].value_counts().index,  # Сортировка по убыванию
    palette='coolwarm'  # Приятная цветовая палитра
)

# Заголовок и подписи осей
plt.title('Частота билетов по пунктам назначения', fontsize=16)
plt.xlabel('Количество билетов', fontsize=12)
plt.ylabel('Пункт назначения', fontsize=12)

# Добавление значений на столбцы
for index, value in enumerate(df['destination'].value_counts()):
    plt.text(value, index, str(value), va='center', ha='left', fontsize=10)

# Сетка по оси X для лучшей читаемости
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Отображение графика
plt.show()


# In[101]:


plt.figure(figsize=(15, 6))
df[:10000].groupby('fare')['destination'].value_counts().plot(kind='bar');
plt.title('Частота билетов в зависимости от пункта назначения и тарифа')
plt.show()


# In[102]:


# Размер фигуры
plt.figure(figsize=(12, 6))

# Сгруппируем данные по тарифам и пунктам назначения
fare_destination_counts = df[:10000].groupby(['fare', 'destination']).size().unstack()

# Построение stacked bar графика
fare_destination_counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')

# Заголовок и подписи осей
plt.title('Частота билетов в зависимости от пункта назначения и тарифа', fontsize=16)
plt.xlabel('Тариф', fontsize=12)
plt.ylabel('Количество билетов', fontsize=12)

# Поворот меток по оси X
plt.xticks(rotation=45)

# Легенда
plt.legend(title='Пункт назначения', bbox_to_anchor=(1.05, 1), loc='upper left')

# Сетка по оси Y
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Отображение графика
plt.tight_layout()
plt.show();


# In[103]:


# Размер графика
plt.figure(figsize=(12, 5))

# Строим график с countplot с разбивкой по тарифам
sns.countplot(x='destination', data=df[:10000], hue='fare', palette='Set2')

# Заголовок и подписи осей
plt.title('Частота билетов по пунктам назначения и тарифам', fontsize=16)
plt.xlabel('Пункт назначения', fontsize=12)
plt.ylabel('Количество билетов', fontsize=12)

# Поворот меток оси X для лучшей читаемости
plt.xticks(rotation=45, ha='right')

# Легенда вынесена за пределы графика
plt.legend(title='Тариф', bbox_to_anchor=(1.05, 1), loc='upper left')

# Сетка для оси Y
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Обрезаем пространство вокруг графика для лучшего отображения
plt.tight_layout()

# Отображение графика
plt.show()


# In[104]:


df.groupby('destination')['fare'].value_counts()


# In[105]:


dataFiltr = df.query("fare not in ['Promo', 'Flexible']")


# In[106]:


dataFiltr['fare'].unique()


# In[107]:


dataSample = df.sample(100)
dataSample.head(2)


# ### Корреляция данных

# In[108]:


df_filtered


# In[109]:


# Выбираем только числовые столбцы
numeric_df = df.select_dtypes(include=[np.number])

# Вычисляем корреляцию
corr = numeric_df.corr()

# Создаем маску для верхнего треугольника
mask = np.triu(np.ones_like(corr, dtype=bool))

# Размер графика
plt.figure(figsize=(10, 8))

# Отображаем корреляционную матрицу
sns.heatmap(corr, annot=True, mask=mask, cmap="coolwarm", 
            vmin=-1, vmax=1, center=0, 
            linewidths=0.5, linecolor='gray',
            fmt='.2f', annot_kws={"size": 10})

# Заголовок и подписи осей
plt.title('Корреляционная матрица числовых данных', fontsize=16)
plt.xlabel('Переменные', fontsize=12)
plt.ylabel('Переменные', fontsize=12)

# Отображение графика
plt.tight_layout()
plt.show()


# In[110]:


# Построение таблицы (матрицы) корреляции методом phik
phik_view = df[:1000].phik_matrix()
phik_view_rounded = phik_view.round(2)


# In[111]:


phik_view_rounded


# In[112]:


# Графическое представление матрицы корреляции методом phik
plt.figure(figsize=(15, 8))
sns.heatmap(phik_view_rounded, 
            annot=True, 
            cmap="crest", 
            vmin=0, 
            vmax=1, 
            center=0.5,  # Поставим центр на 0.5 для наглядности
            fmt='.2f',  # Формат для аннотаций
            linewidths=0.5, 
            linecolor='gray', 
            cbar_kws={'shrink': 0.8, 'label': 'Phik Correlation'},  # Настройка цветовой шкалы
            annot_kws={"size": 10})

# Заголовок и подписи осей
plt.title("Корреляция признаков методом phik", fontsize=16)
plt.xlabel('Признаки', fontsize=12)
plt.ylabel('Признаки', fontsize=12)

# Плотное размещение элементов
plt.tight_layout()
plt.show()


# ### Агрегация данных: один признак под разными углами

# Например:
# - цена билета с выборкой по одной стране в зависимости от дат
# - цена билета за выбранный год
# - средня цена билета по разным странам
# - максимум, минимум по разным странам

# In[113]:


df2 = df[(df['origin']=="MADRID") & (df['destination']=="BARCELONA")]
df2.head(3)


# # Аналитика временного ряда

# ## Создание временного ряда

# In[114]:


# Создаем новый датафрейм с выбранными столбцами и устанавливаем 'departure' как индекс
selected_columns = ['price', 'duration']  # выборка нескольких столбцов датафрейма
df_ts = df.set_index('departure')[selected_columns]


# In[115]:


df_ts.head()


# Либо в таком варианте, дата отдельным столбцом, не в индексе

# In[116]:


# Создаем новый датафрейм с выбранными столбцами и устанавливаем 'departure' как индекс
selected_columns = ['departure', 'price', 'duration']  # выборка нескольких столбцов датафрейма
df_ts = df[selected_columns]


# In[117]:


df_ts.head()


# In[118]:


# Извлекаем только дату (без времени)
df_ts['date'] = df_ts['departure'].dt.date


# In[119]:


# Группируем по дате и суммируем значения в столбце 'price'
grouped_df = df_ts.groupby('date')['price'].sum().reset_index()


# In[120]:


grouped_df.head(10)


# ## Проверка временного ряда на последовательность дат

# In[121]:


# # Создаем новый датафрейм с выбранными столбцами и устанавливаем 'departure' как индекс
# selected_columns = ['price', 'duration', 'departure_hour']  # выборка нескольких столбцов датафрейма
# df_ts = df.set_index('departure')[selected_columns]


# In[122]:


# # Проверяем, что даты в индексе монотонно возрастают
# if df_ts.index.is_monotonic_increasing:
#     print("Даты идут по возрастанию.")
# else:
#     print("Даты не идут по возрастанию.")


# In[123]:


# # Проверяем на пропуски
# full_range = pd.date_range(start=df_ts.index.min(), end=df_ts.index.max(), freq='D')
# missing_dates = full_range.difference(df_ts.index)

# if len(missing_dates) > 0:
#     print("Пропущенные даты:", missing_dates)
# else:
#     print("Даты последовательны, пропусков нет.")


# In[124]:


# Проверяем, что даты в индексе монотонно возрастают
if grouped_df['date'].is_monotonic_increasing:
    print("Даты идут по возрастанию.")
else:
    print("Даты не идут по возрастанию.")


# In[125]:


# Проверяем на пропуски
full_range = pd.date_range(start=grouped_df['date'].min(), end=grouped_df['date'].max(), freq='D')
missing_dates = full_range.difference(grouped_df['date'])

if len(missing_dates) > 0:
    print("Пропущенные даты:", missing_dates)
else:
    print("Даты последовательны, пропусков нет.")


# ## Восстановление временного ряда, если даты непоследовательны

# In[126]:


def time_series_recovery(time_series):
    '''
    функция для создания восстановленного (равномерного) временного ряда,
    т.е. добавляются пропущенные даты, в добавленных строках "Количество" заполняются нулями
    
    '''
    time_series_recovery = time_series.reindex(
        pd.date_range(start=time_series.index.min(),
                      end=time_series.index.max(), freq='D'), fill_value=0) 
    
    return time_series_recovery


# In[127]:


df_ts_2 = grouped_df.set_index('date')


# In[128]:


df_ts_2[:5]


# In[129]:


df_ts_2_groupby = df_ts.groupby('departure')['price'].sum()
df_ts_2_groupby.head(10)


# In[130]:


# вызов функции для создания восстановленного (равномерого) временного ряда
ts_recovery = time_series_recovery(df_ts_2)
ts_recovery.head(10)


# ## Проверка временного ряда на стационарность

# In[131]:


# Проверка стационарности временного ряда
result = adfuller(grouped_df['price'])
print("\nADF статистика:", result[0])
print("p-value:", result[1])
print("Критические значения:")
for key, value in result[4].items():
    print(f"\t{key}: {value}")

if result[1] < 0.05:
    print("\nРяд стационарен (отвергается гипотеза о нестационарности)")
else:
    print("\nРяд нестационарен (не удается отвергнуть гипотезу о нестационарности)")


# ## Сезонная декомпозиция

# In[132]:


df_ts_resemp = df_ts.copy()


# In[133]:


df_ts_resemp = df_ts_resemp.set_index('departure')


# In[134]:


df_ts_resemp = df_ts_resemp['price']


# In[135]:


df_ts_resemp


# In[136]:


# измененим интервал значений временного ряда на часовой таймфрем
df_ts_resemp_H = df_ts_resemp.resample('1H').sum()
print(f'Количество строк после ресемплинга на таймфрейм в 1ч: {df_ts_resemp_H.shape[0]}')


# In[137]:


print('Начало временного интервала:', df_ts_resemp_H.index[0])
print('Конец временного интервала:', df_ts_resemp_H.index[-1])
print('Общее количество дней:',df_ts_resemp_H.index.max() - df_ts_resemp_H.index.min())


# In[138]:


df_ts_resemp_H = df_ts_resemp.copy()
df_ts_resemp_H = df_ts_resemp_H.resample('1H').sum()

df_ts_resemp_D = df_ts_resemp.copy()
df_ts_resemp_D = df_ts_resemp_D.resample('1D').sum()

df_ts_resemp_W = df_ts_resemp.copy()
df_ts_resemp_W = df_ts_resemp_W.resample('1W').sum()


# In[139]:


# print(df_ts_resemp_D.index.is_monotonic)


# In[140]:


# общий график за весь промежуток времени
df_ts_resemp_H.plot(figsize=(12, 3))
plt.legend()
plt.title('Изменение price во времени')
plt.xlabel('Временной интервал')
plt.ylabel('price')
plt.show;


# In[141]:


# график заказов такси за 10 дней
plt.figure(figsize=(11, 3))
df_ts_resemp_H['2019-05-15':'2019-05-25'].plot(figsize=(12, 3))
plt.legend()
plt.title('Количество заказов на временном интервале: 10 дней')
plt.xlabel('Временной интервал')
plt.ylabel('Количество заказов')
plt.show;


# ## Скользящие средние

# In[142]:


# построим скользящие средние с размером окна = 24ч
plt.figure(figsize=(12, 5))
plt.plot(df_ts_resemp_H, label = 'число заказов в час')
plt.plot(df_ts_resemp_H.rolling(24).mean(), label = 'скользящее среднее за сутки')
plt.plot(df_ts_resemp_H.rolling(24).std(), label = 'скользящее стандартное отклонение', color='green')

plt.legend()
plt.title('Скользящее среднее')
plt.xlabel('Временной интервал')
plt.ylabel('Количество заказов')
plt.show;


# ## Тренды и сезонность
# 

# In[143]:


# разделим на тренды функцией seasonal_decompose
decomposed_h = seasonal_decompose(df_ts_resemp_H)

plt.figure(figsize=(11, 9))

plt.subplot(311)
# Чтобы график корректно отобразился,
# указываем его оси ax, равными plt.gca() (англ. get current axis, получить текущие оси)
decomposed_h.trend.plot(ax=plt.gca())
plt.title('Trend')

plt.subplot(312)
decomposed_h.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')

plt.subplot(313)
decomposed_h.resid.plot(ax=plt.gca())
plt.title('Residuals')

plt.tight_layout()


# In[144]:


decomp_df = seasonal_decompose(df_ts_resemp_H, period=30)  # , model='additive', period=30 Период сезонности (можно изменить)
trend = decomp_df.trend
seasonal = decomp_df.seasonal
residual = decomp_df.resid


# In[145]:


decomp_df


# In[146]:


plt.figure(figsize=(10, 8))

plt.subplot(411)
plt.plot(df_ts_resemp_H, color = 'green', label='price')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')

plt.tight_layout()


# In[147]:


# функции автокорреляции и частичной автокорреляции
fig, ax = plt.subplots(2, figsize=(12, 8))
plot_acf(df_ts_resemp_H, lags=90, ax=ax[0])
plot_pacf(df_ts_resemp_H, lags=90, ax=ax[1])
ax[0].grid(True)
ax[1].grid(True)
plt.show()


# ## Преобразование Фурье

# In[148]:


# выделяем временной ряд
time_series_1 = df_ts_resemp_H
time_series_1


# In[149]:


# Преобразование Фурье
fft_result = np.fft.fft(time_series_1)


# In[150]:


# Получение амплитуд спектра
amplitudes = np.abs(fft_result)
# Получение частот (циклов в день)
N = len(time_series_1)
freqs = np.fft.fftfreq(N)


# In[151]:


# Оставляем только положительную часть спектра
amplitudes_pos = amplitudes[:len(amplitudes)//2]
freqs_pos = freqs[:len(freqs)//2]


# In[152]:


# Находим 5 наиболее значимых частот, без первого (очень большого) значения 
top_freq_indices = np.argsort(amplitudes_pos)[::-1][1:6]


# In[153]:


# Выводим дни периодичности (сезонности)
periods = 1 / freqs_pos[top_freq_indices]
print("Наиболее значимые частоты и соответствующие им дни периодичности (сезонности):")
for i, period in enumerate(periods):
    print(f"Частота: {freqs_pos[top_freq_indices[i]]:.4f} циклов/день, Дни периодичности: {period:.2f} дней")


# In[154]:


# Визуализация амплитуд спектра 
plt.figure(figsize=(10, 5))
plt.plot(freqs_pos[1:], amplitudes_pos[1:], color = 'green')
plt.title('Амплитудный спектр временного ряда без первого значения')
plt.xlabel('Частота (циклов/день)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.show()


# ### Фильтр Савицкого-Голея

# In[155]:


from scipy.signal import savgol_filter


# In[156]:


# Применяем фильтр Савицкого-Голея для удаления шума
smoothed_time_series = savgol_filter(time_series_1, window_length=5, polyorder=2)


# In[157]:


# график
plt.figure(figsize=(12, 4))
plt.plot(time_series_1.index, time_series_1, label='Исходный временной ряд')
plt.plot(time_series_1.index, smoothed_time_series, label='Очищенный от шума', linestyle='--')
plt.title('Применение фильтра Савицкого-Голея')
plt.legend()
plt.show()


# ### Фильтр Калмана

# In[158]:


def kalman_filter(time_series):
    # Создаем объект фильтра Калмана
    kf = KalmanFilter(dim_x=1, dim_z=1)

    # Инициализируем состояние и ковариацию
    kf.x = np.array([[time_series[0]]])  # начальное состояние
    kf.P *= 1000  # начальная ковариация

    # Определяем матрицы перехода и наблюдения
    kf.F = np.array([[1]])  # матрица перехода
    kf.H = np.array([[1]])  # матрица наблюдения

    # Устанавливаем ковариации шумов
    kf.R *= 0.01  # ковариация измерения
    kf.Q *= 0.01  # ковариация процесса

    # Применяем фильтр Калмана к временному ряду
    filtered_values = []
    for measurement in time_series:
        kf.predict()  # предсказываем следующее состояние
        kf.update(measurement)  # корректируем состояние на основе измерения
        filtered_values.append(kf.x[0, 0])  # добавляем скорректированное значение в список

    return filtered_values


# In[159]:


filtered_series = kalman_filter(time_series_1)


# In[160]:


# график
plt.figure(figsize=(12, 4))
plt.plot(time_series_1.index, time_series_1, label='Исходный временной ряд')
plt.plot(time_series_1.index, filtered_series, label='Очищенный от шума', linestyle='--')
plt.title('Применение фильтра Калмана')
plt.legend()
plt.show()


# # Сохранение выборки в файл

# In[161]:


# # сохранение таблицы в файл
# path = r'D:\MyProjects\save_data\df_ts.xlsx'
# df_ts[:100].to_excel(path, index=True)
# # [:100] - сохранение только первых 100 строк для уменьшения разммера файла


# In[ ]:




