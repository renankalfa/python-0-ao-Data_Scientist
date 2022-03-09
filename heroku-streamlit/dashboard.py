import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px

st.set_page_config(layout='wide')
st.title('House Rocket Company')
st.markdown('Welcome to House Rocket Data Analysis.')


@st.cache(allow_output_mutation=True)
def get_data(path_of_data):
    data_raw = pd.read_csv(path_of_data)
    return data_raw


def set_feature(data):
    # add new features
    data['price_m2'] = data['price'] / (data['sqft_lot'] / 10.764)
    return data


def overview_data(data):
    data_r = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    f_attributes = st.sidebar.multiselect('Columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Zipcode', data['zipcode'].unique())

    st.title('Data Overview')

    if (f_zipcode != []) and (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_zipcode != []) and (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_zipcode == []) and (f_attributes != []):
        data = data.loc[:, f_attributes]
    else:
        data = data.loc[:, :]

    st.dataframe(data)

    c1, c2 = st.columns((2, 1))

    # Average
    df1 = data_r[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data_r[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data_r[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data_r[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df_new = pd.merge(m2, df4, on='zipcode', how='inner')
    df_new.columns = ['zipcode', 'total_houses', 'm_price', 'm_sqft_living', 'm_price_m2']

    c1.header('Per Zipcode')
    c1.dataframe(df_new.head(10), height=600)

    # Descriptive Statistic
    num_attributes = data_r.select_dtypes(include=['int64', 'float'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))

    std = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df_descriptive = pd.concat([media, mediana, max_, min_, std], axis=1).reset_index()
    df_descriptive.columns = ['atributos', 'media', 'mediana', 'maximo', 'minimo', 'desvio_padrao']

    c2.header('Per Attribute')
    c2.dataframe(df_descriptive, height=310)
    return None


def portfolio_density(data_r):
    # Density of Portfolio
    st.title('Region Overview')
    c3, c4 = st.columns((100000, 1))

    df_density = data_r.sample(1000)

    # Base map - Biblioteca: Folium

    # Densidade de casas (map 1)
    c3.header('Portfolio Density')
    density_map = folium.Map(location=[data_r['lat'].mean(), data_r['long'].mean()], zoom_start=9)
    make_cluster = MarkerCluster().add_to(density_map)

    for name, row in df_density.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup=f'Price R${row["price"]}').add_to(make_cluster)

    with c3:
        folium_static(density_map)
    return None


def commercial(data_r):
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    # Commercial Options
    data_r['date'] = pd.to_datetime(data_r['date']).dt.strftime('%Y-%m-%d')

    st.sidebar.subheader('Select Max Year Built')
    min_year_built = int(data_r['yr_built'].min())
    max_year_built = int(data_r['yr_built'].max())
    f_year = st.sidebar.slider('date',
                               min_year_built,
                               max_year_built,
                               max_year_built)

    st.sidebar.subheader('Select Max Date')
    min_day = datetime.strptime(data_r['date'].min(), '%Y-%m-%d')
    max_day = datetime.strptime(data_r['date'].max(), '%Y-%m-%d')
    f_day = st.sidebar.slider('day',
                              min_day,
                              max_day,
                              max_day)

    # Filtering
    data_r['date'] = pd.to_datetime(data_r['date'])
    data_graph = data_r[(data_r['yr_built'] <= f_year) &
                        (data_r['date'] <= f_day)]

    # Average Price by Year
    st.header('Average Price by Year Built')
    df_year = data_graph[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
    fig = px.line(df_year, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Average Price by Day
    st.header('Average Price by Day')
    df_day = data_graph[['date', 'price']].groupby('date').mean().reset_index()
    fig = px.line(df_day, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    return data_graph


def histogram_graph(data_r, data_graph):
    # Histogram
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # Histogram options
    min_price = int(data_graph['price'].min())
    max_price = int(data_graph['price'].max())
    avg_price = int(data_graph['price'].mean())

    f_price = st.sidebar.slider('price',
                                min_price,
                                max_price,
                                avg_price)

    # Filtering
    data_h = data_graph.loc[data['price'] <= f_price]

    # Ploting
    fig = px.histogram(data_h, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    # Distribuição dos imóveis por categorias físicas

    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # Attributes Options
    f_bed = st.sidebar.selectbox('max bedrooms',
                                 sorted(data_h['bedrooms'].unique()))

    f_bath = st.sidebar.selectbox('max bathrooms',
                                  sorted(data_h['bathrooms'].unique()))

    f_floors = st.sidebar.selectbox('max floors',
                                    sorted(set(data_h['floors'].unique())))

    f_water = st.sidebar.checkbox('only water view')

    c5, c6 = st.columns(2)
    c7, c8 = st.columns(2)
    # Filtering
    data_h = data_h[(data_h['bedrooms'] < f_bed) &
                    (data_h['bathrooms'] < f_bath) &
                    (data_h['floors'] < f_floors)]
    if f_water:
        data_h = data_h[data_h['waterfront'] == 1]
    else:
        data_h = data_h.copy()

    # Houses per bedrooms
    c5.header('Houses per bedrooms')
    fig = px.histogram(data_h, x='bedrooms', nbins=19)
    c5.plotly_chart(fig, use_container_width=True)

    # Houses per bathrooms
    c6.header('Houses per bathrooms')
    fig = px.histogram(data_h, x='bathrooms', nbins=19)
    c6.plotly_chart(fig, use_container_width=True)

    # Houses per floors
    c7.header('Houses per floors')
    fig = px.histogram(data_h, x='floors', nbins=19)
    c7.plotly_chart(fig, use_container_width=True)

    # Houses per waterfront
    c8.header('Houses per waterfront')
    fig = px.histogram(data_h, x='waterfront', nbins=19)
    c8.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    # Data extration
    # Get Data
    path = 'datasets/kc_house_data.csv'
    data = get_data(path)
    # Transformation
    data = set_feature(data)
    data_r = data.copy()
    overview_data(data)
    portfolio_density(data_r)
    data_graph = commercial(data_r)
    histogram_graph(data_r, data_graph)
