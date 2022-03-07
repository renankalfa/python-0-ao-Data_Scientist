import plotly.express as px
import pandas as pd

data = pd.read_csv('kc_house_data.csv')
data_mapa = data[['id', 'lat', 'long', 'price']]

grafico1 = px.scatter_mapbox(data_mapa, lat='lat', lon='long',
                             hover_name='id', hover_data=['price'],
                             color_discrete_sequence=['fuchsia'],
                             zoom=3, height=300)
grafico1.update_layout(mapbox_style='open-street-map')
grafico1.update_layout(height=600, margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
grafico1.show()

grafico1.write_html('map_house_rocket.html')
