import time
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent='geoTrain1')


def get_data(x):
    index, row = x
    time.sleep(1.5)

    # Requisição da API
    response = geolocator.reverse(row['query'])
    address = response.raw['address']
    try:
        place_id = response.raw['place_id'] if 'place_id' in response.raw else 'Na'
        osm_type = response.raw['osm_type'] if 'place_id' in response.raw else 'Na'

        country = address['country'] if 'country' in address else 'Na'
        country_code = address['country_code'] if 'country_code' in address else 'Na'

        return place_id, osm_type, country, country_code
    except:
        return None, None, None, None
