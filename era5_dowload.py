# README: This script when run downloads ERA5 data from a specified dataset
# USER and API key must be stored in a %USERPROFILE%\.cdsapirc file (.file. text file, see below)
# USER and API key info can be found @ https://cds.climate.copernicus.eu/user after logging in
# make a cdsapirc.txt file and rename it to .cdsapirc. --this file should be as shown in https://github.com/ecmwf/cdsapi

import os
import logging


def init_logger(filename):
    """Initializes logger w/ same name as python file"""
    logging.basicConfig(filename=os.path.basename(filename).replace('.py', '.log'), filemode='w', level=logging.INFO)
    stderr_logger = logging.StreamHandler()
    stderr_logger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(stderr_logger)

    return

def parse_dataset_url(url):
    """return the ERA5 API compatible dataset name given the dataset url"""
    basename = os.path.basename(url)
    out = basename.split('?')[0]

    return out



def get_era5_boundingbox(place, state_override=False):
    """
    get the bounding box of a country or US state in EPSG4326 given it's name
    based on work by @mattijin (https://github.com/mattijn)

    Parameters:
    place - a name (str) of a country, city, or state in english and lowercase (i.e., beunos aires)
    output_as - a ERA5 API compatible 'boundingbox' list w/ [latmax, lonmin , latmin , lonmin]
    state_override - default is False (bool), only make True if mapping a state
    ------------------
    Returns:
    output - a list with coordinates as floats i.e., [[11.777, 53.7253321, -70.2695876, 7.2274985]]
    """
    import requests
    import iso3166
    # create url to pull openstreetmap data
    url_prefix = 'http://nominatim.openstreetmap.org/search?country='

    country_list = [j.lower() for j in iso3166.countries_by_name.keys()]

    if place not in country_list:
        if state_override:
            url_prefix = url_prefix.replace('country=', 'state=')
        else:
            url_prefix = url_prefix.replace('country=', 'city=')

    url = '{0}{1}{2}'.format(url_prefix, place, '&format=json&polygon=0')
    response = requests.get(url).json()[0]

    # parse response to list, convert to integer if desired
    lst = response['boundingbox']
    coors = [float(i) for i in lst]
    era5_box = [coors[-1], coors[0], coors[-2], coors[1]]

    return era5_box


def api_request(dataset, format, out_dir):
    import cdsapi
    c = cdsapi.Client()

    # get format extension
    format_dict = {'netcdf': '.nc', 'grib': '.grib'}
    ext = format_dict[format]

    # build api request dictionary

    c.retrieve(
        dataset,
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                '100m_u_component_of_wind', '100m_v_component_of_wind', 'boundary_layer_height',
            ],
            'year': '2019',
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'time': '00:00',
            'area': [
                49.38, -124.84, 24.39,
                -66.88,
            ],
            'format': format,
        },
        out_dir + '\\era5_download%s' % ext)


######################### RUN MAIN FUNCTION ###################################
dataset_url = 'https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview'
variables = ['100m_u_component_of_wind', '100m_v_component_of_wind']
years = [2019]
months = list(range(1, 13))

area = ''
format = 'netcdf'


def main(dataset_url, out_dir, variables, years, months, format, area=None):
    """

    :param dataset_url: dataset url from cds.climate.copernicus.eu
    :param out_dir: a string directory/folder name where 'era5_download.ext' is saved
    :param variables: a list of variable names to download data for
    :param years: a list of ints or string years i.e., [2018, 2019] or 2018
    :param months: a list of strings, a string, representing month numbers (i.e., 01 or 2). OR can  be the string 'ALL'
    :param area: a place name like 'Argentina' or 'Atlanta'. OR a bounding box list: [latmax, lonmin , latmin , lonmin]
    :param format: a string representing output format. Options depend on dataset; i.e., 'netcdf' or 'grib'
    :return:
    """
    # find dataset name and log download information
    dataset = parse_dataset_url(dataset_url)

    if area is None:
        area = 'Entire available region'

    logging.info('Dataset: %s' % dataset)
    logging.info('Variables: %s' % variables)
    logging.info('Years: %s' % years)
    logging.info('Months: %s' % months)
    logging.info('Area: %s' % area)
    logging.info('Output format: %s' % format)

    # make sure output directory exists
    if isinstance(out_dir, str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        return logging.error('TYPE ERROR: out_dir parameter must be a valid folder name string')

    # convert years from int to string if necessary
    if isinstance(years, int):
        years = str(years)
    elif isinstance(years, str):
        years = years
    elif isinstance(years, list):
        if isinstance(years[0], int):
            years = [str(i) for i in years]
        else:
            years = years
    else:
        return logging.error('TYPE ERROR: Years parameter must be an int, string, or a list of ints or strings')

    # make a list of month codes formatted for ERA5
    form_months = []
    if isinstance(months, int):
        form_months.append("{0:0=2d}".format(months))

    elif isinstance(months, list) and '0' in months[0]:
        form_months = months.sort()

    elif isinstance(months, list):
        for m in months:
            form_months.append("{0:0=2d}".format(m))

    if isinstance(area, list):
        bbox = area
    elif isinstance(area, str) and area != 'Entire available region':
        bbox = get_era5_boundingbox(area, state_override=False)
    else:
        return logging.error('Area - %s - invalid: must be empty, a bounding box list, or a place name string' % area)
    return


if __name__ == "__main__":
    main(dataset_url, variables, years, months, area, format)
