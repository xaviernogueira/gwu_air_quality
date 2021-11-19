# README: This script when run downloads ERA5 data from a specified dataset
# USER and API key must be stored in a %USERPROFILE%\.cdsapirc file (.file. text file, see below)
# USER and API key info can be found @ https://cds.climate.copernicus.eu/user after logging in
# make a cdsapirc.txt file and rename it to .cdsapirc. --this file should be as shown in https://github.com/ecmwf/cdsapi
# if having issues, follow the advice here https://tinyurl.com/troubleshoot202

import os
import logging


def init_logger(filename):
    """
    Initializes logger w/ same name as python file
    :param filename:
    """
    logging.basicConfig(filename=os.path.basename(filename).replace('.py', '.log'), filemode='w', level=logging.INFO)
    stderr_logger = logging.StreamHandler()
    stderr_logger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(stderr_logger)

    return


def parse_dataset_url(url):
    """
    return the ERA5 API compatible dataset name given the dataset url
    :param url: url to the ERA5 dataset to download
    :return: just the dataset name
    """
    basename = os.path.basename(url)
    out = basename.split('?')[0]

    return out


def which_data_product(dataset_url):
    """
    This function extracts the input datasets name and type (i.e., monthly vs hourly)
    :param dataset_url: url of the chosen dataset
    :return: [dataset_name, dataset type]
    """
    # get dataset name
    dataset = parse_dataset_url(dataset_url)

    if 'monthly' in dataset:
        d_type = 'monthly'
    elif 'hourly' in dataset or dataset == 'reanalysis-era5-land':
        d_type = 'hourly'
    else:
        return logging.error('ERROR: Cant recognize dataset %s' % dataset)

    return [dataset, d_type]


def monthly_product_types(product_list):
    """
    This function returns a list (len=2) cotaining a list of appropraite product type names, and a list of day codes
    based on user input. The function identifies monthly vs daily data, and guides the user to correct api commands.
    :param dataset: name of the dataset (string)
    :return: list (len=2) containing product type names list [0] and day codes list [1] (one or both may be empty)
    """

    # get items from previous function
    dataset = product_list[0]
    d_type = product_list[1]

    # if a monthly dataset, prompt user to select reanalysis product types
    if d_type == 'monthly':
        p_types = ['monthly_averaged_reanalysis', 'monthly_averaged_reanalysis_by_hour_of_day']

        if 'land' in dataset:
            logging.info('Possible product types: %s' % p_types)

        elif 'levels' in dataset:
            p_types = p_types + ['monthly_averaged_ensemble_members', 'monthly_averaged_ensemble_members_by_hour_of_day']
            logging.info('Possible product types: %s' % p_types)

        else:
            return logging.error('Cant recognize monthly dataset, please input product type list manually.')

        logging.info('To select product types: Input ALL for all listed product types. Or input their list indexes '
                     'separated by commas w/o spaces (i.e., inputing: 0,1 gets %s' % p_types[:2])

        p_prod = input('Input product type selection: ')
        if p_prod == 'ALL':
            c_types = p_types

        else:
            inds = [int(i) for i in p_prod.split(',')]
            c_types = [p_types[i] for i in inds]

    elif 'hourly' in dataset:
        c_types = []

    else:
        return logging.error('ERROR: The chosen dataset is not ERA5, please make days and product type lists manually.')

    return c_types


def form_years(years):
    """
    Convert years from int to string if necessary
    :param years: a string, int, or list of ints or strings
    :return: a formatted list of year strings
    """
    if isinstance(years, int):
        out_years = str(years)
    elif isinstance(years, str):
        out_years = years
    elif isinstance(years, list):
        if isinstance(years[0], int):
            out_years = [str(i) for i in years]
        else:
            out_years = years
    else:
        return logging.error('TYPE ERROR: Years parameter must be an int, string, or a list of ints or strings')

    return out_years


def form_months(months):
    """
    Make a list of month codes formatted for ERA5
    :return: list of formatted month codes
    """

    form_months = []
    if isinstance(months, int):
        form_months.append("{0:0=2d}".format(months))

    elif isinstance(months, list) and '0' in str(months[0]):
        form_months = months.sort()

    elif isinstance(months, list):
        for m in months:
            form_months.append("{0:0=2d}".format(m))

    elif isinstance(months, str):
        if months == 'ALL':
            form_months = ["{0:0=2d}".format(m) for m in list(range(1, 13))]

    return form_months


def form_hours(hours):
    """
    Creates a formatted list of hours for the api request
    :param hours: Either a list of strings, a list of integers (military times), or the string 'ALL'
    :return: formatted list of hours
    """
    if hours == 'ALL':
        fixed_form = ["{0:0=2d}".format(h) for h in list(range(0, 24))]

    elif isinstance(hours[0], str):
        fixed_form = ["{0:0=2d}".format(int(h)) for h in hours]

    elif isinstance(hours[0], int):
        fixed_form = ["{0:0=2d}".format(h) for h in hours]

    else:
        return logging.error('ERROR: hours must be the string ALL, a list of strings, or a list of integers')

    out_hours = ['%s:00' % h for h in fixed_form]

    return out_hours


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


def read_area_to_bboox(area):
    """
    Make bounding box list if desired
    :param area: A string place name, a bounding box list, or the string 'Entire available region' (auto formatted)
    :return: bounding box lat/long list
    """
    if isinstance(area, list):
        bbox = area
    elif isinstance(area, str) and area != 'Entire available region':
        bbox = get_era5_boundingbox(area, state_override=False)
    elif area == 'Entire available region':
        bbox = None
    else:
        return logging.error('Area - %s - invalid: must be empty, a bounding box list, or a place name string' % area)

    if bbox is not None:
        logging.info('Bounding box: %s' % bbox)

    return bbox


def api_request(dataset, products, variables, years, months, days, hours, bbox, form, out_name):
    import cdsapi
    c = cdsapi.Client()

    # get format extension
    format_dict = {'netcdf': '.nc', 'grib': '.grib'}
    ext = format_dict[form]

    # make output save path
    out_path = out_name + ext

    # build api request dictionary
    req_dir = {'variable': variables,
               'year': years,
               'month': months,
               'format': form}

    # add bounding box if desired
    if bbox is not None:
        req_dir['area'] = bbox

    # add product path, days, and hours parameter when appropriate
    if products is not None:
        req_dir['product_type'] = products

    if days is not None:
        req_dir['day'] = days

    # set up hours input
    if hours is not None:
        req_dir['time'] = hours
    else:
        req_dir['time'] = '00:00'

    # make the api request
    c.retrieve(dataset, req_dir, out_path)

    return logging.info('%s should be saved @ %s' % (dataset, out_path))


def main(dataset, out_dir, variables, years, months, hours, form, LOOP_MONTHS=False, LOOP_DAYS=False, area=None):
    """
    Main function facilitating ERA5 API request downloads.
    :param dataset_url: dataset url from cds.climate.copernicus.eu
    :param out_dir: a string directory/folder name where 'era5_download.ext' is saved
    :param variables: a list of variable names to download data for
    :param years: a list of ints or string years i.e., [2018, 2019] or 2018
    :param months: a list of strings, a string, representing month numbers (i.e., 01 or 2). OR can  be the string 'ALL'
    :param f_hours: can be None or the string ALL (for all hours) or a list of strings/ints in military time
    :param form: a string representing output format. Options depend on dataset; i.e., 'netcdf' or 'grib'
    :param area: a place name like 'Argentina' or 'Atlanta'. OR a bounding box list: [latmax, lonmin , latmin , lonmin]
    :return:
    """
    # make sure output directory exists
    if isinstance(out_dir, str):
        if isinstance(variables, str):
            out_dir = out_dir + '\\%s' % variables

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    else:
        return logging.error('TYPE ERROR: out_dir parameter must be a valid folder name string')

    # log input setting
    if area is None or area == '':
        area = 'Entire available region'

    bbox = read_area_to_bboox(area)

    logging.info('Dataset: %s' % dataset)
    logging.info('Variables: %s' % variables)
    logging.info('Years: %s' % years)
    logging.info('Months: %s' % months)
    logging.info('Area: %s' % area)
    logging.info('Output format: %s' % form)

    # support a single hour everyday/month

    # make product types list for monthly data using user input
    if d_type == 'monthly':
        products = monthly_product_types([dataset, d_type])
        days = None

        # this makes combines all monthly reanalysis by year
        if not LOOP_MONTHS:
            for year in years:
                logging.info('Making API request for %s' % year)
                out_name = out_dir + '\\%s' % (dataset + '_%s' % year)
                api_request(dataset, products, variables, year, months, days, hours, bbox, form, out_name)

        # this downloads each monthly reanalysis by year separatly
        else:
            for year in years:
                for m in f_months:
                    logging.info('Making API request for %s- %s' % (year, m))
                    out_name = out_dir + '\\%s' % (dataset + '_%s_%s' % (year, m))
                    api_request(dataset, products, variables, year, m, days, hours, bbox, form, out_name)

    # allow looping for hourly data
    else:
        # create dictionary with months and their day amounts
        months_dict = {}
        correct_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for i, m in enumerate(months):
            months_dict[m] = ["{0:0=2d}".format(d) for d in list(range(correct_days[i]))]

        products = None

        # this just downloads one file for each year
        if not LOOP_DAYS and not LOOP_MONTHS:
            for year in years:
                logging.info('Making API request for %s' % year)
                out_name = out_dir + '\\%s' % (dataset + '_%s' % year)
                f_days = months_dict['01']
                api_request(dataset, products, variables, year, months, f_days, f_hours, bbox, form, out_name)

        # this just loops months, but not days
        elif LOOP_MONTHS and not LOOP_DAYS:
            for year in years:
                for m in f_months:
                    logging.info('Making API request for %s - %s' % (year, m))
                    m_days = months_dict[m]
                    out_name = out_dir + '\\%s' % (dataset + '_%s_%s' % (year, m))
                    api_request(dataset, products, variables, year, m, m_days, f_hours, bbox, form, out_name)

        # this loops every day in every year, and downloads a file for each
        elif LOOP_DAYS and not LOOP_MONTHS:
            for year in years:
                for m in f_months:
                    m_days = months_dict[m]
                    for d in m_days:
                        logging.info('Making API request for %s - %s - %s' % (year, m, d))
                        out_name = out_dir + '\\%s' % (dataset + '_%s_%s_%s' % (year, m, d))
                        api_request(dataset, products, variables, year, m, d, f_hours, bbox, form, out_name)

        else:
            return logging.error('ERROR: LOOP_DAYS and LOOP_MONTHS parameters cannot both be set to True.')

    logging.info('Done')

    return


#  ######################## SET INPUTS ###################################
d_url = 'https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form'
out_dir = r'C:\Users\xrnogueira\Documents\era5_code_test'
variables = ['10m_u_component_of_wind', 'runoff']
years = [2018, 2019]
form = 'netcdf'

# tune time parameters. If both set to 'ALL', all hours and months are downloaded
months = 'ALL'
hours = 'ALL'
days = 'ALL'

# can be a place name (string), if None data is downloaded globally.
area = 'Germany'

# if true, each variable is output separately
LOOP_VARS = True

# if True, a file is downloaded for the selected hours within each day (for hourly data only).
LOOP_DAYS = True

# if True, a file is downloaded for each month (for monthly data only)
LOOP_MONTHS = False


#  ######################## RUN MAIN FUNCTION ###################################
if __name__ == "__main__":
    # initiate logger file and identify data type
    init_logger(__file__)
    dataset, d_type = which_data_product(d_url)
    f_years = form_years(years)
    f_months = form_months(months)
    f_hours = form_hours(hours)

    if not LOOP_VARS:
        if len(variables) == 1:
            v = variables[0]
            main(dataset, out_dir, v, f_years, f_months, f_hours, form, LOOP_MONTHS, LOOP_DAYS, area)
        else:
            main(dataset, out_dir, variables, f_years, f_months, f_hours, form, LOOP_MONTHS, LOOP_DAYS, area)

    else:
        for v in variables:
            main(dataset, out_dir, v, f_years, f_months, f_hours, form, LOOP_MONTHS, LOOP_DAYS, area)



