# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: sco1
# Collaborators (discussion): N/A
# Time: 1:30

import pylab
import re
import calendar
import datetime

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    fitcoeffs = []
    for deg in degs:
        fitcoeffs.append(pylab.polyfit(x, y, deg))

    return fitcoeffs


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    numerator = sum((y-estimated)**2)
    denominator = sum((y-pylab.mean(y))**2)

    rsquared = 1 - numerator/denominator

    return rsquared

def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        modeldata = pylab.polyval(model, x)
        
        pylab.plot(x, y, 'bo', label='Historical Data')
        pylab.plot(x, modeldata, 'r-', label='Model Fit')
        pylab.xlabel('Year')
        pylab.ylabel('Degrees Celsius')

        # Build the appropriate title string based on the degree of the model
        # being plotted
        degree = len(model) - 1
        rsquared = r_squared(y, modeldata)
        if degree == 1:
            seoverslope = se_over_slope(x, y, modeldata, model)
            titlestr = 'Climate Regression Model, Degree {0}\nR-squared: {1:.3f}, SE/slope: {2:.3f}'.format(degree, rsquared, seoverslope)
        else:
            titlestr = 'Climate Regression Model, Degree {0}\nR-squared: {1:.3f}'.format(degree, rsquared)
        
        pylab.title(titlestr)
        pylab.show()


def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    multi_city_average = []
    for year in years:
        # Check for leap year to determine the number of days in the year
        if calendar.isleap(year):
            ndays = 366
        else:
            ndays = 365
        
        city_averages = []
        for city in multi_cities:
            city_averages.append(sum(climate.get_yearly_temp(city, year))/ndays)
        else:
            multi_city_average.append(sum(city_averages)/len(city_averages))
    
    return pylab.array(multi_city_average)


def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    # We define moving average of ​y[i]​ as the average of ​y[i-window_length+1]​ to y[i]
    movavg = []
    for idx in range(len(y)):
        if idx < window_length:
            movavg.append(sum(y[:idx+1])/len(y[:idx+1]))
        else:
            movavg.append(sum(y[idx-window_length+1:idx+1])/len(y[idx-window_length+1:idx+1]))

    return movavg

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    return pylab.sqrt(sum((y-estimated)**2)/len(y))

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    # For each day of the year, find the average temperature across the input 
    # cities for that day. Find the standard deviation of these means for each
    # year
    multi_city_stddev = []
    for year in years:
        # Check for leap year to determine the number of days in the year
        if calendar.isleap(year):
            ndays = 366
        else:
            ndays = 365
        
        daily_temps = []
        for day in range(1, ndays+1):
            # Convert year day to datetime to get month & date for Climate.get_daily_temp()
            # From http://stackoverflow.com/questions/2427555
            tmpdate = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)

            city_temps = []
            for city in multi_cities:
                city_temps.append(climate.get_daily_temp(city, tmpdate.month, tmpdate.day, year))
            else:
                daily_temps.append(pylab.mean(city_temps))
        else:
            multi_city_stddev.append(pylab.std(daily_temps))

    return pylab.array(multi_city_stddev)


def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        modeldata = pylab.polyval(model, x)
        
        pylab.plot(x, y, 'bo', label='Historical Data')
        pylab.plot(x, modeldata, 'r-', label='Model Prediction')
        pylab.xlabel('Year')
        pylab.ticklabel_format(useOffset=False)  # Force full year display
        pylab.ylabel('Degrees Celsius')
        
        degree = len(model) - 1
        model_rmse = rmse(y, modeldata)
        titlestr = 'Climate Model Prediction, Degree {0}\nRMSE: {1:.3f}'.format(degree, model_rmse)
        
        pylab.title(titlestr)
        pylab.show()

if __name__ == '__main__':
    climatedata = Climate('data.csv')
    years = pylab.array(TRAINING_INTERVAL)
    testing_years = pylab.array(TESTING_INTERVAL)
    
    # Part A.4
    # # I
    # jantenth = []
    # for year in TRAINING_INTERVAL:
    #     jantenth.append(climatedata.get_daily_temp('NEW YORK', 1, 10, year))

    # jantenth = pylab.array(jantenth)
    # modelA = generate_models(years, jantenth, [1])
    # evaluate_models_on_training(years, jantenth, modelA)

    # # II
    # yearavg = []
    # for year in TRAINING_INTERVAL:
    #     if calendar.isleap(year):
    #         ndays = 366
    #     else:
    #         ndays = 365
    #     yearavg.append(sum(climatedata.get_yearly_temp('NEW YORK', year))/ndays)
    
    # yearavg = pylab.array(yearavg)
    # modelB = generate_models(years, yearavg, [1])
    # evaluate_models_on_training(years, yearavg, modelB)

    # Part B
    # national_average = gen_cities_avg(climatedata, CITIES, years)
    # modelC = generate_models(years, national_average, [1])
    # evaluate_models_on_training(years, national_average, modelC)

    # Part C
    # national_average = gen_cities_avg(climatedata, CITIES, years)
    # national_average_windowed_5year = moving_average(national_average, 5)
    # modelD = generate_models(years, national_average_windowed_5year, [1])
    # evaluate_models_on_training(years, national_average_windowed_5year, modelD)
    
    # Part D.2
    # # Generate training data
    # national_average = gen_cities_avg(climatedata, CITIES, years)
    # national_average_windowed_5year = moving_average(national_average, 5)
    # modelE = generate_models(years, national_average_windowed_5year, [1, 2, 20])
    # evaluate_models_on_training(years, national_average_windowed_5year, modelE)

    # # Generate testing data
    # national_average_new = gen_cities_avg(climatedata, CITIES, testing_years)
    # national_average_new_windowed_5year = moving_average(national_average_new, 5)
    # evaluate_models_on_testing(testing_years, national_average_new_windowed_5year, modelE)

    # Part E
    # national_deviation = gen_std_devs(climatedata, CITIES, years)
    # national_deviation_windowed_5year = moving_average(national_deviation, 5)
    # modelF = generate_models(years, national_deviation, [1])
    # evaluate_models_on_training(years, national_deviation, modelF)
