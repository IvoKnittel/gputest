import pickle
import math
import numpy as np
from scipy.signal import chirp

def items():
    return odd_step(10)


def sz(items):
    return len(items)

class Item:
    selected=False
    def __init__(self, *args):
        """
        Initialize an Item object.
        Can be initialized with a single value or two Item objects.
        """
        if len(args) == 1 and isinstance(args[0], (int, float)):
            # Initialize with a single value
            value = args[0]
            self.sum = value
            self.sum_of_squares = value * value
            self.num = 1
        elif len(args) == 2 and all(isinstance(arg, Item) for arg in args):
            # Initialize with two Item objects
            item1, item2 = args
            if item1.num ==0:
                self.sum = item2.sum
                self.sum_of_squares = item2.sum_of_squares
                self.num = item2.num
            elif item2.num==0:
                self.sum = item1.sum
                self.sum_of_squares = item1.sum_of_squares
                self.num = item1.num
            else:
                self.sum = item1.sum + item2.sum
                self.sum_of_squares = item1.sum_of_squares + item2.sum_of_squares
                self.num = item1.num + item2.num

        elif len(args) == 0:
            self.sum = math.nan
            self.sum_of_squares = math.nan
            self.num = 0
            self.quality = -1.0
        else:
            raise ValueError("Invalid arguments. Expected a single value or two Item objects.")
        if self.num == 1:
            self.quality = 1
        elif np.isnan(self.sum):
            self.quality = -1.0
        elif math.fabs(self.sum) > 1e-5:
            n=self.num
            error = (self.sum_of_squares / n - pow(self.sum / n, 2)) / pow(self.sum / n, 2)
            self.quality = (n - 1 - error) / (n - 1)
        else:
            self.quality = 1
    def value(self):
        return self.sum/self.num
    def std_var(self):
        n = self.num
        error =  math.sqrt((self.sum_of_squares / n) - pow(self.sum / n, 2))
        return self.sum/self.num


    def __repr__(self):
        """
        String representation of the Item object.
        """
        return f"Item(sum={self.sum}, sum_of_squares={self.sum_of_squares}, num={self.num})"



def random_items_(N:int):
    values = np.random.rand(N)
    with open('random_items.pkl', 'wb') as f:
        pickle.dump(values, f)

    return np.array([Item(val) for val in values])

def random_items(N:int):
    with open('random_items.pkl', 'rb') as f:
        values = pickle.load(f)
    return np.array([Item(val) for val in values])

def odd_step(N:int):
    step_at_pos5 = np.zeros(N)
    step_at_pos5[5:]=1
    return np.array([Item(val) for val in step_at_pos5])

def calendar_items():
    # Define the number of days in each month (non-leap year)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Create a vector of Items representing the calendar year
    calendar_items = []

    for month in range(12):  # Months are 0-indexed (0 = January, 11 = December)
        month_number = month + 1  # Convert to 1-indexed month number
        for day in range(1, days_in_month[month] + 1):  # Days are 1-indexed
            value = day + (month_number * 1000)  # Value = day + (month * 1000)
            item = Item(value)
            calendar_items.append(item)

    return np.array(calendar_items)


def sine_items():
    xvec = np.linspace(0, 10, 200)
    items=[]
    for x in xvec:
        y = np.sin(x)
        items.append(Item(y))

    return np.array(items, dtype=Item)



def chirp_items():
    # Parameters for the chirp signal
    tvec = np.linspace(0, 10, 1000)  # Time array from 0 to 10 seconds with 1000 points
    f0 = 0.1  # Start frequency of the chirp
    f1 = 3  # End frequency of the chirp at time t1
    t1 = 10  # Time at which f1 is reached

    # Generate the chirp signal
    yvec = chirp(tvec, f0=f0, f1=f1, t1=t1, method='linear')

    items = [Item(y) for y in yvec]
    return items

