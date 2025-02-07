class Item:
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
            self.sum = item1.sum + item2.sum
            self.sum_of_squares = item1.sum_of_squares + item2.sum_of_squares
            self.num = item1.num + item2.num
        else:
            raise ValueError("Invalid arguments. Expected a single value or two Item objects.")
        if self.num==1:
            self.quality=1
        else:
            n=self.num
            error = ((self.sum_of_squares/n - pow(self.sum/n,2))/pow(self.sum/n,2)
            self.quality=((n-1)-error)/(n-1)

    def __repr__(self):
        """
        String representation of the Item object.
        """
        return f"Item(sum={self.sum}, sum_of_squares={self.sum_of_squares}, num={self.num})"

class StructuredItems():
    def __init__(self):

        # Define the number of days in each month (non-leap year)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Create a vector of Items representing the calendar year
        self.calendar_items = []

        for month in range(12):  # Months are 0-indexed (0 = January, 11 = December)
            month_number = month + 1  # Convert to 1-indexed month number
            for day in range(1, days_in_month[month] + 1):  # Days are 1-indexed
                value = day + (month_number * 1000)  # Value = day + (month * 1000)
                item = Item.from_value(value)
                self.calendar_items.append(item)


class StructuredItems():
    def __init__(self):

        # Define the number of days in each month (non-leap year)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Create a vector of Items representing the calendar year
        self.calendar_items = []

        for month in range(12):  # Months are 0-indexed (0 = January, 11 = December)
            month_number = month + 1  # Convert to 1-indexed month number
            for day in range(1, days_in_month[month] + 1):  # Days are 1-indexed
                value = day + (month_number * 1000)  # Value = day + (month * 1000)
                item = Item.from_value(value)
                self.calendar_items.append(item)
