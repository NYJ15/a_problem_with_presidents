"""Solution for A Problem with Presidents; a data to information challenge."""
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dataframe_image as dfi


class Solution:
    """Class for all the functions"""

    def __init__(self, filename):
        """Loading data from file to dataframe"""
        self.dataframe = pd.read_csv(filename, sep=',', header=0,
                                     parse_dates=["BIRTH DATE"])[:-1]

    def pre_processing(self):
        """This function solves requirements 1, 2, 3, 4"""

        # Req 1: Add column year_of_birth
        self.dataframe["year_of_birth"] = pd.DatetimeIndex(
            self.dataframe["BIRTH DATE"]).year.astype(int)

        # Getting last alive date to calculate lived years, months and days
        self.dataframe["last_alive"] = self.dataframe["DEATH DATE"]

        # Replacing Nan values with today's date to indicate the president
        # is alive
        self.dataframe["last_alive"].fillna(datetime.now().strftime(
            "%b %d, %Y"), inplace=True)
        # Replacing blank values with "Alive" for locations of death column
        self.dataframe["LOCATION OF DEATH"].fillna('ALIVE', inplace=True)

        # Replacing blank values with "Alive" for death date column
        self.dataframe["DEATH DATE"].fillna('ALIVE', inplace=True)

        # Calculating total time lived
        date_diff = pd.DatetimeIndex(self.dataframe["last_alive"]) - \
                    pd.DatetimeIndex(self.dataframe["BIRTH DATE"])

        # Req 2: Add column lived_years
        self.dataframe["lived_years"] = (date_diff / np.timedelta64(
            1, "Y")).astype(int)

        # Req 3: Add column lived_months
        self.dataframe["lived_months"] = (date_diff / np.timedelta64(
            1, "M")).astype(int)

        # Req 4: Add column lived_days
        self.dataframe["lived_days"] = (date_diff / np.timedelta64(
            1, "D")).astype(int)

        # Styling the dataframe
        styles = [{
            "selector": "caption",
            "props": [
                ("text-align", "center"),
                ("color", 'black')]
        }]

        columns_with_colors = {'year_of_birth': '#99ddff',
                               'lived_years': '#66ccff',
                               'lived_months': '#33bbff',
                               'lived_days': '#00aaff',
                               }

        def assign_bg_color(dat, colors):
            return [f'background-color: {colors}' for i in dat]

        styled_dataframe = self.dataframe.style
        for column, color in columns_with_colors.items():
            styled_dataframe = styled_dataframe.apply(assign_bg_color, axis=0,
                                                      subset=column,
                                                      colors=color)

        styled_dataframe.set_caption("U.S. Presidents Birth and Death "
                                     "Information").set_table_styles(styles)
        styled_dataframe = styled_dataframe.hide(axis="index")
        styled_dataframe = styled_dataframe.hide(["last_alive"],
                                                 axis="columns")

        # Saving the dataframe as image and displaying
        dfi.export(styled_dataframe, "AddedInformation.png")
        president_data_image = mpimg.imread('AddedInformation.png')
        plt.imshow(president_data_image)
        plt.axis('off')
        plt.show()

    def oldest_to_youngest(self):
        """Req 5: Ranking the top 10 Presidents from the longest lived to the
        shortest lived"""

        oldest_to_youngest_df = self.dataframe.sort_values(
            "lived_days", ascending=False).head(10)
        # Renaming column lived_years to age
        oldest_to_youngest_df.rename(columns={"lived_years": "AGE (years)"},
                                     inplace=True)

        oldest_to_youngest_df = oldest_to_youngest_df.style.hide(axis="index")
        oldest_to_youngest_df = oldest_to_youngest_df.hide(["BIRTH PLACE",
                                                            "LOCATION OF "
                                                            "DEATH",
                                                            "lived_months",
                                                            "lived_days",
                                                            "year_of_birth",
                                                            "last_alive"],
                                                           axis="columns")

        # Saving the dataframe as image and displaying
        oldest_to_youngest_df.set_caption("Top 10 Presidents from the "
                                          "longest lived to the shortest "
                                          "lived")
        dfi.export(oldest_to_youngest_df, "LongestToShortest.png")
        img = mpimg.imread('LongestToShortest.png')
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def youngest_to_oldest(self):
        """Req 6: Ranking the top 10 Presidents from the shortest lived to the
        longest lived"""

        youngest_to_oldest_df = self.dataframe.sort_values("lived_days").head(
            10)

        # Renaming column lived_years to age
        youngest_to_oldest_df.rename(columns={"lived_years": "AGE (years)"},
                                     inplace=True)

        youngest_to_oldest_df = youngest_to_oldest_df.style.hide(axis="index")
        youngest_to_oldest_df = youngest_to_oldest_df.hide(["BIRTH PLACE",
                                                            "LOCATION OF "
                                                            "DEATH",
                                                            "lived_months",
                                                            "lived_days",
                                                            "year_of_birth",
                                                            "last_alive"],
                                                           axis="columns")

        # Saving the dataframe as image and displaying
        youngest_to_oldest_df.set_caption("Top 10 Presidents from the "
                                          "shortest lived to the longest "
                                          "lived")
        dfi.export(youngest_to_oldest_df, "ShortestToLongest.png")
        img = mpimg.imread('ShortestToLongest.png')
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def calculate_measures(self):
        """Req 7: Calculate the mean, weighted average, median, mode, max,
        min and standard deviation of lived_days """

        # Calculating mean of lived days
        lived_days_mean = self.dataframe['lived_days'].mean()
        # Calculating median of lived days
        lived_days_median = self.dataframe['lived_days'].median()
        # Calculating mode of lived days
        lived_days_mode = self.dataframe['lived_years'].mode() * 365
        # Calculating max value in lived days
        lived_days_max = self.dataframe['lived_days'].max()
        # Calculating min value in lived days
        lived_days_min = self.dataframe['lived_days'].min()
        # Calculating standard deviation of lived days
        lived_days_sd = self.dataframe['lived_days'].std()

        # Creating formatted table for these measures
        measures = {
            'Mean': [lived_days_mean],
            'Weighted average': [lived_days_mean],
            'Median': [lived_days_median],
            'Mode': [list(lived_days_mode)],
            'Max': [lived_days_max],
            'Min': [lived_days_min],
            'Standard Deviation': [lived_days_sd],
        }

        measures_table = pd.DataFrame(measures, index=['Lived Days'])

        # Saving the dataframe as image and displaying
        dfi.export(measures_table.T.style.set_caption(
            "The mean, weighted average, median, mode, max, min and standard"
            " deviation of lived_days"), "Measures.png")
        img = mpimg.imread('Measures.png')
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def plot_distribution(self):
        """Req 8: Creating a plot to show distribution of the data"""
        self.dataframe['lived_years'].plot(
            kind='hist',
            bins=[40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
            color='#80d4ff', edgecolor='black')

        plt.xlabel("Years Lived")
        plt.ylabel("Number of Presidents")
        plt.title("Distribution of Ages")

        lived_years_mean = self.dataframe['lived_years'].mean()
        lived_years_median = self.dataframe['lived_years'].median()
        lived_years_mode = self.dataframe['lived_years'].mode()

        measures = [lived_years_mode[0], lived_years_mode[1],
                    lived_years_median, lived_years_mean]
        for measurement, name, color in zip(
                measures, ["Mode 1", "Mode 2", "Median", "Mean"],
                ['green', 'green', 'red', 'purple']):
            plt.axvline(x=measurement, linestyle=':', linewidth=1.5,
                        label='{0} at {1}'.format(name, measurement), c=color)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('Distribution.png', bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    FILE = "U.S. Presidents Birth and Death Information - Sheet1.csv"
    solution = Solution(FILE)

    # Solution to requirements 1, 2, 3 and 4.
    solution.pre_processing()

    # Solution to requirement 5.
    solution.oldest_to_youngest()

    # Solution to requirement 6.
    solution.youngest_to_oldest()

    # Solution to requirement 7.
    solution.calculate_measures()

    # Solution to requirement 8.
    solution.plot_distribution()
