import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np

def recession_indicator(data, country):
    country_data = data.clone().filter(pl.col("Country Name") == f"{country}")

    # Compare current year's GDP with next year's GDP. 
    # If the current year's GDP is greater than next year's GDP then we can assume a recessiont was present.
    country_data = country_data.with_columns(
        Recession = pl.when(pl.col("Value") > pl.col("Value").shift(-1).over("Country Name"))
                    .then(pl.lit("recession"))
                    .otherwise(pl.lit("N/A"))
    )

    recession_data = country_data.clone().filter(pl.col("Recession") == "recession")
    
    return country_data, recession_data

 
def recession_figure(data, countries):
    recession = data.clone().filter(pl.col('Country Name').is_in(countries))

    if len(countries) == 0:
        fig = None

    elif len(countries) == 1:
        country = countries[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            
        cdata, crecession = recession_indicator(recession, f"{country}")

        sns.lineplot(data=cdata, x="Year", y="Value", ax=ax)
        sns.scatterplot(data=crecession, x="Year", y="Value", color='red', marker='X', s=50, ax=ax, label="Start of recession")    
        
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))   
        ax.yaxis.set_major_formatter(formatter)

        ax.legend(title="")
        ax.set_title(f"{country}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    else:
        fig, axes = plt.subplots(len(countries), 1, figsize=(10, 10))

        for ax, country in zip(axes, countries):
            
            cdata, crecession = recession_indicator(recession, f"{country}")

            sns.lineplot(data=cdata, x="Year", y="Value", ax=ax)
            sns.scatterplot(data=crecession, x="Year", y="Value", color='red', marker='X', s=50, ax=ax, label="Start of recession")    
            
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))   
            ax.yaxis.set_major_formatter(formatter)

            ax.legend(title="")
            ax.set_title(f"{country}")
            ax.set_xlabel("")
            ax.set_ylabel("")

    if fig is not None:
        fig.supylabel("GDP")
        fig.supxlabel("Year")
        plt.tight_layout(h_pad=2)

    return fig


def gdp_growth_figure(data, country):
    if len(country) == 1:
        country = country[0]
    else: 
        fig = None
        return fig
    
    max_year = data['Year'].max()
    min_year = max_year - 10

    gdp = data.clone().filter((pl.col("Country Name") == f"{country}") & (pl.col("Year") <= max_year) & (pl.col("Year") >= min_year))
    gdp_data, gdp_recession = recession_indicator(gdp, f'{country}')

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    sns.lineplot(data=gdp_data, x="Year", y="Value", ax=ax)
    sns.scatterplot(data=gdp_recession, x="Year", y="Value", color='red', marker='X', s=100, label="Start of a recession", ax=ax)

    plt.title(f"{country}")
    plt.xticks(np.arange(2013, 2024, 1), rotation=45)
    plt.ylabel("GPD ($)")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))   
    plt.gca().yaxis.set_major_formatter(formatter)

    return fig