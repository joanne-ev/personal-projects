import streamlit as st 
import polars as pl
import matplotlib.pyplot as plt 
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from random import sample

from functions import recession_figure, gdp_growth_figure

# Title
st.title("🌎 Exploring the economic development of countries in Central America")

st.markdown("""
**Portfolio inspiration:** https://www.datacamp.com/datalab/datasets/dataset-python-gdp

**Data source:** https://datahub.io/core/gdp
""")

# Loading data
data = pl.read_csv("gdp.csv")

# Challenges
st.divider()
st.header("Initial data exploration")

# 📉 Challenge 1: Recession
st.subheader("📉 Identifying periods of recession")
st.markdown("""
Although there is no official definition of a recession, a **recession** is generally known as a significant decline in economic activity ([Wikipedia](https://en.wikipedia.org/wiki/Recession)) over an extended period of time, or more specifically a recession is negative GDP growth for at least two consecutive quarters ([Munivenkatappa, 2024](https://doi.org/10.29121/shodhkosh.v5.i7.2024.6217)). 
            
This section will identify recessions in up to three selected countries. The data available for this analysis provides yearly rather than quarterly GDP values for each country. Therefore, a recession here will be alternatively defined as significant negative GDP growth for at least one year. 
""")

# Multi-select options
options = ['Randomise'] + data['Country Name'].unique().sort().to_list()

with st.expander("List of Countries/Regions for Analysis"):
    st.write(f"{options}")

rcountries = st.multiselect(
    "Select up to three countries/regions or `Randomise` to get three random selections:",
    options,
    max_selections=3
)

if "Randomise" in rcountries:
    rcountries = sample(data['Country Name'].unique().to_list(), k=3)
    st.write("Only the three randomly selected countries/regions are included")

st.write(f"**You've selected:**", rcountries)

# Data visualisation
fig = recession_figure(data, rcountries)

if fig is not None:
    st.pyplot(fig)
else:
    st.write("Please choose an option above")

st.divider()

# ---------------------------------------------------------------------------------------------------------
# 💹 Challenge 2: GDP Change
st.subheader("💹 Change in GDP over the past decade")

with st.expander("List of Countries/Regions for Analysis"):
    st.write(f"{options}")

gdp_country = st.multiselect(
    "Select a country/region or `Randomise` to get a random selection:",
    options,
    max_selections=1
)

if "Randomise" in gdp_country:
    gdp_country = sample(data['Country Name'].unique().to_list(), k=1)

st.write("**You've selected:**", gdp_country)

# Data visualisation
fig = gdp_growth_figure(data, gdp_country)

if fig is not None:
    st.pyplot(fig)
else:
    st.write("Please select an option above")

# ---------------------------------------------------------------------------------------------------------
# 📈 Challenge 3: GDP Growth
st.divider()
st.subheader("📈 Highest percentage growth in GDP over the past decade")

growth = data.clone().drop('Country Code').filter(pl.col("Year") >= (pl.col("Year").max().over("Country Name") - 9)).sort(by=['Country Name', 'Year'])

percent_growth = {}
average_growth = {}

for country in growth['Country Name'].unique().to_list():
    df = growth.filter(pl.col("Country Name") == country)
    max_year = df['Year'].max()
    min_year = df['Year'].min()

    max_year_value = df.filter(pl.col("Year") == max_year)['Value'].item()
    min_year_value = df.filter(pl.col("Year") == min_year)['Value'].item()

    percent_change = ((max_year_value - min_year_value) / min_year_value) * 100

    percent_growth[f'{country}\n({str(min_year)}-{str(max_year)[-2:]})'] = round(percent_change, 2)

    average_growth[f'{country}\n({str(min_year)}-{str(max_year)[-2:]})'] = round(percent_change, 2) / 10


percent_growth_df = pl.DataFrame(
    {
        "Country" : percent_growth.keys(),
        "Growth (%)"  : percent_growth.values()
    }
).sort(by="Growth (%)", descending=True).head()

average_growth_df = pl.DataFrame(
    {
        "Country" : average_growth.keys(),
        "Average Growth per Year (%)" : average_growth.values()
    }
).sort(by="Average Growth per Year (%)", descending=True).head()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

sns.barplot(data=percent_growth_df, x="Country", y="Growth (%)", color='skyblue', ax=ax1)
sns.barplot(data=average_growth_df, x="Country", y="Average Growth per Year (%)", color='skyblue', ax=ax2)

ax1.set_title("Percentage Change in the Last Decade")
ax2.set_title("Average Annual Growth in the Last Decade")

for ax in [ax1, ax2]:
    ax.set_xlabel("Country (Year Range)")
    ax.tick_params(axis='x', rotation=45)

st.pyplot(fig)

# ---------------------------------------------------------------------------------------------------------
# Scenario
st.divider()
st.header("Project Brief")

st.write("""
This project is for an NGO interested in the economic development of seven countries in Central America: 

1. 🇧🇿 Belize
2. 🇨🇷 Costa Rica
3. 🇸🇻 El Salvador
4. 🇬🇹 Guatemala
5. 🇭🇳 Honduras
6. 🇳🇮 Nicaragua
7. 🇵🇦 Panama

This project will give a deep-dive on the GDP growth of each country using data from 1960 through to 2016. The analysis will cover the GDP growth of each country per year and decade as well as how each country compares to the regional average.
""")

ca_countries = ['Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Nicaragua', 'Panama']

ca_data = data.clone().filter(
    (pl.col("Country Name").is_in(ca_countries)) & (pl.col('Year') >= 1960) & (pl.col('Year') <= 2016)
)

ca_grouped_data = ca_data.drop('Country Code').group_by(['Country Name', 'Year']).mean().sort(by='Country Name')

for country in ca_countries:
    if ca_grouped_data.filter(pl.col("Country Name") == country)['Year'].shape[0] != 57:
        ca_grouped_data = ca_grouped_data.with_columns(
        pl.col("Country Name").replace({f'{country}': f'{country}*'})
    )

new_ca_countries = ca_grouped_data['Country Name'].unique().sort().to_list()

ca_grouped_year_data = ca_data.drop(['Country Name', 'Country Code']).group_by(['Year']).mean().sort(by="Year")

ca_colours = sns.color_palette("bright")

# ---------------------------------------------------------------------------------------------------------
# Comparing GDP across six Central American countries over time
fig, ax = plt.subplots(1, 1)

sns.lineplot(data=ca_grouped_data, x='Year', y='Value', hue='Country Name', ax=ax)
plt.plot(ca_grouped_year_data['Year'], ca_grouped_year_data['Value'], label='Regional average', color='black', linestyle='--')

plt.text(2000, -15e9, '*Data is unavailable for some years')

plt.ylabel("GDP ($)")
plt.legend()
plt.title("Comparing GDP across six Central\nAmerican countries over time")

st.pyplot(fig)

# ---------------------------------------------------------------------------------------------------------
# Compound Annual Growth Rate (CAGR) between 1960 and 2016
compound_change = {}

for country in new_ca_countries:
    country_data = ca_grouped_data.filter((pl.col("Country Name") == f"{country}"))

    min_year = country_data['Year'].min()
    max_year = country_data['Year'].max()

    min_year_value = country_data.filter(pl.col("Year") == min_year)['Value'].item()
    max_year_value = country_data.filter(pl.col("Year") == max_year)['Value'].item()

    change = ((max_year_value - min_year_value)**(1/(max_year - min_year))) - 1

    compound_change[f'{country}'] = round(change * 100, 2)

compound_change_df = pl.DataFrame(
    {
        "Country" : compound_change.keys(),
        "Percentage"  : compound_change.values()
    }
)

regional_1960 = ca_grouped_year_data.filter(pl.col("Year") == 1960)['Value'].item()
regional_2016 = ca_grouped_year_data.filter(pl.col("Year") == 2016)['Value'].item()

regional_change = round((((regional_2016 - regional_1960)**(1/(2016 - 1960))) - 1) * 100, 2)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

sns.barplot(data=compound_change_df, x="Country", y="Percentage", color='skyblue', ax=ax)
plt.axhline(y=regional_change, color='black', linestyle='--', label=f'Regional average: {regional_change}%')   

plt.xticks(rotation=45)
plt.title("Changes in Compound Annual Growth Rate (CAGR)")
plt.ylabel("Percentage change (%)")
plt.legend()

plt.text(4, -45, '*Data is unavailable for some years')

st.pyplot(fig)

# ---------------------------------------------------------------------------------------------------------
# Changes in GDP Over Time for Six Central American Countries
fig, axes = plt.subplots(4, 2, figsize=(12, 13))

zipped_features = zip(
    axes.flatten(), 
    new_ca_countries, 
    ca_colours
)

for ax, country, colour in zipped_features:
    ca = ca_grouped_data.filter(pl.col('Country Name').is_in([f"{country}"]))

    sns.lineplot(
        data=ca,
        x='Year', y='Value', 
        label=f'Growth over {ca['Year'].max() - ca['Year'].min()} years: {compound_change[f"{country}"]}%', 
        ax=ax, color=colour
    )

    ax.set_title(f'{country} between {ca['Year'].min()} and {ca['Year'].max()}')
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    
axes.flatten()[-1].axis("off")
axes.flatten()[-1].text(0.45, -0.1, '*Data is unavailable for some years')

fig.suptitle("Changes in GDP Over Time for Six Central American Countries")
fig.supxlabel("Year")
fig.supylabel("GDP ($)")
plt.tight_layout(pad=1.7, h_pad=3, w_pad=3)

st.pyplot(fig)

# ---------------------------------------------------------------------------------------------------------
# Changes in GDP Over Time for Six Central American Countries and the Regional Average

fig, axes = plt.subplots(4, 2, figsize=(12, 13))

zipped_features = zip(
    axes.flatten(), 
    new_ca_countries, 
    ca_colours
)

for ax, country, colour in zipped_features:
    ca = ca_grouped_data.filter(pl.col('Country Name').is_in([f"{country}"]))
    ca_reg = ca_grouped_year_data.filter((pl.col('Year') >= ca['Year'].min()) & (pl.col('Year') <= ca['Year'].max()))

    sns.lineplot(
        data=ca,
        x='Year', y='Value', 
        label=f'GDP percentage over {ca['Year'].max() - ca['Year'].min()} years: {compound_change[f"{country}"]}%', 
        ax=ax, color=colour
    )

    ax.plot(ca_reg['Year'], ca_reg['Value'], label=f'Regional average: {regional_change}%', color='black', linestyle='--')
    ax.legend()

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    if ca_reg['Year'].shape[0] != 57:
        ax.set_title(f'{country}* between {ca['Year'].min()} and {ca['Year'].max()}')
    else:
        ax.set_title(f'{country} between {ca['Year'].min()} and {ca['Year'].max()}')
    
    ax.set_xlabel("")
    ax.set_ylabel("")

axes.flatten()[-1].axis("off")
axes.flatten()[-1].text(0.45, -0.1, '*Data is unavailable for some years')

fig.suptitle("Changes in GDP Over Time for Six Central American Countries and the Regional Average")
fig.supxlabel("Year")
fig.supylabel("GDP ($)")
plt.tight_layout(pad=1.7, h_pad=3, w_pad=3)

st.pyplot(fig)
