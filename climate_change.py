import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from numpy.polynomial import Polynomial
from scipy import stats
import seaborn as sns

# Read and prepare the data
df = pd.read_csv(r"C:\Users\kisha\Downloads\archive\CO2_emission.csv")

# Convert years to numeric
year_columns = [str(year) for year in range(1990, 2020)]
for col in year_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Get top 4 emitters for 2019
top_emitters = df.nlargest(4, '2019')[['Country Name'] + year_columns]
print("\nTop 4 CO2 Emitters per capita in 2019:")
print(top_emitters[['Country Name', '2019']].to_string())
print("\nNote: Qatar is shown for reference but excluded from detailed analysis due to extreme non-linearity")

# Part 1: Original Trend Analysis
plt.figure(figsize=(15, 10))
years = np.array(range(1990, 2020))
colors = ['green', 'crimson', 'blue', 'purple']
color_idx = 0

for _, row in top_emitters.iterrows():
    country_name = row['Country Name']
    if country_name != 'Qatar':
        emissions = row[year_columns].astype(float)
        
        # Plot actual data
        plt.plot(years, emissions, marker='o', label=f'{country_name}',
                color=colors[color_idx], markersize=4)
        
        # Fit polynomial (degree=3 for cubic fit)
        mask = ~emissions.isna()
        valid_years = years[mask]
        valid_emissions = emissions[mask]
        
        # Scale years
        years_scaled = (valid_years - valid_years.min()) / (valid_years.max() - valid_years.min())
        
        # Fit polynomial
        p = Polynomial.fit(years_scaled, valid_emissions, deg=3)
        years_scaled_full = (years[mask] - years[mask].min()) / (years[mask].max() - years[mask].min())
        y_pred = p(years_scaled_full)
        
        # Plot polynomial fit
        plt.plot(valid_years, y_pred, '--', 
                label=f'{country_name} (trend)',
                color=colors[color_idx], alpha=0.7)
        
        # Calculate R-squared
        r2 = r2_score(valid_emissions, y_pred)
        
        print(f"\nAnalysis for {country_name}:")
        print(f"R-squared: {r2:.4f}")
        print(f"Average emissions: {emissions.mean():.2f}")
        print(f"Change 1990-2019: {((emissions.iloc[-1] - emissions.iloc[0])/emissions.iloc[0]*100):.1f}%")
        
        color_idx += 1

plt.title('CO2 Emissions Per Capita (1990-2019) with Trend Analysis')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.annotate('Peak emissions period', 
            xy=(2005, 30), 
            xytext=(2000, 33),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center')

plt.annotate('General downward\ntrend begins', 
            xy=(2010, 20), 
            xytext=(2012, 23),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='left')

plt.tight_layout()
plt.show()

# Part 2: Prediction Analysis with Confidence Intervals
plt.figure(figsize=(12, 6))
future_years = np.array(range(1990, 2026))
colors = ['#2ecc71', '#e74c3c', '#3498db']
color_idx = 0

for _, row in top_emitters.iterrows():
    country_name = row['Country Name']
    if country_name != 'Qatar':
        emissions = row[year_columns].astype(float)
        
        # Plot historical data
        plt.plot(years, emissions, marker='o', label=f'{country_name} (Historical)',
                color=colors[color_idx], markersize=4)
        
        # Fit polynomial for prediction
        mask = ~emissions.isna()
        valid_years = years[mask]
        valid_emissions = emissions[mask]
        
        # Scale years
        years_scaled = (valid_years - valid_years.min()) / (valid_years.max() - valid_years.min())
        future_scaled = (future_years - valid_years.min()) / (valid_years.max() - valid_years.min())
        
        # Fit polynomial
        p = Polynomial.fit(years_scaled, valid_emissions, deg=2)
        y_pred = p(future_scaled)
        
        # Calculate confidence intervals
        y_hist_pred = p(years_scaled)
        rmse = np.sqrt(np.mean((valid_emissions - y_hist_pred) ** 2))
        conf_interval = 1.96 * rmse
        
        # Plot prediction
        plt.plot(future_years[len(years):], y_pred[len(years):], '--', 
                label=f'{country_name} (Predicted)',
                color=colors[color_idx], alpha=0.7)
        
        # Add confidence interval shading
        plt.fill_between(future_years[len(years):], 
                        y_pred[len(years):] - conf_interval,
                        y_pred[len(years):] + conf_interval,
                        color=colors[color_idx], alpha=0.1)
        
        print(f"\nPrediction Analysis for {country_name}:")
        print(f"Predicted 2025 emissions: {y_pred[-1]:.2f} ± {conf_interval:.2f}")
        
        color_idx += 1

plt.axvline(x=2019, color='gray', linestyle=':', alpha=0.5)
plt.text(2019.5, plt.ylim()[0], 'Predictions →', alpha=0.5)
plt.title('CO2 Emissions Predictions to 2025\n(Based on Historical Trends)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Part 3: Regional Analysis
regional_avg = df.groupby('Region')['2019'].agg(['mean', 'count']).sort_values('mean', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(regional_avg.index, regional_avg['mean'])
plt.xticks(rotation=45, ha='right')
plt.title('Average CO2 Emissions per Capita by Region (2019)')
plt.ylabel('CO2 Emissions (metric tons per capita)')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Part 4: Changes Over Time Analysis
df['pct_change'] = ((df['2019'] - df['1990']) / df['1990'] * 100)
top_increase = df.nlargest(5, 'pct_change')[['Country Name', 'pct_change', 'Region']]
top_decrease = df.nsmallest(5, 'pct_change')[['Country Name', 'pct_change', 'Region']]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(top_increase['Country Name'], top_increase['pct_change'], color='red')
plt.title('Top 5 Increases in Emissions\n(1990-2019)')
plt.xlabel('Percentage Change')

plt.subplot(1, 2, 2)
plt.barh(top_decrease['Country Name'], top_decrease['pct_change'], color='green')
plt.title('Top 5 Decreases in Emissions\n(1990-2019)')
plt.xlabel('Percentage Change')

plt.tight_layout()
plt.show()

# Print decade analysis
print("\nDecadal Average Emissions:")
decades = {
    '1990s': [str(year) for year in range(1990, 2000)],
    '2000s': [str(year) for year in range(2000, 2010)],
    '2010s': [str(year) for year in range(2010, 2020)]
}

for _, row in top_emitters.iterrows():
    country_name = row['Country Name']
    if country_name != 'Qatar':
        print(f"\n{country_name}:")
        for decade, decade_years in decades.items():
            decade_avg = row[decade_years].mean()
            print(f"{decade}: {decade_avg:.2f} metric tons per capita")