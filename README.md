# World Happiness Report Analysis Dashboard

Plotly Dash dashboard for Project 2: World Happiness Report Analysis.

## Dataset

The dashboard uses the cleaned local dataset at `data/cleaned_data.csv`, built from the supplied World Happiness CSV files in `data/`. The available project files cover 2015-2019 and include:

- Country
- Year
- Happiness score
- GDP per capita
- Social support
- Life expectancy
- Freedom
- Generosity
- Corruption trust

Region labels are loaded from the original 2015 and 2016 source files, with a small manual fill for countries that appear only in later years.

## Required Chart Coverage

All chart types required in `Project-Details.pdf` appear in the dashboard:

- Week 1: Column chart, bar chart
- Week 2: Stacked column chart, stacked bar chart, clustered column chart, clustered bar chart
- Week 3: Scatter chart
- Week 4: Bubble chart
- Week 5: Histogram chart
- Week 6: Box chart
- Week 7: Violin chart
- Week 8: Line chart
- Week 9: Area chart

## Interactivity

The dashboard includes connected Dash controls for year, country, distribution metric, selected regions, top-country count, and area-chart threshold. These controls update the charts through Dash callbacks.

## Run Instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8050
```
