from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


# =========================
# Data loading and setup
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(DATA_DIR / "cleaned_data.csv")
df["year"] = df["year"].astype(int)


def load_region_map():
    """Use the original WHR region labels where they exist, then fill known gaps."""
    region_map = {}
    for year_file in ("2015.csv", "2016.csv"):
        path = DATA_DIR / year_file
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        if {"Country", "Region"}.issubset(raw.columns):
            region_map.update(dict(zip(raw["Country"], raw["Region"])))

    region_map.update(
        {
            "Gambia": "Sub-Saharan Africa",
            "Hong Kong S.A.R., China": "Eastern Asia",
            "North Macedonia": "Central and Eastern Europe",
            "Northern Cyprus": "Middle East and Northern Africa",
            "Taiwan Province of China": "Eastern Asia",
            "Trinidad & Tobago": "Latin America and Caribbean",
        }
    )
    return region_map


df["region"] = df["country"].map(load_region_map()).fillna("Other")

FACTOR_COLUMNS = [
    "gdp_per_capita",
    "social_support",
    "life_expectancy",
    "freedom",
    "generosity",
    "corruption",
]

METRIC_LABELS = {
    "happiness_score": "Happiness Score",
    "gdp_per_capita": "GDP per Capita",
    "social_support": "Social Support",
    "life_expectancy": "Life Expectancy",
    "freedom": "Freedom",
    "generosity": "Generosity",
    "corruption": "Corruption Trust",
}

FACTOR_LABELS = {
    "gdp_per_capita": "GDP",
    "social_support": "Social",
    "life_expectancy": "Life Expect.",
    "freedom": "Freedom",
    "generosity": "Generosity",
    "corruption": "Trust",
}

SHORT_REGION_LABELS = {
    "Australia and New Zealand": "Aus/NZ",
    "Central and Eastern Europe": "C/E Europe",
    "Eastern Asia": "East Asia",
    "Latin America and Caribbean": "LatAm/Carib.",
    "Middle East and Northern Africa": "MENA",
    "North America": "N. America",
    "Southeastern Asia": "SE Asia",
    "Southern Asia": "S. Asia",
    "Sub-Saharan Africa": "Sub-Sahara",
    "Western Europe": "W. Europe",
    "Other": "Other",
}
SHORT_TO_REGION = {short: region for region, short in SHORT_REGION_LABELS.items()}

REGION_ORDER = [
    "Western Europe",
    "North America",
    "Australia and New Zealand",
    "Latin America and Caribbean",
    "Central and Eastern Europe",
    "Eastern Asia",
    "Southeastern Asia",
    "Middle East and Northern Africa",
    "Southern Asia",
    "Sub-Saharan Africa",
    "Other",
]

REGION_PALETTE = {
    "Western Europe": "#bfdbfe",
    "North America": "#bbf7d0",
    "Australia and New Zealand": "#fde68a",
    "Latin America and Caribbean": "#fecaca",
    "Central and Eastern Europe": "#93c5fd",
    "Eastern Asia": "#ddd6fe",
    "Southeastern Asia": "#fbcfe8",
    "Middle East and Northern Africa": "#bae6fd",
    "Southern Asia": "#e9d5ff",
    "Sub-Saharan Africa": "#e5e7eb",
    "Other": "#cbd5e1",
}

FACTOR_PALETTE = {
    "gdp_per_capita": "#bfdbfe",
    "social_support": "#93c5fd",
    "life_expectancy": "#60a5fa",
    "freedom": "#3b82f6",
    "generosity": "#2563eb",
    "corruption": "#1d4ed8",
}

HIGHLIGHT_PALETTE = {
    "Highest": "#bbf7d0",
    "Lowest": "#fecaca",
    "Other": "#bfdbfe",
}

HISTOGRAM_PALETTE = {
    "Low values": "#fecaca",
    "Middle values": "#bfdbfe",
    "High values": "#bbf7d0",
}

OUTLIER_PALETTE = {
    "Other countries": "#bfdbfe",
    "Outlier": "#bbf7d0",
}

PLOT_BG = "#081524"
PAPER_BG = "#0b1e33"
GRID_COLOR = "#203852"
AXIS_COLOR = "#d7e7ff"
TEXT_COLOR = "#edf6ff"
MUTED_TEXT = "#a8bed8"
LEGEND_BG = "rgba(9, 24, 42, 0.94)"
LEGEND_BORDER = "#29496d"
MARKER_BORDER = "#020617"
GUIDE_LINE = "#e0f2fe"
MEDIAN_LINE = "#c084fc"

YEARS = sorted(df["year"].unique())
COUNTRIES = sorted(df["country"].unique())
REGIONS = [r for r in REGION_ORDER if r in set(df["region"])]
DEFAULT_REGIONS = [
    r
    for r in [
        "Western Europe",
        "North America",
        "Middle East and Northern Africa",
        "Sub-Saharan Africa",
    ]
    if r in REGIONS
]


def metric_options(include_score=True):
    columns = ["happiness_score", *FACTOR_COLUMNS] if include_score else FACTOR_COLUMNS
    return [{"label": METRIC_LABELS[col], "value": col} for col in columns]


def filter_regions(data, selected_regions):
    if selected_regions:
        return data[data["region"].isin(selected_regions)].copy()
    return data.copy()


def grouped_region_year_metric(metric, selected_regions):
    data = filter_regions(df[df["region"] != "Other"], selected_regions)
    return (
        data.groupby(["year", "region"], as_index=False)[metric]
        .mean()
        .sort_values(["year", "region"])
    )


def year_region_factors(year, selected_regions):
    data = filter_regions(df[(df["year"] == year) & (df["region"] != "Other")], selected_regions)
    grouped = data.groupby("region", as_index=False)[FACTOR_COLUMNS + ["happiness_score"]].mean()
    grouped["factor_total"] = grouped[FACTOR_COLUMNS].sum(axis=1)
    grouped["short_region"] = grouped["region"].map(SHORT_REGION_LABELS).fillna(grouped["region"])
    return grouped.sort_values("happiness_score", ascending=False)


def year_region_factor_totals(year, selected_regions):
    data = filter_regions(df[(df["year"] == year) & (df["region"] != "Other")], selected_regions)
    grouped = data.groupby("region", as_index=False)[FACTOR_COLUMNS].sum()
    grouped["country_count"] = data.groupby("region")["country"].nunique().reindex(grouped["region"]).values
    grouped["factor_total"] = grouped[FACTOR_COLUMNS].sum(axis=1)
    grouped["short_region"] = grouped["region"].map(SHORT_REGION_LABELS).fillna(grouped["region"])
    return grouped.sort_values("factor_total", ascending=False)


def finalize_figure(
    fig,
    title,
    x_title,
    y_title,
    *,
    x_zero=False,
    y_zero=False,
    show_legend=True,
    legend_title=None,
    height=430,
):
    legend_margin = 174 if show_legend else 34
    fig.update_layout(
        template="plotly_dark",
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 18}},
        height=height,
        margin={"l": 64, "r": legend_margin, "t": 104, "b": 72},
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "size": 12, "color": TEXT_COLOR},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        hoverlabel={"bgcolor": "#020617", "font_color": "#ffffff", "bordercolor": "#38bdf8"},
        hovermode="closest",
        showlegend=show_legend,
        legend_title_text=legend_title or "",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": LEGEND_BG,
            "bordercolor": LEGEND_BORDER,
            "borderwidth": 1,
            "font": {"size": 11},
        },
    )
    fig.update_xaxes(
        title=x_title,
        showline=True,
        linewidth=1,
        linecolor=AXIS_COLOR,
        mirror=True,
        ticks="outside",
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title=y_title,
        showline=True,
        linewidth=1,
        linecolor=AXIS_COLOR,
        mirror=True,
        ticks="outside",
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    if x_zero:
        fig.update_xaxes(rangemode="tozero")
    if y_zero:
        fig.update_yaxes(rangemode="tozero")
    return fig


def shorten_region_legend(fig):
    for trace in fig.data:
        if getattr(trace, "name", None) in SHORT_REGION_LABELS:
            trace.name = SHORT_REGION_LABELS[trace.name]
    return fig


def empty_figure(title):
    fig = go.Figure()
    fig.add_annotation(
        text="No data for the selected filters",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    return finalize_figure(fig, title, "", "", show_legend=False)


def histogram_bin_edges(values, metric):
    minimum = float(values.min())
    maximum = float(values.max())
    if metric == "happiness_score":
        start = np.floor(minimum * 2) / 2
        end = np.ceil(maximum * 2) / 2
        return np.arange(start, end + 0.5, 0.5)

    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    if iqr > 0:
        width = 2 * iqr / (len(values) ** (1 / 3))
    else:
        width = (maximum - minimum) / 10
    if width <= 0:
        width = 1
    bin_count = int(np.clip(np.ceil((maximum - minimum) / width), 8, 18))
    return np.linspace(minimum, maximum, bin_count + 1)


def histogram_band(center, metric, values):
    if metric == "happiness_score":
        if center < 4:
            return "Low values"
        if center < 6:
            return "Middle values"
        return "High values"

    q25, q75 = np.percentile(values, [25, 75])
    if center < q25:
        return "Low values"
    if center <= q75:
        return "Middle values"
    return "High values"


def kde_scaled_to_counts(values, x_grid, bin_width):
    if len(values) < 2:
        return np.zeros_like(x_grid)

    std = values.std(ddof=1)
    if std <= 0:
        return np.zeros_like(x_grid)

    bandwidth = 1.06 * std * (len(values) ** (-1 / 5))
    bandwidth = max(bandwidth, std / 12, 0.001)
    z = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    return density * len(values) * bin_width


def kde_density(values, x_grid):
    if len(values) < 2:
        return np.zeros_like(x_grid)

    std = values.std(ddof=1)
    if std <= 0:
        return np.zeros_like(x_grid)

    bandwidth = 1.06 * std * (len(values) ** (-1 / 5))
    bandwidth = max(bandwidth, std / 12, 0.001)
    z = (x_grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def add_positive_residual_outlier(data, x_metric, y_metric):
    plot_data = data.copy()
    plot_data["point_role"] = "Other countries"

    clean = plot_data[[x_metric, y_metric]].dropna()
    if len(clean) < 3 or clean[x_metric].nunique() < 2:
        return plot_data, None, None, None

    slope, intercept = np.polyfit(clean[x_metric], clean[y_metric], 1)
    predicted = slope * clean[x_metric] + intercept
    residuals = clean[y_metric] - predicted
    positive_idx = residuals.idxmax()
    plot_data["residual"] = np.nan
    plot_data.loc[residuals.index, "residual"] = residuals
    plot_data.loc[positive_idx, "point_role"] = "Outlier"

    x_vals = np.linspace(clean[x_metric].min(), clean[x_metric].max(), 50)
    y_vals = slope * x_vals + intercept
    return plot_data, plot_data.loc[positive_idx], x_vals, y_vals


def control_block(label, child, class_name="control"):
    return html.Div(
        className=class_name,
        children=[html.Label(label, className="control-label"), child],
    )


def chart_card(week, chart_type, graph_id, class_name="chart-card"):
    return html.Article(
        className=class_name,
        children=[
            html.Div(
                className="chart-heading",
                children=[
                    html.Span(week, className="week-pill"),
                    html.H3(chart_type),
                ],
            ),
            dcc.Graph(
                id=graph_id,
                className="chart-graph",
                config={"displayModeBar": False, "responsive": True},
                style={"height": "450px", "minHeight": "450px"},
            ),
        ],
    )


# =========================
# App layout
# =========================
app = Dash(__name__)
app.title = "World Happiness Report Analysis"

app.layout = html.Div(
    className="page",
    children=[
        html.Header(
            className="page-header",
            children=[
                html.Div(
                    children=[
                        html.H1("World Happiness Report Analysis"),
                        html.P(
                            "Interactive Plotly Dash dashboard covering Weeks 1-9 chart requirements.",
                            className="header-copy",
                        ),
                    ]
                ),
                html.Div(
                    className="coverage-strip",
                    children=[
                        html.Span("Column"),
                        html.Span("Bar"),
                        html.Span("Stacked Column"),
                        html.Span("Stacked Bar"),
                        html.Span("Clustered Column"),
                        html.Span("Clustered Bar"),
                        html.Span("Scatter"),
                        html.Span("Bubble"),
                        html.Span("Histogram"),
                        html.Span("Box"),
                        html.Span("Violin"),
                        html.Span("Line"),
                        html.Span("Area"),
                    ],
                ),
            ],
        ),
        html.Section(
            className="controls-panel",
            children=[
                control_block(
                    "Year",
                    dcc.Dropdown(
                        id="year-control",
                        options=[{"label": str(year), "value": year} for year in YEARS],
                        value=max(YEARS),
                        clearable=False,
                    ),
                ),
                control_block(
                    "Country",
                    dcc.Dropdown(
                        id="country-control",
                        options=[{"label": country, "value": country} for country in COUNTRIES],
                        value="Egypt" if "Egypt" in COUNTRIES else COUNTRIES[0],
                        clearable=False,
                    ),
                ),
                control_block(
                    "Distribution Metric",
                    dcc.Dropdown(
                        id="metric-control",
                        options=metric_options(include_score=True),
                        value="happiness_score",
                        clearable=False,
                    ),
                ),
                control_block(
                    "Regions",
                    dcc.Dropdown(
                        id="region-control",
                        options=[{"label": region, "value": region} for region in REGIONS],
                        value=DEFAULT_REGIONS,
                        multi=True,
                    ),
                    "control wide-control",
                ),
                control_block(
                    "Top Countries",
                    dcc.Slider(
                        id="top-n-control",
                        min=5,
                        max=15,
                        step=1,
                        value=10,
                        marks={5: "5", 10: "10", 15: "15"},
                    ),
                ),
                control_block(
                    "Happiness Threshold",
                    dcc.Slider(
                        id="threshold-control",
                        min=3,
                        max=8,
                        step=0.25,
                        value=6,
                        marks={3: "3", 5: "5", 6: "6", 7: "7", 8: "8"},
                    ),
                ),
            ],
        ),
        html.Main(
            className="dashboard",
            children=[
                html.Section(
                    className="topic-section",
                    children=[
                        html.H2("Week 1: Comparison Charts"),
                        html.Div(
                            className="chart-grid two-col",
                            children=[
                                chart_card("Week 1", "Column Chart: Happiness Factor Profile", "column-chart"),
                                chart_card("Week 1", "Bar Chart: Happiness Ranking", "bar-chart"),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="topic-section",
                    children=[
                        html.H2("Week 2: Stacked and Clustered Comparison Charts"),
                        html.Div(
                            className="chart-grid two-col",
                            children=[
                                chart_card("Week 2", "Stacked Column Chart: Regional Factor Mix", "stacked-column-chart"),
                                chart_card("Week 2", "Stacked Bar Chart: Regional Happiness Class Mix", "stacked-bar-chart"),
                                chart_card("Week 2", "Clustered Column Chart: Regional Scores Over Time", "clustered-column-chart"),
                                chart_card("Week 2", "Clustered Bar Chart: GDP vs Social Support", "clustered-bar-chart"),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="topic-section",
                    children=[
                        html.H2("Weeks 3-4: Relationship Charts"),
                        html.Div(
                            className="chart-grid two-col",
                            children=[
                                chart_card("Week 3", "Scatter Chart: Social Support and Happiness", "scatter-chart"),
                                chart_card("Week 4", "Bubble Chart: Life Expectancy, Happiness, and GDP", "bubble-chart"),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="topic-section",
                    children=[
                        html.H2("Weeks 5-7: Distribution Charts"),
                        html.Div(
                            className="chart-grid three-col",
                            children=[
                                chart_card(
                                    "Week 5",
                                    "KDE Density Plot: Global Distribution",
                                    "histogram-chart",
                                    "chart-card feature-card",
                                ),
                                chart_card("Week 6", "Box Chart: Regional Spread", "box-chart"),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="topic-section",
                    children=[
                        html.H2("Weeks 7-9: Distribution and Time-Series Charts"),
                        html.Div(
                            className="chart-grid two-col",
                            children=[
                                chart_card("Week 7", "Violin Chart: Regional Density", "violin-chart"),
                                chart_card("Week 9", "Area Chart: Total Happiness Volume Over Time", "area-chart"),
                                chart_card(
                                    "Week 8",
                                    "Line Chart: Regional Happiness Trend",
                                    "line-chart",
                                    "chart-card full-card",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# =========================
# Week 1 callbacks
# =========================
@app.callback(
    Output("column-chart", "figure"),
    Input("year-control", "value"),
    Input("country-control", "value"),
)
def update_column_chart(year, country):
    data = df[(df["year"] == year) & (df["country"] == country)]
    if data.empty:
        data = df[df["year"] == year].sort_values("happiness_score", ascending=False).head(1)
    if data.empty:
        return empty_figure("Column Chart: Happiness Factor Profile")

    row = data.iloc[0]
    profile = pd.DataFrame(
        {
            "factor": FACTOR_COLUMNS,
            "factor_label": [FACTOR_LABELS[col] for col in FACTOR_COLUMNS],
            "value": [row[col] for col in FACTOR_COLUMNS],
        }
    ).sort_values("value", ascending=False)
    profile["role"] = np.where(profile["value"] == profile["value"].max(), "Highest", "Other")

    fig = px.bar(
        profile,
        x="factor_label",
        y="value",
        color="role",
        text=profile["value"].round(2),
        color_discrete_map=HIGHLIGHT_PALETTE,
        category_orders={"factor_label": profile["factor_label"].tolist()},
        hover_data={"factor": False, "factor_label": False, "role": False, "value": ":.3f"},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_color=MARKER_BORDER)
    fig.update_yaxes(range=[0, max(0.1, profile["value"].max() * 1.2)])
    title = f"Column Chart: factor contribution profile for {row['country']} in {year}"
    return finalize_figure(fig, title, "Happiness factor", "Contribution value", y_zero=True, legend_title="Role")


@app.callback(
    Output("bar-chart", "figure"),
    Input("year-control", "value"),
    Input("top-n-control", "value"),
    Input("region-control", "value"),
)
def update_bar_chart(year, top_n, selected_regions):
    data = filter_regions(df[(df["year"] == year) & (df["region"] != "Other")], selected_regions)
    if data.empty:
        return empty_figure("Bar Chart: Happiness Ranking")

    top = data.nlargest(int(top_n), "happiness_score").sort_values("happiness_score")
    top["role"] = np.where(top["happiness_score"] == top["happiness_score"].max(), "Highest", "Other")

    fig = px.bar(
        top,
        y="country",
        x="happiness_score",
        color="role",
        text=top["happiness_score"].round(2),
        orientation="h",
        color_discrete_map=HIGHLIGHT_PALETTE,
        hover_data={"region": True, "role": False, "happiness_score": ":.3f"},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_color=MARKER_BORDER)
    fig.update_xaxes(range=[0, max(0.1, top["happiness_score"].max() * 1.16)])
    title = f"Bar Chart: top {len(top)} happiness ranking in {year}"
    return finalize_figure(fig, title, "Happiness Score", "Country", x_zero=True, legend_title="Role")


# =========================
# Week 2 callbacks
# =========================
@app.callback(
    Output("stacked-column-chart", "figure"),
    Input("year-control", "value"),
    Input("region-control", "value"),
)
def update_stacked_column(year, selected_regions):
    grouped = year_region_factor_totals(year, selected_regions)
    if grouped.empty:
        return empty_figure("Stacked Column Chart: Regional Factor Mix")

    # Use only 4 factors for this chart
    chart_factors = FACTOR_COLUMNS[:4]
    
    fig = go.Figure()
    for factor in chart_factors:
        segment_labels = [f"{value:.1f}" if value > 0 else "" for value in grouped[factor]]
        fig.add_trace(
            go.Bar(
                name=FACTOR_LABELS[factor],
                x=grouped["short_region"],
                y=grouped[factor],
                text=segment_labels,
                textposition="inside",
                insidetextanchor="middle",
                textfont={"color": MARKER_BORDER, "size": 10},
                cliponaxis=False,
                marker={"color": FACTOR_PALETTE[factor], "line": {"color": MARKER_BORDER, "width": 0.6}},
                customdata=np.stack([grouped["region"], grouped["country_count"]], axis=-1),
                hovertemplate="Region: %{customdata[0]}<br>"
                + "Countries: %{customdata[1]}<br>"
                + f"{FACTOR_LABELS[factor]}: "
                + "%{y:.3f}<extra></extra>",
            )
        )

    for _, row in grouped.iterrows():
        row_total = row[chart_factors].sum()
        fig.add_annotation(
            x=row["short_region"],
            y=row_total,
            text=f"Total {row_total:.1f}",
            showarrow=False,
            yshift=12,
            font={"size": 11, "color": TEXT_COLOR},
        )

    fig.update_layout(barmode="stack", uniformtext_minsize=8, uniformtext_mode="hide")
    title = f"Stacked Column Chart: total happiness factor composition by region in {year}"
    fig = finalize_figure(fig, title, "Region", "Total contribution", y_zero=True, legend_title="Factor")
    # Calculate max based on only the 4 factors used in chart
    max_total = grouped[chart_factors].sum(axis=1).max()
    fig.update_yaxes(range=[0, max(0.1, max_total * 1.18)])
    return fig


@app.callback(
    Output("stacked-bar-chart", "figure"),
    Input("year-control", "value"),
    Input("region-control", "value"),
    Input("threshold-control", "value"),
)
def update_stacked_bar(year, selected_regions, threshold):
    data = filter_regions(df[(df["year"] == year) & (df["region"] != "Other")], selected_regions)
    if data.empty:
        return empty_figure("Stacked Bar Chart: Regional Happiness Class Mix")

    data["happiness_class"] = np.where(
        data["happiness_score"] >= threshold,
        f"At/above {threshold:.2f}",
        f"Below {threshold:.2f}",
    )
    grouped = (
        data.groupby(["region", "happiness_class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    class_order = [f"Below {threshold:.2f}", f"At/above {threshold:.2f}"]
    for class_name in class_order:
        if class_name not in grouped:
            grouped[class_name] = 0
    grouped["total_count"] = grouped[class_order].sum(axis=1)
    grouped["above_count"] = grouped[f"At/above {threshold:.2f}"]
    grouped = grouped.sort_values(["total_count", "above_count"], ascending=True)
    if grouped.empty:
        return empty_figure("Stacked Bar Chart: Regional Happiness Class Mix")

    fig = go.Figure()
    class_colors = {
        f"Below {threshold:.2f}": "#bfdbfe",
        f"At/above {threshold:.2f}": "#bbf7d0",
    }
    for class_name in class_order:
        segment_labels = [str(int(value)) if value > 0 else "" for value in grouped[class_name]]
        fig.add_trace(
            go.Bar(
                name=class_name,
                y=grouped["region"],
                x=grouped[class_name],
                orientation="h",
                text=segment_labels,
                textposition="inside",
                insidetextanchor="middle",
                textfont={"color": MARKER_BORDER, "size": 11},
                cliponaxis=False,
                marker={"color": class_colors[class_name], "line": {"color": MARKER_BORDER, "width": 0.7}},
                customdata=grouped["total_count"],
                hovertemplate="Region: %{y}<br>"
                + "Total countries: %{customdata}<br>"
                + f"{class_name}: "
                + "%{x}<extra></extra>",
            )
        )

    for _, row in grouped.iterrows():
        fig.add_annotation(
            x=row["total_count"],
            y=row["region"],
            text=f"{int(row['total_count'])}",
            showarrow=False,
            xshift=18,
            font={"size": 11, "color": TEXT_COLOR},
        )

    fig.update_layout(barmode="stack", uniformtext_minsize=8, uniformtext_mode="hide")
    title = f"Stacked Bar Chart: country count split by happiness threshold in {year}"
    fig = finalize_figure(fig, title, "Number of countries", "Region", x_zero=True, legend_title="Happiness class")
    fig.update_xaxes(range=[0, max(1, grouped["total_count"].max() * 1.18)])
    return fig


@app.callback(
    Output("clustered-column-chart", "figure"),
    Input("region-control", "value"),
)
def update_clustered_column(selected_regions):
    grouped = grouped_region_year_metric("happiness_score", selected_regions)
    if grouped.empty:
        return empty_figure("Clustered Column Chart: Regional Scores Over Time")

    fig = px.bar(
        grouped,
        x="year",
        y="happiness_score",
        color="region",
        barmode="group",
        color_discrete_map=REGION_PALETTE,
        category_orders={"region": REGIONS},
        hover_data={"happiness_score": ":.3f"},
    )
    fig.update_traces(marker_line_color=MARKER_BORDER, marker_line_width=0.5)
    fig.update_xaxes(type="category")
    shorten_region_legend(fig)
    return finalize_figure(
        fig,
        "Clustered Column Chart: regional happiness scores side by side",
        "Year",
        "Average Happiness Score",
        y_zero=True,
        legend_title="Region",
    )


@app.callback(
    Output("clustered-bar-chart", "figure"),
    Input("year-control", "value"),
    Input("region-control", "value"),
)
def update_clustered_bar(year, selected_regions):
    grouped = year_region_factors(year, selected_regions).sort_values("happiness_score")
    if grouped.empty:
        return empty_figure("Clustered Bar Chart: GDP vs Social Support")

    melted = grouped.melt(
        id_vars=["region"],
        value_vars=["gdp_per_capita", "social_support"],
        var_name="factor",
        value_name="value",
    )
    melted["factor"] = melted["factor"].map(FACTOR_LABELS)

    fig = px.bar(
        melted,
        y="region",
        x="value",
        color="factor",
        text="value",
        orientation="h",
        barmode="group",
        color_discrete_map={"GDP": FACTOR_PALETTE["gdp_per_capita"], "Social": FACTOR_PALETTE["social_support"]},
        hover_data={"value": ":.3f"},
    )
    fig.update_traces(
        texttemplate="%{x:.2f}",
        textposition="outside",
        textfont={"color": TEXT_COLOR, "size": 11},
        cliponaxis=False,
        marker_line_color=MARKER_BORDER,
        marker_line_width=0.5,
    )
    title = f"Clustered Bar Chart: GDP vs social support by region in {year}"
    fig = finalize_figure(fig, title, "Average contribution", "Region", x_zero=True, legend_title="Factor")
    fig.update_xaxes(range=[0, max(0.1, melted["value"].max() * 1.22)])
    return fig


# =========================
# Weeks 3 and 4 callbacks
# =========================
@app.callback(
    Output("scatter-chart", "figure"),
    Input("year-control", "value"),
    Input("region-control", "value"),
)
def update_scatter(year, selected_regions):
    data = filter_regions(df[(df["year"] == year) & (df["region"] != "Other")], selected_regions)
    if data.empty:
        return empty_figure("Scatter Chart: Social Support and Happiness")

    x_metric = "social_support"
    corr = data[x_metric].corr(data["happiness_score"])
    plot_data, outlier, x_vals, y_vals = add_positive_residual_outlier(
        data,
        x_metric,
        "happiness_score",
    )
    fig = px.scatter(
        plot_data,
        x=x_metric,
        y="happiness_score",
        color="point_role",
        color_discrete_map=OUTLIER_PALETTE,
        hover_name="country",
        hover_data={
            "region": True,
            "point_role": False,
            "residual": ":.3f",
            x_metric: ":.3f",
            "happiness_score": ":.3f",
        },
    )
    for trace in fig.data:
        if trace.name == "Outlier":
            trace.marker.update(size=15, opacity=0.98, line={"color": MARKER_BORDER, "width": 1.4})
        else:
            trace.marker.update(size=9, opacity=0.8, line={"color": MARKER_BORDER, "width": 0.7})

    if x_vals is not None:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Trend",
                line={"color": GUIDE_LINE, "width": 2, "dash": "dash"},
                hoverinfo="skip",
            )
        )
    if outlier is not None:
        fig.add_annotation(
            x=outlier[x_metric],
            y=outlier["happiness_score"],
            text=f"Outlier: {outlier['country']}",
            showarrow=True,
            arrowhead=2,
            ax=28,
            ay=-34,
            bgcolor=LEGEND_BG,
            bordercolor=LEGEND_BORDER,
            font={"size": 11, "color": TEXT_COLOR},
        )

    title = f"Scatter Chart: social support vs happiness in {year} (r={corr:.2f})"
    return finalize_figure(fig, title, "Social Support", "Happiness Score", x_zero=True, y_zero=True, legend_title="Point type")


@app.callback(
    Output("bubble-chart", "figure"),
    Input("year-control", "value"),
    Input("region-control", "value"),
)
def update_bubble(year, selected_regions):
    # Use the full selected year for defensible global residual outliers.
    # Region filtering can hide clearer positive cases such as Costa Rica.
    data = df[(df["year"] == year) & (df["region"] != "Other")].copy()
    if data.empty:
        return empty_figure("Bubble Chart: Life Expectancy, Happiness, and GDP")

    plot_data, outlier, x_vals, y_vals = add_positive_residual_outlier(
        data,
        "gdp_per_capita",
        "happiness_score",
    )
    plot_data["bubble_weight"] = np.sqrt(plot_data["life_expectancy"].clip(lower=0) + 0.02)
    fig = px.scatter(
        plot_data,
        x="gdp_per_capita",
        y="happiness_score",
        size="bubble_weight",
        color="point_role",
        color_discrete_map=OUTLIER_PALETTE,
        hover_name="country",
        size_max=24,
        hover_data={
            "region": True,
            "point_role": False,
            "residual": ":.3f",
            "gdp_per_capita": ":.3f",
            "life_expectancy": ":.3f",
            "bubble_weight": False,
            "happiness_score": ":.3f",
        },
    )
    for trace in fig.data:
        if trace.name == "Outlier":
            trace.marker.update(opacity=0.98, line={"color": MARKER_BORDER, "width": 1.5})
        else:
            trace.marker.update(opacity=0.46, line={"color": MARKER_BORDER, "width": 0.7})
    if x_vals is not None:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Trend",
                line={"color": GUIDE_LINE, "width": 2, "dash": "dash"},
                hoverinfo="skip",
            )
        )
    if outlier is not None:
        fig.add_annotation(
            x=outlier["gdp_per_capita"],
            y=outlier["happiness_score"],
            text=f"Outlier: {outlier['country']}",
            showarrow=True,
            arrowhead=2,
            ax=28,
            ay=-34,
            bgcolor=LEGEND_BG,
            bordercolor=LEGEND_BORDER,
            font={"size": 11, "color": TEXT_COLOR},
        )
    title = f"Bubble Chart: GDP vs happiness with life-expectancy bubble weight in {year}"
    return finalize_figure(fig, title, "GDP per Capita", "Happiness Score", x_zero=True, y_zero=True, legend_title="Point type")


# =========================
# Weeks 5, 6, and 7 callbacks
# =========================
@app.callback(
    Output("histogram-chart", "figure"),
    Input("year-control", "value"),
    Input("metric-control", "value"),
)
def update_histogram(year, metric):
    data = df[(df["year"] == year) & (df["region"] != "Other")].copy()
    if data.empty:
        return empty_figure("KDE Density Plot: Metric Distribution")

    label = METRIC_LABELS[metric]
    values = data[metric].dropna().to_numpy(dtype=float)
    x_grid = np.linspace(values.min(), values.max(), 260)
    density = kde_density(values, x_grid)
    peak_idx = int(np.argmax(density))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="KDE density",
            x=x_grid,
            y=density,
            mode="lines",
            fill="tozeroy",
            line={"color": "#bfdbfe", "width": 3},
            fillcolor="rgba(191, 219, 254, 0.28)",
            hovertemplate=f"{label}: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra>KDE density</extra>",
        )
    )

    mean_value = float(values.mean())
    median_value = float(np.median(values))

    fig.add_vline(
        x=mean_value,
        line_width=2,
        line_dash="dash",
        line_color=GUIDE_LINE,
        annotation_text=f"Mean {mean_value:.2f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=median_value,
        line_width=2,
        line_dash="dot",
        line_color=MEDIAN_LINE,
        annotation_text=f"Median {median_value:.2f}",
        annotation_position="bottom",
    )
    fig.add_annotation(
        x=x_grid[peak_idx],
        y=density[peak_idx],
        text=f"Peak density: {x_grid[peak_idx]:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=25,
        ay=-38,
        bgcolor=LEGEND_BG,
        bordercolor=LEGEND_BORDER,
        font={"size": 11, "color": TEXT_COLOR},
    )
    fig.update_yaxes(range=[0, max(0.001, density.max() * 1.26)])
    title = (
        f"KDE Density Plot: {label} Distribution ({year})"
        f"<br><sup>{len(values)} countries, smooth probability density curve</sup>"
    )
    return finalize_figure(fig, title, label, "Probability density", y_zero=True, show_legend=False, height=450)


@app.callback(
    Output("box-chart", "figure"),
    Input("year-control", "value"),
    Input("metric-control", "value"),
    Input("region-control", "value"),
)
def update_box(year, metric, selected_regions):
    data = filter_regions(df[(df["year"] == year) & (df["region"] != "Other")], selected_regions)
    if data.empty:
        return empty_figure("Box Chart: Regional Spread")

    data = data.assign(short_region=data["region"].map(SHORT_REGION_LABELS).fillna(data["region"]))
    label = METRIC_LABELS[metric]
    medians = data.groupby("short_region")[metric].median().sort_values(ascending=True)
    fig = px.box(
        data,
        x=metric,
        y="short_region",
        color="region",
        points="outliers",
        color_discrete_map=REGION_PALETTE,
        category_orders={"short_region": medians.index.tolist()},
        hover_data={"country": True, "region": True, "short_region": False},
    )
    shorten_region_legend(fig)
    title = f"Box Chart: regional spread of {label.lower()} in {year}"
    return finalize_figure(fig, title, label, "Region", x_zero=True, legend_title="Region")


@app.callback(
    Output("violin-chart", "figure"),
    Input("year-control", "value"),
    Input("metric-control", "value"),
    Input("region-control", "value"),
)
def update_violin(year, metric, selected_regions):
    data = filter_regions(df[(df["year"] == year) & (df["region"] != "Other")], selected_regions)
    if data.empty:
        return empty_figure("Violin Chart: Regional Density")

    data = data.assign(short_region=data["region"].map(SHORT_REGION_LABELS).fillna(data["region"]))
    label = METRIC_LABELS[metric]
    medians = data.groupby("short_region")[metric].median().sort_values(ascending=False)
    fig = px.violin(
        data,
        x="short_region",
        y=metric,
        color="region",
        box=True,
        points=False,
        color_discrete_map=REGION_PALETTE,
        category_orders={"short_region": medians.index.tolist()},
        hover_data={"country": True, "region": True, "short_region": False},
    )
    fig.update_traces(meanline_visible=True, line_color=GUIDE_LINE, opacity=0.82)
    shorten_region_legend(fig)
    for trace in fig.data:
        region = SHORT_TO_REGION.get(trace.name, trace.name)
        color = REGION_PALETTE.get(region)
        if color:
            trace.fillcolor = color
            trace.marker.color = color
    title = f"Violin Chart: density shape of {label.lower()} in {year}"
    return finalize_figure(fig, title, "Region", label, y_zero=True, legend_title="Region")


# =========================
# Weeks 8 and 9 callbacks
# =========================
@app.callback(
    Output("line-chart", "figure"),
    Input("region-control", "value"),
)
def update_line(selected_regions):
    grouped = grouped_region_year_metric("happiness_score", selected_regions)
    if grouped.empty:
        return empty_figure("Line Chart: Regional Happiness Trend")

    fig = go.Figure()
    annotations = []
    for region, series in grouped.groupby("region"):
        series = series.sort_values("year")
        color = REGION_PALETTE.get(region, "#4e79a7")
        fig.add_trace(
            go.Scatter(
                x=series["year"],
                y=series["happiness_score"],
                mode="lines+markers",
                name=region,
                line={"color": color, "width": 2.5},
                marker={"size": 7, "line": {"color": MARKER_BORDER, "width": 0.6}},
                hovertemplate="Year: %{x}<br>Average Happiness Score: %{y:.3f}<extra>"
                + region
                + "</extra>",
            )
        )
        last = series.iloc[-1]
        annotations.append(
            {
                "x": last["year"],
                "y": last["happiness_score"],
                "xref": "x",
                "yref": "y",
                "text": SHORT_REGION_LABELS.get(region, region),
                "showarrow": False,
                "xanchor": "left",
                "xshift": 8,
                "font": {"size": 11, "color": color},
            }
        )

    fig.update_layout(annotations=annotations)
    fig.update_xaxes(dtick=1, range=[min(YEARS), max(YEARS) + 0.85])
    shorten_region_legend(fig)
    return finalize_figure(
        fig,
        "Line Chart: average happiness trend by region",
        "Year",
        "Average Happiness Score",
        show_legend=False,
    )


@app.callback(
    Output("area-chart", "figure"),
    Input("region-control", "value"),
)
def update_area(selected_regions):
    data = filter_regions(df[df["region"] != "Other"], selected_regions)
    if data.empty:
        return empty_figure("Area Chart: Total Happiness Volume Over Time")

    area = (
        data.groupby("year", as_index=False)
        .agg(total_happiness=("happiness_score", "sum"), country_count=("country", "nunique"))
    )

    fig = go.Figure(
        go.Scatter(
            x=area["year"],
            y=area["total_happiness"],
            mode="lines+markers",
            fill="tozeroy",
            name="Total happiness score",
            line={"color": "#bfdbfe", "width": 2.5},
            marker={"size": 7, "line": {"color": MARKER_BORDER, "width": 0.6}},
            fillcolor="rgba(191, 219, 254, 0.30)",
            customdata=area["country_count"],
            hovertemplate="Year: %{x}<br>Total happiness score: %{y:.2f}<br>Countries: %{customdata}<extra></extra>",
        )
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, max(1, area["total_happiness"].max() * 1.18)])
    title = "Area Chart: total happiness score volume over time"
    return finalize_figure(fig, title, "Year", "Total Happiness Score", y_zero=True, show_legend=False)


if __name__ == "__main__":
    app.run(debug=True)
