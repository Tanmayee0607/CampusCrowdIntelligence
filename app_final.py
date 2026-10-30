
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Campus Crowd Intelligence",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. COLOUR PALETTE
# =========================================================

NAVY = "#0F172A"
TEAL = "#0D9488"
BLUE = "#163A5F"
MINT = "#5EEAD4"
BACKGROUND = "#F8FAFC"

GREEN = "#16A34A"
YELLOW = "#FACC15"
RED = "#DC2626"


# =========================================================
# 3. PROFESSIONAL WEBSITE DESIGN
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F8FAFC;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

[data-testid="stSidebar"] {
    background-color: #0F172A;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] hr {
    border-color: #334155;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #E2E8F0 !important;
    font-size: 14px;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #0F172A;
    font-weight: 800;
}

p {
    color: #475569;
}

.hero {
    background: linear-gradient(115deg, #0F172A 0%, #163A5F 65%, #0D9488 100%);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #DCE7F0;
    font-size: 15px;
    margin-bottom: 0;
}

.section-heading {
    color: #0F172A;
    font-size: 21px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 14px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

.metric-label {
    color: #64748B;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.4px;
}

.metric-value {
    color: #0F172A;
    font-size: 27px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.metric-caption {
    color: #0D9488;
    font-size: 12px;
    margin-top: 5px;
}

.info-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

.recommendation-card {
    background: linear-gradient(120deg, #0F172A, #163A5F, #0D9488);
    border-radius: 18px;
    padding: 27px;
    color: white;
    margin-top: 15px;
    margin-bottom: 15px;
}

.recommendation-card h2 {
    color: white;
    font-size: 32px;
}

.recommendation-card p {
    color: #E2F3F4;
}

.small-label {
    color: #64748B;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button {
    background: #0D9488;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #0F766E;
    color: white;
    border: none;
}

.stDownloadButton > button {
    background: #163A5F;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

.stDownloadButton > button:hover {
    background: #0F172A;
    color: white;
    border: none;
}

div[data-testid="stSelectbox"] label {
    color: #0F172A;
    font-weight: 700;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    possible_files = [
        Path("campus_crowd_dataset.csv"),
        Path("campus_crowd_dataset (1).csv")
    ]

    csv_file = None

    for file in possible_files:
        if file.exists():
            csv_file = file
            break

    if csv_file is None:
        matching_files = list(
            Path(".").glob("campus_crowd_dataset*.csv")
        )

        if matching_files:
            csv_file = matching_files[0]

    if csv_file is None:
        return None

    data = pd.read_csv(csv_file)

    required_columns = [
        "date",
        "hour",
        "location",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "temperature",
        "crowd_count",
        "is_start_of_semester",
        "is_during_semester"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        st.error(
            "Missing dataset columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    numeric_columns = [
        "hour",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "temperature",
        "crowd_count",
        "is_start_of_semester",
        "is_during_semester"
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "hour",
            "location",
            "crowd_count",
            "temperature"
        ]
    )

    data["hour"] = data["hour"].astype(int)

    return data


df = load_data()

if df is None:
    st.error(
        "Dataset not found! Keep your CSV file in the same "
        "folder as app.py."
    )
    st.stop()

if df.empty:
    st.error("The dataset has no usable records.")
    st.stop()


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def metric_card(label, value, caption):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


def section_heading(title):

    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


def get_crowd_status(value, facility_data):

    low_limit = facility_data["crowd_count"].quantile(0.33)
    high_limit = facility_data["crowd_count"].quantile(0.67)

    if value <= low_limit:
        return "Low", GREEN

    elif value <= high_limit:
        return "Moderate", YELLOW

    else:
        return "High", RED


def format_hour(hour):

    return pd.Timestamp(
        "2000-01-01"
    ).replace(
        hour=int(hour)
    ).strftime("%I:%M %p")


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("## 🏫 CampusIQ")

    st.markdown("""
    <p style="color:#94A3B8; font-size:11px;
    font-weight:700; letter-spacing:1px;">
        CAMPUS CROWD INTELLIGENCE
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p style="color:#94A3B8; font-size:11px;
    font-weight:700;">
        NAVIGATION
    </p>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Crowd Analysis",
            "Crowd Heatmap",
            "Model Comparison",
            "Create My Visit Recommendations"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <p style="color:#94A3B8; font-size:11px;
    font-weight:700;">
        PROJECT STATUS
    </p>
    <p style="color:#5EEAD4; font-size:13px;">
        ● Dataset loaded
    </p>
    <p style="color:#CBD5E1; font-size:11px;">
        Academic demonstration version
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p style="color:#94A3B8; font-size:11px;">
        Campus Crowd Intelligence
        <br>
        Data Science Project
    </p>
    """, unsafe_allow_html=True)


# =========================================================
# 7. COMMON HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>Campus Crowd Intelligence</h1>
    <p>
        Understand campus crowd patterns, explore facility usage,
        and discover smarter times to visit.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    section_heading("Campus at a Glance")

    total_records = len(df)
    total_facilities = df["location"].nunique()
    average_crowd = df["crowd_count"].mean()
    maximum_crowd = df["crowd_count"].max()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "TOTAL RECORDS",
            f"{total_records:,}",
            "Dataset observations"
        )

    with col2:
        metric_card(
            "FACILITIES",
            str(total_facilities),
            "Campus locations"
        )

    with col3:
        metric_card(
            "AVERAGE CROWD",
            f"{average_crowd:.1f}",
            "People per observation"
        )

    with col4:
        metric_card(
            "HIGHEST CROWD",
            f"{maximum_crowd:.0f}",
            "Maximum recorded sample"
        )

    section_heading("Average Crowd by Facility")

    facility_average = (
        df.groupby("location")["crowd_count"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))

    sns.barplot(
        x=facility_average.values,
        y=facility_average.index,
        color=TEAL,
        ax=ax
    )

    ax.set_xlabel("Average Crowd Count")
    ax.set_ylabel("")
    ax.set_title(
        "Facility-wise Average Crowd",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

   


# =========================================================
# 9. CROWD ANALYSIS PAGE
# =========================================================

elif page == "Crowd Analysis":

    section_heading("Explore Facility Crowd Patterns")

    selected_facility = st.selectbox(
        "Choose a campus facility",
        sorted(df["location"].unique())
    )

    facility_data = df[
        df["location"] == selected_facility
    ].copy()

    average = facility_data["crowd_count"].mean()

    hourly_data = (
        facility_data.groupby("hour")["crowd_count"]
        .mean()
        .sort_index()
    )

    busiest_hour = hourly_data.idxmax()
    quietest_hour = hourly_data.idxmin()

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "AVERAGE CROWD",
            f"{average:.1f}",
            selected_facility
        )

    with c2:
        metric_card(
            "BUSIEST HOUR",
            format_hour(busiest_hour),
            "Highest average crowd"
        )

    with c3:
        metric_card(
            "QUIETEST HOUR",
            format_hour(quietest_hour),
            "Lowest average crowd"
        )

    section_heading("Hourly Crowd Trend")

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        hourly_data.index,
        hourly_data.values,
        marker="o",
        linewidth=2.5,
        color=TEAL
    )

    ax.fill_between(
        hourly_data.index,
        hourly_data.values,
        alpha=0.12,
        color=TEAL
    )

    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Average Crowd")
    ax.set_xticks(sorted(hourly_data.index.unique()))

    ax.set_title(
        f"Hourly Crowd Pattern - {selected_facility}",
        loc="left",
        fontweight="bold",
        color=NAVY
    )

    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    section_heading("Crowd Distribution")

    fig, ax = plt.subplots(figsize=(11, 3.5))

    sns.histplot(
        facility_data["crowd_count"],
        bins=20,
        kde=True,
        color=TEAL,
        ax=ax
    )

    ax.set_xlabel("Crowd Count")
    ax.set_ylabel("Frequency")
    ax.set_title(
        "Distribution of Crowd Counts",
        loc="left",
        fontweight="bold",
        color=NAVY
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    section_heading("Weekday vs Weekend")

    day_comparison = (
        facility_data.groupby("is_weekend")["crowd_count"]
        .mean()
        .reset_index()
    )

    day_comparison["Day Type"] = day_comparison[
        "is_weekend"
    ].map({
        0: "Weekday",
        1: "Weekend"
    })

    fig, ax = plt.subplots(figsize=(8, 3.5))

    sns.barplot(
        data=day_comparison,
        x="Day Type",
        y="crowd_count",
        color=BLUE,
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("Average Crowd")
    ax.set_title(
        "Weekday and Weekend Comparison",
        loc="left",
        fontweight="bold",
        color=NAVY
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =========================================================
# 10. CROWD HEATMAP PAGE
# =========================================================

elif page == "Crowd Heatmap":

    section_heading("Campus Crowd Intensity")

    st.write(
        "Explore how average crowd levels change across "
        "facilities and hours of the day."
    )

    heatmap_data = df.pivot_table(
        values="crowd_count",
        index="location",
        columns="hour",
        aggfunc="mean"
    )

    # Determine facility-specific crowd thresholds
    low_limits = df.groupby("location")[
        "crowd_count"
    ].quantile(0.33)

    high_limits = df.groupby("location")[
        "crowd_count"
    ].quantile(0.67)

    # Convert crowd values into levels for colouring
    level_data = heatmap_data.copy()

    for facility in level_data.index:

        low_limit = low_limits.loc[facility]
        high_limit = high_limits.loc[facility]

        level_data.loc[facility] = level_data.loc[
            facility
        ].apply(
            lambda value:
                0 if value <= low_limit
                else 1 if value <= high_limit
                else 2
        )

    from matplotlib.colors import ListedColormap, BoundaryNorm

    crowd_cmap = ListedColormap([
        GREEN,
        YELLOW,
        RED
    ])

    crowd_norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5],
        crowd_cmap.N
    )

    fig, ax = plt.subplots(figsize=(13, 5))

    sns.heatmap(
        level_data,
        annot=heatmap_data,
        fmt=".0f",
        cmap=crowd_cmap,
        norm=crowd_norm,
        linewidths=0.7,
        linecolor="white",
        ax=ax,
        cbar=False
    )

    ax.set_title(
        "Facility vs Hour Crowd Heatmap",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY
    )

    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Campus Facility")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    section_heading("Crowd Level Legend")

    l1, l2, l3 = st.columns(3)

    with l1:
        st.markdown(
            f"<div style='background:{GREEN}; padding:12px;"
            "border-radius:9px; color:white; text-align:center;"
            "font-weight:700;'>LOW CROWD</div>",
            unsafe_allow_html=True
        )

    with l2:
        st.markdown(
            f"<div style='background:{YELLOW}; padding:12px;"
            "border-radius:9px; color:#422006; text-align:center;"
            "font-weight:700;'>MODERATE CROWD</div>",
            unsafe_allow_html=True
        )

    with l3:
        st.markdown(
            f"<div style='background:{RED}; padding:12px;"
            "border-radius:9px; color:white; text-align:center;"
            "font-weight:700;'>HIGH CROWD</div>",
            unsafe_allow_html=True
        )

    st.caption(
        "The colours show crowd categories based on each facility's "
        "own distribution. Numbers show average crowd counts."
    )


# =========================================================
# 11. MODEL COMPARISON PAGE
# =========================================================

elif page == "Model Comparison":

    section_heading("Crowd Prediction Model Evaluation")

    st.write(
        "Compare three Linear Regression models to understand "
        "how adding more input features affects prediction performance."
    )

    feature_sets = {
        "Model 1 - Basic": [
            "hour",
            "temperature"
        ],

        "Model 2 - Facility": [
            "hour",
            "temperature",
            "location"
        ],

        "Model 3 - Full Features": [
            "hour",
            "temperature",
            "location",
            "day_of_week",
            "is_weekend",
            "is_holiday",
            "is_start_of_semester",
            "is_during_semester"
        ]
    }

    results = []

    for model_name, features in feature_sets.items():

        X = df[features]
        y = df["crowd_count"]

        
        categorical_features = [
            col for col in features
            if (
                X[col].dtype == "object"
                or isinstance(X[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(X[col])
            )
        ]

        numeric_features = [
            col for col in features
            if col not in categorical_features
        ]

        transformers = []

        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features
                )
            )

        if numeric_features:
            transformers.append(
                (
                    "numeric",
                    "passthrough",
                    numeric_features
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers
        )


        numeric_features = [
            col for col in features
            if col not in categorical_features
        ]

        transformers = []

        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features
                )
            )

        if numeric_features:
            transformers.append(
                (
                    "numeric",
                    "passthrough",
                    numeric_features
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "Model": model_name,
            "MAE": mae,
            "R2 Score": r2
        })

    results_df = pd.DataFrame(results)

    best_model = results_df.loc[
        results_df["MAE"].idxmin()
    ]

    c1, c2 = st.columns(2)

    with c1:
        metric_card(
            "BEST MODEL BY MAE",
            best_model["Model"],
            "Lowest prediction error"
        )

    with c2:
        metric_card(
            "BEST MODEL MAE",
            f'{best_model["MAE"]:.2f}',
            "Average absolute error"
        )

    section_heading("Performance Comparison")

    st.dataframe(
        results_df.style.format({
            "MAE": "{:.3f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        data=results_df,
        x="Model",
        y="MAE",
        color=TEAL,
        ax=ax
    )

    ax.set_title(
        "Mean Absolute Error Comparison",
        loc="left",
        fontweight="bold",
        color=NAVY
    )

    ax.set_ylabel("MAE (lower is better)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="info-card">
        <b style="color:#0F172A;">Understanding the metrics</b>
        <p>
        <b>MAE:</b> Measures the average absolute difference
        between actual and predicted crowd counts. Lower is better.
        </p>
        <p>
        <b>R² Score:</b> Indicates how much variation in the
        target is explained by the model. Higher is generally better.
        </p>
    </div>
    """, unsafe_allow_html=True)



# =========================================================
# 12. CREATE MY VISIT RECOMMENDATIONS
# =========================================================

elif page == "Create My Visit Recommendations":

    section_heading("Create My Visit Recommendations")

    st.write(
        "Generate a personalised visit plan using the crowd "
        "patterns of your selected campus facility."
    )

    # Remember recommendation history during this session
    if "recommended_slots" not in st.session_state:
        st.session_state.recommended_slots = {}

    selected_facility = st.selectbox(
        "Choose your campus facility",
        sorted(df["location"].unique())
    )

    facility_data = df[
        df["location"] == selected_facility
    ].copy()

    hourly_crowd = (
        facility_data.groupby("hour")["crowd_count"]
        .mean()
        .sort_values()
    )

    st.markdown("""
    <div class="info-card">
        <b style="color:#0F172A;">Your personalised visit planner</b>
        <p>
        Select a facility and create a recommendation report.
        The system considers average crowd levels and avoids
        repeating previously recommended hours during this session.
        </p>
    </div>
    """, unsafe_allow_html=True)

    used_hours = st.session_state.recommended_slots.get(
        selected_facility, []
    )

    st.caption(
        f"{len(used_hours)} different time slot(s) already "
        f"recommended for {selected_facility} in this session."
    )

    # Generate recommendation
    if st.button(
        "✨ Create My Visit Recommendations",
        use_container_width=True
    ):

        available_hours = hourly_crowd[
            ~hourly_crowd.index.isin(used_hours)
        ]

        cycle_restarted = False

        # Start a new cycle when all hours have been used
        if available_hours.empty:

            used_hours = []
            available_hours = hourly_crowd
            cycle_restarted = True

        # Select least crowded unused hour
        recommended_hour = int(
            available_hours.idxmin()
        )

        expected_crowd = float(
            available_hours.loc[recommended_hour]
        )

        # Save the recommended hour
        used_hours.append(recommended_hour)

        st.session_state.recommended_slots[
            selected_facility
        ] = used_hours

        visit_time = format_hour(recommended_hour)

        crowd_status, crowd_colour = get_crowd_status(
            expected_crowd,
            facility_data
        )

        # Save latest report
        st.session_state.visit_report = {
            "facility": selected_facility,
            "time": visit_time,
            "hour": recommended_hour,
            "crowd": expected_crowd,
            "status": crowd_status,
            "colour": crowd_colour,
            "restarted": cycle_restarted
        }

    # Display generated report
    if "visit_report" in st.session_state:

        report = st.session_state.visit_report

        if report["facility"] == selected_facility:

            if report["restarted"]:
                st.info(
                    "All available hours were already recommended. "
                    "A new recommendation cycle has started."
                )

            st.markdown("---")

            section_heading("Your Visit Recommendation Report")

            st.markdown(f"""
            <div class="recommendation-card">
                <p style="color:#5EEAD4; font-weight:700;">
                    PERSONAL VISIT PLAN
                </p>
                <h2>{report["time"]}</h2>
                <p>Recommended visit time for {report["facility"]}</p>
                <div style="
                    display:inline-block;
                    background:{report["colour"]};
                    color:#FFFFFF;
                    padding:8px 15px;
                    border-radius:8px;
                    font-weight:700;">
                    {report["status"]} CROWD
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card(
                    "FACILITY",
                    report["facility"],
                    "Selected location"
                )

            with col2:
                metric_card(
                    "EXPECTED CROWD",
                    f'{report["crowd"]:.1f}',
                    "Average people per observation"
                )

            with col3:
                metric_card(
                    "CROWD STATUS",
                    report["status"],
                    "Based on facility distribution"
                )

            st.markdown("---")

            section_heading("Your Recommendation History")

            history = st.session_state.recommended_slots.get(
                selected_facility, []
            )

            history_rows = []

            for index, hour in enumerate(history, start=1):

                crowd_value = float(
                    hourly_crowd.loc[hour]
                )

                formatted_time = format_hour(hour)

                status, colour = get_crowd_status(
                    crowd_value,
                    facility_data
                )

                history_rows.append({
                    "Request": index,
                    "Suggested Time": formatted_time,
                    "Average Crowd": round(crowd_value, 1),
                    "Crowd Level": status
                })

            history_df = pd.DataFrame(history_rows)

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

            # Create downloadable report
            report_text = f"""
CAMPUS CROWD INTELLIGENCE
PERSONALISED VISIT RECOMMENDATION REPORT

Facility: {report["facility"]}
Recommended Visit Time: {report["time"]}
Expected Average Crowd: {report["crowd"]:.1f} people
Crowd Status: {report["status"]}

RECOMMENDATION HISTORY
"""

            for row in history_rows:

                report_text += (
                    f'\n{row["Request"]}. '
                    f'{row["Suggested Time"]} | '
                    f'Average Crowd: {row["Average Crowd"]} | '
                    f'Crowd Level: {row["Crowd Level"]}'
                )

            report_text += """

NOTE:
This report is generated using a synthetic academic dataset.
It does not represent live campus occupancy.
Recommendations are based on historical average crowd patterns.
Recommendation history is maintained during the current session.
"""

            st.download_button(
                label="📄 Download My Visit Report",
                data=report_text,
                file_name="my_campus_visit_recommendations.txt",
                mime="text/plain",
                use_container_width=True
            )

    else:

        st.info(
            "Your personalised visit report will appear here "
            "after you click the Create button."
        )

    st.caption(
        "Recommendations are based on synthetic sample data. "
        "History resets when the dashboard session restarts. "
        "Separate browser sessions do not share recommendation history."
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<p style="text-align:center; color:#94A3B8; font-size:12px;">
    CAMPUS CROWD INTELLIGENCE SYSTEM
    <br>
    Academic Data Science Project | Synthetic Dataset Demonstration
</p>
""", unsafe_allow_html=True)
