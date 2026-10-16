import io
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="🤖 Dynamic Data Agent",
    layout="wide",
    page_icon="📊",
)


# ==========================================
# 1. DATASET REGISTRY
# ==========================================
class DatasetRegistry:
    def __init__(self):
        self._df = None

    def load_data(self, df: pd.DataFrame):
        self._df = df

    def get_data(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("No dataset loaded.")
        return self._df


if "registry" not in st.session_state:
    st.session_state.registry = DatasetRegistry()

registry = st.session_state.registry


# ==========================================
# 2. FILE CONVERSION
# Convert supported uploads to CSV text in memory,
# then load the normalized CSV into a DataFrame.
# No temporary CSV file is written to disk.
# ==========================================
def convert_to_csv(uploaded_file) -> str:
    if uploaded_file is None:
        raise ValueError("Please upload a file first.")

    filename = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if filename.endswith(".csv"):
        df = pd.read_csv(
            uploaded_file,
            encoding="latin-1",
            sep=None,
            engine="python",
            on_bad_lines="skip",
        )
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    elif filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith(".json"):
        df = pd.read_json(uploaded_file)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    elif filename.endswith(".txt"):
        df = pd.read_csv(
            uploaded_file,
            encoding="latin-1",
            sep=None,
            engine="python",
            on_bad_lines="skip",
        )
    else:
        supported = ".csv, .xlsx, .xls, .json, .parquet, .txt"
        raise ValueError(
            f"Unsupported file type. Supported formats: {supported}"
        )

    # Standardize the dataset as CSV text in memory.
    return df.to_csv(index=False)


def load_csv_text(csv_text: str) -> pd.DataFrame:
    """Read standardized CSV text into a pandas DataFrame."""
    return pd.read_csv(io.StringIO(csv_text))


# ==========================================
# 3. THE FIVE ANALYSIS TOOLS
# ==========================================
def inspect_dataset_schema() -> str:
    try:
        df = registry.get_data()
        info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": {
                str(col): {
                    "dtype": str(df[col].dtype),
                    "missing_values": int(df[col].isna().sum()),
                    "unique_values": int(df[col].nunique()),
                }
                for col in df.columns
            },
        }
        return json.dumps(info, indent=2, default=str)
    except Exception as e:
        return f"Error: {str(e)}"


def compute_descriptive_statistics(columns: List[str] = None) -> str:
    try:
        df = registry.get_data()
        target_df = df[columns] if columns else df
        numeric_summary = target_df.describe(include=[np.number]).to_dict()
        cat_summary = {
            str(col): target_df[col].value_counts().head(5).to_dict()
            for col in target_df.select_dtypes(exclude=[np.number]).columns
        }
        return json.dumps(
            {
                "numeric": numeric_summary,
                "categorical_top_5": cat_summary,
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error: {str(e)}"


def execute_filtered_aggregation(
    groupby_cols: List[str],
    metric_col: str,
    agg_func: str = "mean",
    filters: List[dict] = None,
) -> str:
    try:
        df = registry.get_data().copy()

        if filters:
            for f in filters:
                col = f["column"]
                op = f["operator"]
                val = f["value"]

                if col not in df.columns:
                    return f"Error: Unknown filter column '{col}'."

                if op == "==":
                    df = df[df[col].astype(str) == str(val)]
                elif op == ">":
                    df = df[df[col] > float(val)]
                elif op == "<":
                    df = df[df[col] < float(val)]
                else:
                    return f"Error: Unsupported filter operator '{op}'."

        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]

        grouped = (
            df.groupby(groupby_cols)[metric_col]
            .agg(agg_func)
            .reset_index()
        )
        return grouped.to_json(orient="records", indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def evaluate_statistical_correlation(columns: List[str]) -> str:
    try:
        df = registry.get_data()

        if not columns or len(columns) < 2:
            return "Error: Provide at least 2 numeric columns."

        return json.dumps(
            df[columns].corr(method="pearson").to_dict(),
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error: {str(e)}"


def perform_predictive_modeling(
    target_column: str,
    feature_columns: List[str],
) -> str:
    try:
        if isinstance(feature_columns, str):
            feature_columns = [feature_columns]

        df = registry.get_data().copy()
        df = df.dropna(subset=[target_column] + feature_columns)

        if len(df) < 5:
            return "Error: At least 5 complete rows are needed for modeling."

        X = pd.get_dummies(df[feature_columns], drop_first=True)
        y = df[target_column]

        if X.empty or X.shape[1] == 0:
            return "Error: No usable feature columns were found."

        # A test set must contain enough rows for the selected model.
        test_size = max(1, int(round(len(df) * 0.2)))
        if len(df) - test_size < 2:
            return "Error: Not enough rows to split into train and test sets."

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
        )

        if y.dtype == "object" or y.nunique() < 10:
            model = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
            )
            model.fit(X_train, y_train)
            result = {
                "metric": {
                    "Accuracy": float(model.score(X_test, y_test))
                }
            }
        else:
            model = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
            )
            model.fit(X_train, y_train)
            result = {
                "metric": {
                    "R2_Score": float(model.score(X_test, y_test))
                }
            }

        result["feature_importances"] = dict(
            sorted(
                zip(X.columns, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Error: {str(e)}"


TOOL_MAP = {
    "inspect_dataset_schema": lambda kw: inspect_dataset_schema(),
    "compute_descriptive_statistics": lambda kw: (
        compute_descriptive_statistics(kw.get("columns"))
    ),
    "execute_filtered_aggregation": lambda kw: execute_filtered_aggregation(
        kw.get("groupby_cols"),
        kw.get("metric_col"),
        kw.get("agg_func", "mean"),
        kw.get("filters"),
    ),
    "evaluate_statistical_correlation": lambda kw: (
        evaluate_statistical_correlation(kw.get("columns"))
    ),
    "perform_predictive_modeling": lambda kw: perform_predictive_modeling(
        kw.get("target_column"),
        kw.get("feature_columns"),
    ),
}


# ==========================================
# 4. LLM ROUTER ROUTINES
# ==========================================
SYSTEM_PROMPT = """You are an expert Data Analysis Router.
Analyze the dataset by choosing one tool at a time.

Respond ONLY with a valid JSON object containing:
'thought', 'tool_name', and 'tool_args'.

Available tools:
- inspect_dataset_schema: {}
- compute_descriptive_statistics: {"columns": ["col1"]}
- execute_filtered_aggregation:
  {"groupby_cols": ["col1"], "metric_col": "col2",
   "agg_func": "mean" or "sum",
   "filters": [{"column": "col3", "operator": "==", "value": "xyz"}]}
- evaluate_statistical_correlation: {"columns": ["col1", "col2"]}
- perform_predictive_modeling:
  {"target_column": "col1", "feature_columns": ["col2", "col3"]}

Use the dataset schema and previous tool observations to decide the next step.
If you have enough information to answer, return "tool_name": "final_answer"
and put the complete analytical conclusion in "thought".
"""


def query_openai(api_key: str, prompt: str, schema_context: str) -> Dict:
    client = OpenAI(api_key=api_key)
    full_content = (
        f"Dataset Schema and Tool Observations:\n{schema_context}\n\n"
        f"User Question: {prompt}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_content},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def query_ollama(endpoint: str, prompt: str, schema_context: str) -> Dict:
    full_content = (
        f"Dataset Schema and Tool Observations:\n{schema_context}\n\n"
        f"User Question: {prompt}"
    )

    response = requests.post(
        f"{endpoint.rstrip('/')}/api/chat",
        json={
            "model": "llama3",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_content},
            ],
            "format": "json",
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])


# ==========================================
# 5. STREAMLIT INTERFACE
# ==========================================
st.sidebar.title("🛠️ Agent Configuration")
model_provider = st.sidebar.selectbox(
    "LLM Core Engine Provider",
    ["OpenAI (Cloud)", "Ollama (Local Installation)"],
)

openai_key = ""
ollama_url = "http://localhost:11434"

if model_provider == "OpenAI (Cloud)":
    openai_key = st.sidebar.text_input(
        "Enter OpenAI API Key",
        type="password",
    )
else:
    ollama_url = st.sidebar.text_input(
        "Ollama Local Host URL",
        value="http://localhost:11434",
    )

st.title("🤖 Dataset-Agnostic Analytical Agent")
st.markdown(
    "Upload a tabular dataset in a supported format, connect your preferred "
    "LLM provider, and let the agent perform iterative data exploration "
    "and modeling."
)

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "json", "parquet", "txt"],
    help=(
        "Supported formats: CSV, Excel (.xlsx/.xls), JSON, Parquet, and "
        "delimited text (.txt). Files are normalized to CSV in memory."
    ),
    key="dataset_uploader",
)

if uploaded_file is not None:
    try:
        csv_text = convert_to_csv(uploaded_file)
        df = load_csv_text(csv_text)
        registry.load_data(df)
        st.success(
            f"Dataset loaded and normalized to CSV successfully! "
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns."
        )
    except Exception as e:
        st.error(f"Error loading or converting file: {e}")

# Display the loaded dataset even when Streamlit reruns without a new upload.
if registry._df is not None:
    df = registry.get_data()

    tab1, tab2 = st.tabs(
        ["📁 Dataset Preview", "💬 Agent Analysis Chat"]
    )

    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("### Column Data Types")
        st.json({str(col): str(dtype) for col, dtype in df.dtypes.items()})
        st.caption(
            "The dataset is held in memory as a pandas DataFrame. "
            "No intermediate CSV file is saved to disk."
        )

    with tab2:
        query_string = st.text_input(
            "🔍 Ask a quantitative/statistical analysis question:",
            placeholder=(
                "e.g., Which features correlate most with price, "
                "and can we predict it?"
            ),
        )

        if query_string:
            if model_provider == "OpenAI (Cloud)" and not openai_key:
                st.warning(
                    "⚠️ Please provide an OpenAI API key in the sidebar."
                )
            else:
                with st.spinner("Agent running live ReAct loop..."):
                    schema_data = inspect_dataset_schema()
                    max_iterations = 3

                    for i in range(max_iterations):
                        st.markdown(
                            f"---\n#### 🧠 Iteration Step {i + 1}"
                        )

                        try:
                            if model_provider == "OpenAI (Cloud)":
                                action = query_openai(
                                    openai_key,
                                    query_string,
                                    schema_data,
                                )
                            else:
                                action = query_ollama(
                                    ollama_url,
                                    query_string,
                                    schema_data,
                                )

                            thought = action.get(
                                "thought",
                                "No thought provided.",
                            )
                            st.info(f"**Thought:** {thought}")

                            tool_name = action.get("tool_name")

                            if tool_name == "final_answer" or not tool_name:
                                st.success(
                                    f"### 🎯 Final Answer:\n{thought}"
                                )
                                break

                            tool_args = action.get("tool_args", {})
                            st.code(
                                f"Invoking Tool Call: {tool_name}"
                                f"({json.dumps(tool_args)})",
                                language="python",
                            )

                            if tool_name not in TOOL_MAP:
                                st.error(
                                    "Unknown tool configuration error: "
                                    f"{tool_name}"
                                )
                                break

                            observation = TOOL_MAP[tool_name](tool_args)

                            try:
                                parsed_observation = json.loads(observation)
                                st.json(parsed_observation)
                            except (TypeError, json.JSONDecodeError):
                                st.write(observation)

                            schema_data += (
                                f"\n\nObservation from {tool_name}:\n"
                                f"{observation}"
                            )

                        except Exception as loop_error:
                            st.error(
                                "Error executing agent step: "
                                f"{loop_error}"
                            )
                            break
else:
    st.info(
        "Upload a dataset above to begin. The variable is initialized safely, "
        "so the app will not raise a NameError when no file is selected."
    )
