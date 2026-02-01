import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
import plotly.graph_objects as go
import io
import kaleido
import warnings

st.set_page_config(page_title="Dose-Response Curve Fitter", layout="wide")

st.title("Dose-Response Curve Fitter")

st.write("Upload a CSV with a 'Concentration' column and one or more response columns "
         "(e.g., % Response, Mortality 24h, Viability). Select model, normalize if needed, "
         "and fit curves with statistics.")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Uploaded data:")
    st.dataframe(df)
    
    if 'Concentration' in df.columns:
        priority_keywords = ['%', 'mortality', 'viability', 'response', 'inhibition', 'dead', 'survival', 'ec50', 'ic50']
        
        response_options = [col for col in df.columns if col != 'Concentration']
        
        if response_options:
            def score_column(col_name):
                score = 0
                lower_col = col_name.lower()
                for kw in priority_keywords:
                    if kw in lower_col:
                        score += 1
                return score
            
            response_options_sorted = sorted(response_options, key=lambda col: (-score_column(col), col))
            
            selected_responses = st.multiselect(
                "Select Response column(s) to fit",
                response_options_sorted,
                default=[response_options_sorted[0]] if response_options_sorted else [],
                help="Multi-select for overlaid curves. Suggestions prioritize relevant keywords."
            )
            
            if selected_responses:
                model_choice = st.selectbox(
                    "Select model",
                    ["4-parameter Hill (variable slope)", "3-parameter Hill (fixed slope=1)"],
                    index=0,
                    help="4-param: full flexibility. 3-param: simpler, often sufficient, works with fewer points."
                )
                
                normalize_data = st.checkbox(
                    "Normalize response to 0–100 (multiply by 100 if 0–1 scale)",
                    value=False
                )
                
                conc_unit = st.text_input("Concentration unit (optional)", value="μM")
                resp_unit = st.text_input("Response unit (optional)", value="%")
                
                fig = go.Figure()
                params_dict = {}
                all_fitted_data = []
                
                for idx, resp in enumerate(selected_responses):
                    st.subheader(f"Processing: {resp}")
                    
                    x_data = df['Concentration'].values
                    y_data_raw = df[resp].values
                    
                    if normalize_data:
                        y_data = y_data_raw * 100
                    else:
                        y_data = y_data_raw
                    
                    if np.any(y_data < -50) or np.any(y_data > 200):
                        st.warning(f"Values in {resp} outside typical 0–200 range — ensure normalization if needed.")
                    
                    # Filter positive concentrations for log space
                    valid_mask = x_data > 0
                    if not np.any(valid_mask):
                        st.error(f"No positive concentrations in data for {resp} — log-scale fitting requires x > 0.")
                        continue
                    
                    x_pos = x_data[valid_mask]
                    y_pos = y_data[valid_mask]
                    
                    # Define model based on choice
                    if model_choice == "4-parameter Hill (variable slope)":
                        def model_func(x, top, bottom, ic50, hill):
                            return bottom + (top - bottom) / (1 + (x / ic50)**hill)
                        p0 = [np.max(y_pos) + 0.1*(np.max(y_pos)-np.min(y_pos)),
                              np.min(y_pos) - 0.1*(np.max(y_pos)-np.min(y_pos)),
                              np.median(x_pos),
                              1.0]
                    else:  # 3-param fixed hill=1
                        def model_func(x, top, bottom, ic50):
                            return bottom + (top - bottom) / (1 + (x / ic50))
                        p0 = [np.max(y_pos) + 0.1*(np.max(y_pos)-np.min(y_pos)),
                              np.min(y_pos) - 0.1*(np.max(y_pos)-np.min(y_pos)),
                              np.median(x_pos)]
                    
                    try:
                        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in power")
                        
                        params, pcov = curve_fit(model_func, x_pos, y_pos, p0=p0, maxfev=10000)
                        
                        # Confidence intervals (95%)
                        n = len(y_pos)
                        p = len(params)
                        dof = max(0, n - p)
                        t_val = t.ppf(0.975, dof) if dof > 0 else 0
                        perr = np.sqrt(np.diag(pcov))
                        ci_low = params - t_val * perr
                        ci_high = params + t_val * perr
                        
                        params_dict[resp] = (params, ci_low, ci_high)
                        
                        x_fit = np.logspace(np.log10(min(x_pos)), np.log10(max(x_pos)), 200)
                        y_fit = model_func(x_fit, *params)
                        
                        color = f"rgb({(idx*100)%255}, {(idx*150)%255}, {(idx*200)%255})"
                        fig.add_trace(go.Scatter(x=x_pos, y=y_pos, mode='markers',
                                                 name=f'Data: {resp}', marker=dict(size=10, color=color)))
                        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
                                                 name=f'Fit: {resp}', line=dict(width=3, color=color)))
                        
                        # R²
                        y_pred = model_func(x_pos, *params)
                        ss_res = np.sum((y_pos - y_pred)**2)
                        ss_tot = np.sum((y_pos - np.mean(y_pos))**2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
                        
                        st.metric("R² (goodness-of-fit)", f"{r_squared:.4f}" if not np.isnan(r_squared) else "N/A")
                        
                        # Parameters with CIs
                        param_names = ["Top (max response)", "Bottom (min response)", "IC50", "Hill slope"] if len(params) == 4 else ["Top (max response)", "Bottom (min response)", "IC50"]
                        for i, name in enumerate(param_names):
                            unit = resp_unit if "response" in name.lower() else conc_unit
                            st.write(f"**{name}**: {params[i]:.4f} ({ci_low[i]:.4f} – {ci_high[i]:.4f}) {unit}")
                        
                        # Fitted data
                        fitted_df = pd.DataFrame({'Concentration': x_fit, f'Predicted_{resp}': y_fit})
                        all_fitted_data.append(fitted_df)
                    
                    except Exception as e:
                        st.error(f"Fit failed for {resp}: {str(e)}")
                        continue
                
                if params_dict:
                    fig.update_layout(
                        title="Dose-Response Curve(s)",
                        xaxis_title=f"Concentration ({conc_unit}, log scale)",
                        yaxis_title=f"Response ({resp_unit})",
                        xaxis_type="log",
                        hovermode="x unified",
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # PNG download
                    try:
                        img_bytes = fig.to_image(format="png")
                        st.download_button("Download Plot (PNG)", img_bytes, "dose_response_plot.png", "image/png")
                    except:
                        st.warning("PNG export failed — use interactive plot.")
                    
                    # Parameters CSV
                    param_rows = []
                    for resp, (params, _, _) in params_dict.items():
                        row = {"Response": resp}
                        param_names = ["Top", "Bottom", "IC50", "Hill"] if len(params) == 4 else ["Top", "Bottom", "IC50"]
                        for i, name in enumerate(param_names):
                            row[name] = f"{params[i]:.4f}"
                        param_rows.append(row)
                    params_df = pd.DataFrame(param_rows)
                    csv_buffer = io.StringIO()
                    params_df.to_csv(csv_buffer, index=False)
                    st.download_button("Download Parameters (CSV)", csv_buffer.getvalue(), "parameters.csv", "text/csv")
                    
                    # Fitted points CSV
                    if all_fitted_data:
                        combined_fitted = pd.concat([df.set_index('Concentration') for df in all_fitted_data], axis=1).reset_index()
                        fitted_buffer = io.StringIO()
                        combined_fitted.to_csv(fitted_buffer, index=False)
                        st.download_button("Download Fitted Points (CSV)", fitted_buffer.getvalue(), "fitted_points.csv", "text/csv")
                    
                    st.markdown("---")
                    st.caption("Research/educational use only. Not for clinical decisions. Verify with dedicated software.")
                    with st.expander("About the fit"):
                        st.markdown("""
                        - **4-param Hill**: full sigmoidal fit (variable slope).
                        - **3-param Hill**: fixed slope=1 (simpler, often used for limited data).
                        - **R²**: 1 - (SS_res / SS_tot) — higher = better fit (0–1 scale).
                        - **95% CIs**: approximate via covariance matrix + t-distribution.
                        - **Top**: maximum response (asymptote at high doses).
                        - **Bottom**: minimum response (asymptote at low doses).
                        - **IC50**: concentration for 50% effect between top and bottom.
                        - **Hill slope**: steepness of the curve (higher = sharper transition).
                        """)
            
            else:
                st.info("Select at least one response column.")
        
        else:
            st.warning("No response columns found.")
    
    else:
        st.warning("CSV requires 'Concentration' column.")
else:
    st.info("Upload a CSV to start.")
