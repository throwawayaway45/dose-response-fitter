import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import io
import kaleido  # Helps ensure engine initializes (some envs need explicit import)
import warnings

st.title("Dose-Response Curve Fitter")

st.write("Upload a CSV file containing dose-response data. "
         "It should have a 'Concentration' column and at least one response column "
         "(e.g. % Response, Mortality 24h, Viability, etc.).")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Your uploaded data:")
    st.dataframe(df)
    
    if 'Concentration' in df.columns:
        # Define keywords for auto-suggest priority
        priority_keywords = ['%', 'mortality', 'viability', 'response', 'inhibition', 'dead', 'survival', 'ec50', 'ic50']
        
        # Get all potential response columns
        response_options = [col for col in df.columns if col != 'Concentration']
        
        if response_options:
            # Score columns based on keywords (higher score = better match)
            def score_column(col_name):
                score = 0
                lower_col = col_name.lower()
                for kw in priority_keywords:
                    if kw in lower_col:
                        score += 1
                return score
            
            # Sort options by score descending, then alphabetically
            response_options_sorted = sorted(response_options, key=lambda col: (-score_column(col), col))
            
            # Multi-select for multiple responses (overlay)
            selected_responses = st.multiselect(
                "Select one or more Response columns to fit (auto-suggested first)",
                response_options_sorted,
                default=[response_options_sorted[0]] if response_options_sorted else [],  # Auto-select top suggested
                help="Choose columns with your measured response values (e.g., % mortality, % inhibition, viability). "
                     "Select multiple to overlay fits on the same plot. Suggestions prioritized by relevant keywords."
            )
            
            if selected_responses:
                # Normalize toggle
                normalize_data = st.checkbox(
                    "Normalize response to 0-100 scale (multiply by 100 if data is 0-1)",
                    value=False,
                    help="Check if your response data is in fraction/proportion (0-1) instead of percentage (0-100)."
                )
                
                # Units inputs
                conc_unit = st.text_input("Concentration unit (optional)", value="μM")
                resp_unit = st.text_input("Response unit (optional)", value="%")
                
                # Prepare single figure for overlays if multiple selected
                fig = go.Figure()
                
                params_dict = {}  # Store params for each response
                all_fitted_data = []  # For export
                
                for idx, selected_response in enumerate(selected_responses):
                    st.info(f"Processing **{selected_response}** as a response variable "
                            f"(assumed to be in {resp_unit} or normalized scale).")
                    
                    x_data = df['Concentration'].values
                    y_data = df[selected_response].values
                    
                    if normalize_data:
                        y_data = y_data * 100
                    
                    # Gentle warning if response looks unusual
                    if np.any(y_data < -50) or np.any(y_data > 200):
                        st.warning(f"For {selected_response}: Some response values are outside the typical 0–200 range. "
                                   "Fits may be less reliable if data is not normalized to 0–100.")
                    
                    def hill_function(x, top, bottom, ic50, hill):
                        return bottom + (top - bottom) / (1 + (x / ic50)**hill)
                    
                    # Safer initial guesses
                    y_max = np.max(y_data)
                    y_min = np.min(y_data)
                    initial_guess = [
                        y_max + 0.1 * (y_max - y_min),
                        y_min - 0.1 * (y_max - y_min),
                        np.median(x_data[x_data > 0]) if np.any(x_data > 0) else np.median(x_data),
                        1.0
                    ]
                    
                    try:
                        # Suppress specific warning
                        warnings.filterwarnings(
                            "ignore",
                            category=RuntimeWarning,
                            message="invalid value encountered in power"
                        )
                        
                        params, _ = curve_fit(hill_function, x_data, y_data, p0=initial_guess, maxfev=5000)
                        params_dict[selected_response] = params
                        
                        x_fit = np.logspace(np.log10(min(x_data[x_data > 0])), np.log10(max(x_data)), 200)
                        y_fit = hill_function(x_fit, *params)
                        
                        # Add to figure
                        color = f"rgb({(idx*100)%255}, {(idx*150)%255}, {(idx*200)%255})"  # Simple color cycle
                        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', 
                                                 name=f'Data: {selected_response}', marker=dict(size=10, color=color)))
                        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', 
                                                 name=f'Fit: {selected_response}', line=dict(width=3, color=color)))
                        
                        # Collect fitted data for export
                        fitted_df = pd.DataFrame({
                            'Concentration': x_fit,
                            f'Predicted_{selected_response}': y_fit
                        })
                        all_fitted_data.append(fitted_df)
                    
                    except Exception as e:
                        st.error(f"Fitting failed for {selected_response}: {str(e)}")
                        continue
                
                if params_dict:
                    # Update layout with units
                    fig.update_layout(
                        title="Dose-Response Curves (Overlaid if multiple selected)",
                        xaxis_title=f"Concentration ({conc_unit}, log scale)",
                        yaxis_title=f"Response ({resp_unit})",
                        xaxis_type="log",
                        hovermode="x unified",
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # PNG download for the combined plot
                    try:
                        img_bytes = fig.to_image(format="png")
                        st.download_button(
                            label="Download Combined Plot (PNG)",
                            data=img_bytes,
                            file_name="dose_response_combined.png",
                            mime="image/png"
                        )
                    except Exception as img_err:
                        st.warning(f"PNG download unavailable: {str(img_err)}. "
                                   "Try installing kaleido (`pip install kaleido`) or use the interactive plot above.")
                    
                    # Parameters display and CSV for all
                    for resp, params in params_dict.items():
                        st.subheader(f"Results for {resp}")
                        st.success(f"Estimated IC50: {params[2]:.4f} {conc_unit}")
                        st.write(f"Top (max response): {params[0]:.2f} {resp_unit}")
                        st.write(f"Bottom (min response): {params[1]:.2f} {resp_unit}")
                        st.write(f"Hill slope: {params[3]:.2f}")
                    
                    # Combined parameters CSV
                    all_params_data = []
                    for resp, params in params_dict.items():
                        all_params_data.append({
                            "Response": resp,
                            "Top (max response)": f"{params[0]:.4f}",
                            "Bottom (min response)": f"{params[1]:.4f}",
                            "IC50": f"{params[2]:.4f}",
                            "Hill slope": f"{params[3]:.4f}"
                        })
                    all_params_df = pd.DataFrame(all_params_data)
                    csv_buffer = io.StringIO()
                    all_params_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download All Fit Parameters (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="fit_parameters_all.csv",
                        mime="text/csv"
                    )
                    
                    # Export fitted data points (combined if multiple)
                    if all_fitted_data:
                        combined_fitted = pd.concat(all_fitted_data, axis=1)
                        combined_fitted = combined_fitted.loc[:, ~combined_fitted.columns.duplicated()]  # Avoid duplicate conc
                        fitted_csv_buffer = io.StringIO()
                        combined_fitted.to_csv(fitted_csv_buffer, index=False)
                        st.download_button(
                            label="Download Fitted Data Points (CSV)",
                            data=fitted_csv_buffer.getvalue(),
                            file_name="fitted_data_points.csv",
                            mime="text/csv"
                        )
                    
                    st.markdown("---")
                    st.caption("For research and educational use only. Not validated for clinical decisions. "
                               "Always verify fits with your preferred software/statistics.")
                    with st.expander("How this works"):
                        st.markdown("""
                        - Fits a 4-parameter Hill equation to each selected response.
                        - Overlays multiple fits on one plot if selected.
                        - Uses log-scale x-axis (standard for dose-response).
                        - IC50 is the concentration giving 50% of the span between top and bottom.
                        - Requires ≥4 data points per response for reliable fitting.
                        - Best results when response is normalized (0–100 scale).
                        - Units are optional and for display only.
                        """)
            
            else:
                st.info("Select at least one response column to proceed.")
        
        else:
            st.warning("No response columns found. The CSV needs at least one column besides 'Concentration'.")
    
    else:
        st.warning("CSV must contain a column named exactly 'Concentration' (case-sensitive).")
else:
    st.info("Please upload a CSV file to begin.")