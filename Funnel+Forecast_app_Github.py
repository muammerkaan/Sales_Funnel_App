import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error, r2_score
# streamlit run Funnel+Forecast_app_Github.py


class TimeBasedTopDownFunnel:

    def __init__(self, df):
        self.original_df = df.copy()
        self.current_df = df.copy()  # This will hold the current data being analyzed
        self._recompute()

    def _recompute(self):
        counts_df = self._get_counts(self.current_df)
        self.adjusted_df = self._add_conversion_rates(counts_df)
        self._calculate_conversions()

    def _get_counts(self, df):
        # Count the unique IDs for each stage
        counts = []
        for i in range(1, df.shape[1], 2):  # Assuming columns are in the format: stage_date, stage_id, ...
            stage_name = df.columns[i]
            unique_ids = df[stage_name].nunique()
            counts.append([stage_name, unique_ids])
        counts_df = pd.DataFrame(counts, columns=['Stages', 'Counts'])
        return counts_df

    def _add_conversion_rates(self, df):
        new_rows = []
        for i in range(len(df)-1):
            new_rows.append([f"{df.iloc[i, 0]}_to_{df.iloc[i+1, 0]}", 0])
        expanded_df = pd.DataFrame(new_rows, columns=df.columns)
        final_df = pd.concat([df, expanded_df]).sort_index(kind='merge').reset_index(drop=True)
        return final_df

    def _calculate_conversions(self):
        for i in range(1, len(self.adjusted_df) - 1, 2):
            if self.adjusted_df.iloc[i-1, 1] != 0:  # Check to prevent division by zero
                self.adjusted_df.iloc[i, 1] = self.adjusted_df.iloc[i+1, 1] / self.adjusted_df.iloc[i-1, 1]
            else:
                self.adjusted_df.iloc[i, 1] = 0  # Set conversion rate to 0 if denominator is zero
                
    def set_exchange_goal(self, goal):
        self.adjusted_df.iloc[-1, 1] = goal
        for i in range(len(self.adjusted_df) - 3, -1, -2):
            self.adjusted_df.iloc[i, 1] = self.adjusted_df.iloc[i+2, 1] / self.adjusted_df.iloc[i+1, 1]
        self._calculate_conversions()
                
    def adjust_conversion_rate(self, stage_conversion, rate):
        index = self.adjusted_df[self.adjusted_df['Stages'] == stage_conversion].index[0]
        self.adjusted_df.iloc[index, 1] = rate

        for i in range(index-1, -1, -2):
            self.adjusted_df.iloc[i, 1] = self.adjusted_df.iloc[i+2, 1] / self.adjusted_df.iloc[i+1, 1]
        self._calculate_conversions()

    def set_time_interval(self, start_date, end_date):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter the original dataframe by the specified date range for the first date column
        first_date_column = self.original_df.columns[0]
        self.current_df = self.original_df[self.original_df[first_date_column].between(start_date, end_date)]

        # Recompute counts and conversions for the filtered data
        self._recompute()
    
    def display(self):
            print("\nAdjusted Data After Adjustments:")
            print(self.adjusted_df)
        
    def plot_funnel(self):
        # Extract stages and counts from the adjusted dataframe
        stages = self.adjusted_df['Stages'].tolist()
        counts = self.adjusted_df['Counts'].tolist()
    
        # Create hover text with conversion rate and counts
        hover_text = []
        for i in range(0, len(stages), 2):  # Only consider the actual stages, not the conversion rates
            conversion_rate = 0 if i == len(stages) - 1 else self.adjusted_df.iloc[i+1, 1]
            hover_text.append(f"Counts: {counts[i]}<br>Conversion Rate: {conversion_rate:.2%}")
    
        # Create funnel chart
        fig = go.Figure(go.Funnel(
            y=stages[::2],  # Only consider the actual stages, not the conversion rates
            x=counts[::2],  # Only consider the actual stages, not the conversion rates
            textinfo="value+percent previous",
            hovertext=hover_text,
            #marker={"color": "royalblue"},
            connector={"line": {"color": "royalblue", "width": 3}}
        ))
        return fig

    def plot_conversion_time_boxplot(self, stage1, stage2, exclude_outliers=False):
        # Check if the stages exist in the dataframe
        if stage1 not in self.current_df.columns or stage2 not in self.current_df.columns:
            print(f"One or both of the stages '{stage1}' and '{stage2}' do not exist in the dataframe.")
            return

        # Calculate the time difference between the two stages for each unique ID
        self.current_df['Conversion Time'] = (self.current_df[stage2] - self.current_df[stage1]).dt.days

        df_filtered = self.current_df.copy()

        # If exclude_outliers is True, filter out outliers using IQR
        if exclude_outliers:
            Q1 = df_filtered['Conversion Time'].quantile(0.25)
            Q3 = df_filtered['Conversion Time'].quantile(0.75)
            IQR = Q3 - Q1
            filter = (df_filtered['Conversion Time'] >= Q1 - 1.5 * IQR) & (df_filtered['Conversion Time'] <= Q3 + 1.5 * IQR)
            df_filtered = df_filtered[filter]

        # Plot the boxplot using Plotly Express
        fig = px.box(df_filtered, y='Conversion Time', title=f'Conversion Time from {stage1} to {stage2}', 
                    labels={'Conversion Time': 'Days'})
        return fig

    def display_avg_employee(self, num_employees, stage1, stage2): 
        # Check if the stages exist in the dataframe
        if stage1 not in self.adjusted_df['Stages'].values or stage2 not in self.adjusted_df['Stages'].values:
            print(f"One or both of the stages '{stage1}' and '{stage2}' do not exist in the dataframe.")
            return

        # Filter rows between the stages (inclusive)
        start_idx = self.adjusted_df[self.adjusted_df['Stages'] == stage1].index[0]
        end_idx = self.adjusted_df[self.adjusted_df['Stages'] == stage2].index[0]
        self._avg_df = self.adjusted_df.loc[start_idx:end_idx].copy()  # Store as a temporary attribute

        # Adjust the 'Counts' column for the average employee
        self._avg_df['Counts'] = self._avg_df['Counts'] / num_employees

        print("\nAdjusted Data for Average Employee:")
        print(self._avg_df)
        return self._avg_df

    def plot_avg_funnel(self):
        # Extract stages and counts from the _avg_df
        stages = self._avg_df['Stages'].tolist()
        counts = self._avg_df['Counts'].tolist()

        # Create hover text with conversion rate and counts
        hover_text = []
        for i in range(0, len(stages), 2):  # Only consider the actual stages, not the conversion rates
            conversion_rate = 0 if i == len(stages) - 1 else self._avg_df.iloc[i+1, 1]
            hover_text.append(f"Counts: {counts[i]}<br>Conversion Rate: {conversion_rate:.2%}")

        # Create funnel chart for _avg_df
        fig = go.Figure(go.Funnel(
            y=stages[::2],  # Only consider the actual stages, not the conversion rates
            x=counts[::2],  # Only consider the actual stages, not the conversion rates
            textinfo="value+percent previous",
            hovertext=hover_text,
            connector={"line": {"color": "royalblue", "width": 3}}
        ))
        return fig

class Forecast:
    def __init__(self, df, stage_1, stage_2):
        self.df = df
        self.stage_1_date_col = f"{stage_1}_Date"
        self.stage_1_id_col = f"{stage_1}_ID"
        self.stage_2_date_col = f"{stage_2}_Date"
        self.stage_2_id_col = f"{stage_2}_ID"
        self.forecast_df = None

    def forecast_dataframe(self, forecast_days=90, bin_size=10, stage_1_beginning_date=None, 
                           median_daily_stage_1_frequency=None, median_daily_stage_1_rate_of_change=None):
        df = self.df
        
        if not isinstance(stage_1_beginning_date, pd.Timestamp):
            stage_1_beginning_date = pd.to_datetime(stage_1_beginning_date)
        
        # Calculate median daily stage 1 frequency
        if not median_daily_stage_1_frequency:
            filtered_df = df[(df[self.stage_1_date_col] >= stage_1_beginning_date)]
            filtered_df = filtered_df.drop_duplicates(subset=[self.stage_1_id_col])
            daily_stage_1 = filtered_df.groupby(filtered_df[self.stage_1_date_col].dt.date).size()
            median_daily_stage_1_frequency = daily_stage_1.median()

        # Calculate median daily rate of change in the stage 1 frequency by default
        if median_daily_stage_1_rate_of_change is None:
            daily_stage_1_diff = daily_stage_1.diff().dropna()
            median_daily_stage_1_rate_of_change = daily_stage_1_diff.median()

        # If stage_1_beginning_date is not provided, set it to the earliest date in the stage_1_date_col
        if not stage_1_beginning_date:
            stage_1_beginning_date = df[self.stage_1_date_col].min()

        df1 = df[df[self.stage_2_id_col].isnull() | ~df.duplicated(self.stage_2_id_col, keep='first')]
        df1 = df1[df1[self.stage_1_date_col] >= stage_1_beginning_date]

        start_date = df[self.stage_1_date_col].max()
        if not isinstance(start_date, datetime.datetime):
            start_date = datetime.datetime.strptime(str(start_date), '%Y-%m-%d')
        current_date = start_date

        df1['Age'] = np.where(df1[self.stage_2_date_col].isnull(),
                            (current_date - df1[self.stage_1_date_col]).dt.days,
                            (df1[self.stage_2_date_col] - df1[self.stage_1_date_col]).dt.days)

        df2 = df1.groupby('Age').agg(
            stage_2_count=(self.stage_2_id_col, 'count'),
            not_stage_2_count=(self.stage_2_id_col, lambda x: x.isnull().sum())
        ).reset_index()

        bins = range(0, int(df2['Age'].max()) + bin_size, bin_size)
        df2['Age_Bin'] = pd.cut(df2['Age'], bins, right=True)
        df2['Age'] = df2['Age_Bin'].apply(lambda x: x.right if pd.notna(x.right) else np.nan).astype('Int64')
        df3 = df2.groupby('Age').agg({
            'stage_2_count': 'sum',
            'not_stage_2_count': 'sum'
        }).reset_index()

        df3['cumulative_not_stage_2'] = df3['not_stage_2_count'][::-1].cumsum()[::-1]
        df3['conversion_rate'] = df3['stage_2_count'] / (df3['cumulative_not_stage_2'] + df3['stage_2_count'])

        bins_count = len(df3)
        leads_matrix = np.zeros(bins_count)
        leads_matrix += df3['not_stage_2_count'].values
        conversion_rates = df3['conversion_rate'].values

        results = []

        for _ in range(forecast_days // bin_size):
            leads_matrix[1:] = leads_matrix[:-1]
            
            # Adjust the number of stage 1 values added based on the median daily rate of change
            leads_matrix[0] = (median_daily_stage_1_frequency + median_daily_stage_1_rate_of_change) * bin_size
            
            conversions = leads_matrix * conversion_rates
            leads_matrix -= conversions
            results.append(conversions.sum())

        forecast_dates = pd.date_range(start=current_date, periods=len(results) + 1, freq=f"{bin_size}D")[1:]
        self.forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Conversions': results
        })
        self.forecast_df['Cumulative_Conversions'] = self.forecast_df['Conversions'].cumsum()
        return self.forecast_df

    def plot_cumulative_conversions(self, cumulative):
            if not hasattr(self, 'forecast_df'):
                print("Please run the forecast_dataframe method first.")
                return
            
            fig = go.Figure()
            
            # Decide which columns to plot based on the cumulative flag
            y_col = 'Cumulative_Conversions' if cumulative else 'Conversions'
            title = 'Forecasted Cumulative Conversions' if cumulative else 'Forecasted Conversions'
            
            fig.add_trace(go.Scatter(
                x=self.forecast_df['Date'],
                y=self.forecast_df[y_col],
                mode='lines+markers',
                name=y_col,
                hovertemplate=
                '<b>Date</b>: %{x}<br>' +
                '<b>Conversions</b>: %{text}<br>' +
                '<b>' + y_col + '</b>: %{y}',
                text=self.forecast_df['Conversions'].values
            ))
            
            fig.update_layout(title=title,
                            xaxis_title='Date',
                            yaxis_title=y_col)
            
            return(fig)
        
    def forecast_test(self, forecast_start_date, bin_size=10, stage_1_beginning_date=None, median_daily_stage_1_frequency=None):
        df = self.df
        
        # Convert forecast_start_date to datetime object
        forecast_start_date = pd.to_datetime(forecast_start_date)
        
        # Store the maximum date in the dataframe
        max_date = df[self.stage_1_date_col].max()
        
        # Calculate the forecast range
        forecast_days = (max_date - forecast_start_date).days
        
        # Cut the data off at forecast_start_date
        df = df[df[self.stage_1_date_col] < forecast_start_date]
        
        if not isinstance(stage_1_beginning_date, pd.Timestamp):
            stage_1_beginning_date = pd.to_datetime(stage_1_beginning_date)
        
        # Calculate median daily stage 1 frequency
        if not median_daily_stage_1_frequency:
            filtered_df = df[(df[self.stage_1_date_col] >= stage_1_beginning_date)]
            filtered_df = filtered_df.drop_duplicates(subset=[self.stage_1_id_col])
            daily_stage_1 = filtered_df.groupby(filtered_df[self.stage_1_date_col].dt.date).size()
            median_daily_stage_1_frequency = daily_stage_1.median()
        
        # If stage_1_beginning_date is not provided, set it to the earliest date in the stage_1_date_col
        if not stage_1_beginning_date:
            stage_1_beginning_date = df[self.stage_1_date_col].min()

        df1 = df[df[self.stage_2_id_col].isnull() | ~df.duplicated(self.stage_2_id_col, keep='first')]
        df1 = df1[df1[self.stage_1_date_col] >= stage_1_beginning_date]

        current_date = forecast_start_date

        df1['Age'] = np.where(df1[self.stage_2_date_col].isnull(),
                            (current_date - df1[self.stage_1_date_col]).dt.days,
                            (df1[self.stage_2_date_col] - df1[self.stage_1_date_col]).dt.days)

        df2 = df1.groupby('Age').agg(
            stage_2_count=(self.stage_2_id_col, 'count'),
            not_stage_2_count=(self.stage_2_id_col, lambda x: x.isnull().sum())
        ).reset_index()

        bins = range(0, int(df2['Age'].max()) + bin_size, bin_size)
        df2['Age_Bin'] = pd.cut(df2['Age'], bins, right=True)
        df2['Age'] = df2['Age_Bin'].apply(lambda x: x.right if pd.notna(x.right) else np.nan).astype('Int64')
        df3 = df2.groupby('Age').agg({
            'stage_2_count': 'sum',
            'not_stage_2_count': 'sum'
        }).reset_index()

        df3['cumulative_not_stage_2'] = df3['not_stage_2_count'][::-1].cumsum()[::-1]
        df3['conversion_rate'] = df3['stage_2_count'] / (df3['cumulative_not_stage_2'] + df3['stage_2_count'])

        bins_count = len(df3)
        leads_matrix = np.zeros(bins_count)
        leads_matrix += df3['not_stage_2_count'].values
        conversion_rates = df3['conversion_rate'].values

        results = []

        for _ in range(forecast_days // bin_size):
            leads_matrix[1:] = leads_matrix[:-1]
            leads_matrix[0] = median_daily_stage_1_frequency * bin_size
            conversions = leads_matrix * conversion_rates
            leads_matrix -= conversions
            results.append(conversions.sum())

        forecast_dates = pd.date_range(start=current_date, periods=len(results) + 1, freq=f"{bin_size}D")[1:]
        self.forecast_test_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted_Conversions': results
        })
        self.forecast_test_df['Forecasted_Cumulative_Conversions'] = self.forecast_test_df['Forecasted_Conversions'].cumsum()
        
        # Create date bins for the forecasted period
        forecast_date_bins = pd.date_range(start=forecast_start_date, periods=len(results) + 1, freq=f"{bin_size}D")

        # Categorize each stage 2 date in the original dataframe into these bins
        self.df['Date_Bin'] = pd.cut(self.df[self.stage_2_date_col], bins=forecast_date_bins, right=False)

        # Group by these bins and count the unique stage 2 IDs for actual conversions
        actual_counts_df = self.df.groupby('Date_Bin')[self.stage_2_id_col].nunique().reset_index(name='Actual_Conversions')

        # Merge this count with the forecast dataframe
        self.forecast_test_df['Date_Bin'] = pd.cut(self.forecast_test_df['Date'], bins=forecast_date_bins, right=False)
        self.forecast_test_df = self.forecast_test_df.merge(actual_counts_df, on='Date_Bin', how='left').drop(columns='Date_Bin')
        self.forecast_test_df['Actual_Conversions'] = self.forecast_test_df['Actual_Conversions'].fillna(0).astype(int)
        self.forecast_test_df['Actual_Cumulative_Conversions'] = self.forecast_test_df['Actual_Conversions'].cumsum()
        return self.forecast_test_df

    def plot_forecast_vs_actual(self, cumulative_test=False):
        # Ensure the forecast_test method has been called and the dataframe exists
        if not hasattr(self, 'forecast_test_df'):
            raise ValueError("Please run the forecast_test method first to generate the forecast data.")

        # Create a time series plot
        fig = go.Figure()

        # Decide which columns to plot based on the cumulative flag
        forecast_col = 'Forecasted_Cumulative_Conversions' if cumulative_test else 'Forecasted_Conversions'
        actual_col = 'Actual_Cumulative_Conversions' if cumulative_test else 'Actual_Conversions'

        # Add Forecasted_Conversions or Cumulative Conversions to the plot
        fig.add_trace(go.Scatter(x=self.forecast_test_df['Date'], 
                                 y=self.forecast_test_df[forecast_col],
                                 mode='lines+markers',
                                 name=forecast_col))

        # Add Actual Conversions or Cumulative Actual Conversions to the plot
        fig.add_trace(go.Scatter(x=self.forecast_test_df['Date'], 
                                 y=self.forecast_test_df[actual_col],
                                 mode='lines+markers',
                                 name=actual_col))

        # Update layout for better visualization
        title = 'Cumulative Forecasted vs Cumulative Actual Conversions' if cumulative_test else 'Forecasted vs Actual Conversions'
        fig.update_layout(title=title,
                          xaxis_title='Date',
                          yaxis_title='Conversions',
                          legend_title='Legend')

        # Show the plot
        return fig
    
    def compute_evaluation_metrics(self, cumulative_test=True):
        if not hasattr(self, 'forecast_df'):
            print("Please run the forecast_dataframe method first.")
            return
        
        # Decide which columns to use based on the cumulative flag
        actual_col = 'Actual_Cumulative_Conversions' if cumulative_test else 'Actual_Conversions'
        forecasted_col = 'Forecasted_Cumulative_Conversions' if cumulative_test else 'Forecasted_Conversions'
        
        # Extract actual and forecasted values
        actual_values = self.forecast_test_df[actual_col].values
        forecasted_values = self.forecast_test_df[forecasted_col].values
        
        # Compute R-squared
        r2 = r2_score(actual_values, forecasted_values)
        
        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))
        
        # Compute MAE
        mae = np.mean(np.abs(actual_values - forecasted_values))
        
        # Compute MAPE
        mape = np.mean(np.abs((actual_values - forecasted_values) / actual_values)) * 100
        
        # Print the metrics
        metrics_df = pd.DataFrame({
        'Metric': ['R-squared', 'RMSE', 'MAE', 'MAPE (%)'],
        'Value': [r2, rmse, mae, mape]
        })
        
        return metrics_df


# Run the Streamlit application
if __name__ == "__main__":
    st.title("Time-Based Top-Down Funnel + Connversion-Based Forecasting Analysis")

def process_dataframe(df):
    for i in range(0, len(df.columns), 2):
        df[df.columns[i]] = pd.to_datetime(df[df.columns[i]])
    return df

# Load the data
#@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(file)
    return process_dataframe(df)

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file in a format which includes:" +
    " Two columns per funnel stage with the first column being the stagename_Date and the second column being the stagename_ID" + 
    " and the stages being in chronological order." +
    "Here is a link to the Github repository where you can find the source code and an example data set: " +
    "https://github.com/muammerkaan/Sales_Funnel_App",
    type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    # Create an instance of the TimeBasedTopDownFunnel class
    funnel = TimeBasedTopDownFunnel(df)

    # Create a sidebar for settings
    st.sidebar.title("Funnel Settings")

    # Set the date range
    start_date, end_date = st.sidebar.date_input("Select Date Range", [df.iloc[0, 0], df.iloc[-1, 0]])
    funnel.set_time_interval(start_date, end_date)

    # Set the goal for the last stage
    goal = st.sidebar.number_input("Set Goal for Last Stage", value=int(funnel.adjusted_df.iloc[-1, 1]), step=1)
    funnel.set_exchange_goal(goal)

    # Adjust conversion rates with sliders
    st.sidebar.header("Adjust Conversion Rates")
    for i in range(1, len(funnel.adjusted_df) - 1, 2):
        rate = st.sidebar.slider(f"Rate for {funnel.adjusted_df.iloc[i, 0]}", 0.0, 1.0, funnel.adjusted_df.iloc[i, 1])
        funnel.adjust_conversion_rate(funnel.adjusted_df.iloc[i, 0], rate)

    # Display the funnel plot
    st.header("Conversion Funnel")
    fig = funnel.plot_funnel()
    fig.update_layout(width=900, height=600)
    st.plotly_chart(fig)
    
    conversion_funnel_description = """
    This funnel chart illustrates the customer journey across various stages of the sales process. 
    Adjust the funnel parameters using the 'Funnel Settings' sidebar to the left. 
    Setting a new goal recalibrates stage counts based on existing conversion rates. 
    Alter these rates to dynamically adjust preceding stage counts, ensuring alignment with any modified stage.
    """
    st.write(conversion_funnel_description)

    # Plot conversion time boxplot for selected stages
    st.header("Conversion Funnel Duration Boxplot")
    stage1 = st.selectbox("Select Start Stage", list(funnel.current_df.columns[0::2]))
    stage2 = st.selectbox("Select End Stage", list(funnel.current_df.columns[0::2]), index=1)
    exclude_outliers = st.checkbox("Exclude outliers?")
    fig_box = funnel.plot_conversion_time_boxplot(stage1, stage2, exclude_outliers)
    fig_box.update_layout(width=900, height=600)
    st.plotly_chart(fig_box)
    
    boxplot_description = """
    Explore the time span between funnel stages with this boxplot. 
    Select the starting and ending stages of interest using the provided dropdown menus above and choose to include or exclude outliers. 
    This visualization helps identify time-related trends and deviations in the conversion process.
    """
    st.write(boxplot_description)

    # Plot the funnel for the average employee
    st.header("Conversion Funnel for Average Employee")
    num_employees = st.number_input("Number of Employees", value=1, step=1)
    stage1 = st.selectbox("Select Start Stage", list(funnel.current_df.columns[1::2]))
    stage2 = st.selectbox("Select End Stage", list(funnel.current_df.columns[1::2]), index=1)
    funnel.display_avg_employee(num_employees, stage1, stage2)
    fig_avg = funnel.plot_avg_funnel()
    fig_avg.update_layout(width=900, height=600)
    st.plotly_chart(fig_avg)

    employee_description = """
    This section visualizes the average performance of employees through selected stages of the conversion funnel. 
    Select the starting and ending stages of interest using the provided dropdown menus above and input the number of employees that are relevant to those stages.
    This plot is instrumental in understanding typical employee efficiency and identifying stages needing improvement.
    """
    st.write(employee_description)


    # Create a sidebar for forecast settings
    st.sidebar.title("Forecast Settings")
    
    # Dropdowns for users to select the stages
    available_stages = [col.replace("_Date", "") for col in df.columns if "_Date" in col]
    stage_1 = st.sidebar.selectbox("Select Stage 1", available_stages, index=0)
    stage_2 = st.sidebar.selectbox("Select Stage 2", available_stages, index=1)

    # Create an instance of the Forecast class with the selected stages
    forecast = Forecast(df, stage_1, stage_2)

    # User input for forecast settings
    forecast_days = st.sidebar.number_input("Forecast Days", min_value=1, value=90, step=1)
    bin_size = st.sidebar.number_input("Bin Size", min_value=1, value=10, step=1)
    # Assuming df is your DataFrame and forecast.stage_1_date_col is the column with dates

    # Get the minimum and maximum dates from the DataFrame
    min_date = df[forecast.stage_1_date_col].min().date()
    max_date = df[forecast.stage_1_date_col].max().date()

    # Create a date slider
    selected_date = st.sidebar.slider("Stage 1 Date", min_date, max_date)
    stage_1_beginning_date = pd.Timestamp(selected_date)
  
    # Calculate median_daily_stage_1_frequency directly in the Streamlit app
    filtered_df = df[(df[forecast.stage_1_date_col] >= pd.Timestamp(stage_1_beginning_date))]
    filtered_df = filtered_df.drop_duplicates(subset=[forecast.stage_1_id_col])
    daily_stage_1 = filtered_df.groupby(filtered_df[forecast.stage_1_date_col].dt.date).size()
    default_median_daily_stage_1_frequency = daily_stage_1.median()

    # Add a number input for median_daily_stage_1_frequency
    median_daily_stage_1_frequency = st.sidebar.number_input("Median Daily Stage 1 Frequency", 
                                                             value=float(default_median_daily_stage_1_frequency),
                                                             min_value=0.0,
                                                             step=0.1 )

    # Calculate median_daily_stage_1_rate_of_change directly in the Streamlit app
    daily_stage_1_diff = daily_stage_1.diff().dropna()
    default_median_daily_stage_1_rate_of_change = daily_stage_1_diff.median()

    # Add a number input for median_daily_stage_1_rate_of_change
    median_daily_stage_1_rate_of_change = st.sidebar.number_input("Median Daily Stage 1 Rate of Change", 
                                                                value=float(default_median_daily_stage_1_rate_of_change),
                                                                min_value=-10000.0,
                                                                max_value=10000.0,
                                                                step=0.1 )

    # Call the forecast_dataframe method with user input
    forecast_df = forecast.forecast_dataframe(forecast_days=forecast_days, 
                                            bin_size=bin_size, 
                                            stage_1_beginning_date=stage_1_beginning_date,
                                            median_daily_stage_1_frequency=median_daily_stage_1_frequency,
                                            median_daily_stage_1_rate_of_change=median_daily_stage_1_rate_of_change)

    # Display the forecast results
    st.header("Forecasted Conversions")

    cumulative = st.checkbox("Cumulative?")
    fig_forecast = forecast.plot_cumulative_conversions(cumulative)
    fig_forecast.update_layout(width=900, height=600)
    st.plotly_chart(fig_forecast)
    
    forecast_description = """
    This chart visualizes the projected number of conversions between two stages over a specified forecast period. 
    The forecast is calculated by taking historical conversion rates within the data and simulating them against an adjustable the number of first stage values entering the funnel.
    The 'Forecast Settings' in the sidebar allow you to set the two stages from which the conversion rates are obtained, the latter stage being the forecasted stage. 
    You can also adjust the number of days ahead to forecast, the size of the age bins used to obtain the conversion rates, 
    the stage 1 start date from which the conversion rates are calculated from, 
    the median daily stage frequencies which are generated inorder to simulate the forecast and the rate of change that the daily stage frequency should under go. 
    This chart offers the flexibility to view either cumulative or individual forecasted conversions, allowing for a comprehensive analysis of future trends.
    """
    st.write(forecast_description)

    forecast_start_date = st.sidebar.date_input("Forecast Start Date", df[forecast.stage_1_date_col].max().replace(month=1, day=1))

    # Call the forecast_dataframe method with user input
    forecast_test_df = forecast.forecast_test(forecast_start_date=forecast_start_date, 
                                              bin_size=bin_size, 
                                              stage_1_beginning_date=stage_1_beginning_date)
    
    # Display the forecast test results
    st.header("Forecasted Conversions Test")
    cumulative_test = st.checkbox("Cumulative Test?")
    fig_forecast_test = forecast.plot_forecast_vs_actual(cumulative_test)
    fig_forecast_test.update_layout(width=900, height=600)
    st.plotly_chart(fig_forecast_test)
    
    forecast_test_description = """
    This plot offers a side-by-side comparison of forecasted conversions with actual data, starting from a user-selected date. 
    The Forecasted_Conversions are obtained using the same settings as the above forecast,
    however they are calculated on the dates and first stage counts that occur between the adjustable Forecast Start Date value and the final date your data set.
    Additionally you can choose between a cumulative or non-cumulative view. This visualisation helps understand how well the future forecast performs.
    """
    st.write(forecast_test_description)

    
    # Compute evaluation metrics
    st.header("Evaluation Metrics")
    metrics = forecast.compute_evaluation_metrics(cumulative_test=cumulative_test)
    st.table(metrics)
    
    evaluation_metrics_description = """
    Assess the effectiveness of the forecasting model with key metrics like R-squared, RMSE, MAE, and MAPE. 
    These metrics are calculated based on the above 'Forecasted Conversions Test' are useful for understanding the model's predictive accuracy. 
    The table adapts based on whether the cumulative or individual forecast comparison is selected.
    R-squared indicates the variance in actual data explained by the model, with higher values being better. 
    RMSE and MAE measure average error magnitudes, with lower values indicating a better fit. 
    MAPE expresses accuracy as a percentage, where lower values denote higher accuracy. 

    """
    st.write(evaluation_metrics_description)

