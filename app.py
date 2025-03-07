import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Tuple, Union, Dict
from openai import OpenAI

def is_id_or_reference_column(col: str) -> bool:
    """
    Check if a column name contains ID or reference patterns.
    """
    # Convert column name to lowercase for case-insensitive matching
    col_lower = col.lower()
    
    # Check for common ID/reference patterns
    id_patterns = ['id', 'ref', 'reference', 'key', 'uuid', 'guid', 'hash', 'slug', 'token']
    
    # Check if any pattern is in the column name
    return any(pattern in col_lower for pattern in id_patterns)

def is_meaningful_numeric_column(df: pd.DataFrame, col: str) -> bool:
    """
    Determine if a numeric column is meaningful for analysis.
    Excludes ID columns, timestamps, and other non-meaningful numeric columns.
    """
    # Convert column name to lowercase for case-insensitive matching
    col_lower = col.lower()
    
    # Always include salary-related columns
    if any(term in col_lower for term in ['salary', 'compensation', 'pay', 'wage', 'income', 'earnings']):
        return True
    
    # Comprehensive list of ID patterns to exclude
    id_patterns = [
        # General ID Naming
        'id', 'uuid', 'guid', 'key', 'reference', 'hash', 'slug', 'token',
        
        # Entity-Specific Naming
        # User-Related
        'user_id', 'account_id', 'profile_id', 'member_id', 'customer_id', 'client_id', 'subscriber_id',
        
        # Product & Inventory
        'product_id', 'item_id', 'sku', 'barcode', 'inventory_id', 'asset_id', 'variant_id', 'batch_id', 'stock_id',
        
        # Orders & Transactions
        'order_id', 'transaction_id', 'invoice_id', 'payment_id', 'purchase_id', 'cart_id', 'checkout_id', 'receipt_id',
        
        # Content & Media
        'post_id', 'article_id', 'blog_id', 'comment_id', 'review_id', 'image_id', 'video_id', 'file_id', 
        'attachment_id', 'document_id', 'folder_id',
        
        # Location & Address
        'address_id', 'location_id', 'region_id', 'city_id', 'country_id', 'zone_id', 'district_id', 'state_id', 
        'geo_id', 'coordinates_id',
        
        # Business & Organization
        'company_id', 'business_id', 'department_id', 'team_id', 'office_id', 'branch_id', 'division_id',
        
        # Project & Task Management
        'task_id', 'project_id', 'milestone_id', 'goal_id', 'ticket_id', 'issue_id', 'feature_id', 'bug_id', 'story_id',
        
        # Education & Learning
        'student_id', 'teacher_id', 'course_id', 'class_id', 'lesson_id', 'assignment_id', 'exam_id', 'grade_id', 
        'certificate_id',
        
        # Finance & Accounting
        'budget_id', 'expense_id', 'payroll_id', 'salary_id', 'tax_id', 'ledger_id', 'accounting_id', 'fund_id',
        
        # Security & Access Control
        'session_id', 'token_id', 'access_id', 'permission_id', 'role_id', 'group_id', 'api_key', 'auth_id', 'key_id',
        
        # Healthcare & Medical
        'patient_id', 'doctor_id', 'appointment_id', 'record_id', 'prescription_id', 'medication_id', 'insurance_id',
        
        # Logistics & Transportation
        'shipment_id', 'tracking_id', 'route_id', 'trip_id', 'vehicle_id', 'driver_id', 'log_id',
        
        # IoT & Devices
        'device_id', 'sensor_id', 'gateway_id', 'hardware_id', 'firmware_id', 'mac_address', 'ip_address'
    ]
    
    # Check if column name matches any ID patterns
    if any(pattern in col_lower for pattern in id_patterns):
        return False
    
    # Check if column is a timestamp
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return False
    
    # Check if column has too many unique values (likely an ID)
    if df[col].nunique() > len(df) * 0.8:  # If more than 80% of values are unique
        return False
    
    return True

def is_numeric_column(df: pd.DataFrame, col: str) -> bool:
    """
    Determine if a column contains numeric values, even if stored as strings.
    Handles currency and other formatted numbers.
    """
    # If already numeric (including int64), return True
    if pd.api.types.is_numeric_dtype(df[col]):
        return True
    
    # If string, try to convert to numeric
    if pd.api.types.is_string_dtype(df[col]):
        try:
            # Remove currency symbols and commas, then try to convert
            test_values = df[col].astype(str).str.replace('$', '').str.replace(',', '')
            pd.to_numeric(test_values, errors='raise')
            return True
        except:
            return False
    
    return False

def analyze_dataframe(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """
    Analyze the dataframe and suggest relevant pivot tables.
    Returns a list of tuples containing (description, pivot_table)
    """
    suggestions = []
    
    # Get all columns except ID/reference columns
    all_cols = [col for col in df.columns if not is_id_or_reference_column(col)]
    st.write(f"Columns available for analysis: {all_cols}")
    
    # Get numeric and categorical columns
    numeric_cols = [col for col in all_cols if is_numeric_column(df, col)]
    categorical_cols = [col for col in all_cols if not is_numeric_column(df, col) and not pd.api.types.is_datetime64_any_dtype(df[col])]
    
    st.write(f"Numeric columns found: {numeric_cols}")
    st.write(f"Categorical columns found: {categorical_cols}")
    
    if not numeric_cols:
        st.warning("No numeric columns found in the dataset. Please ensure your CSV contains numeric data.")
        return suggestions
    
    if not categorical_cols:
        st.warning("No categorical columns found in the dataset. Please ensure your CSV contains categorical data.")
        return suggestions
    
    # Convert string numeric columns to actual numeric values
    for col in numeric_cols:
        if pd.api.types.is_string_dtype(df[col]):
            # Handle percentage values
            if any(term in col.lower() for term in ['rate', 'percentage', '%', 'percent']):
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').astype(float) / 100
            else:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    
    # Define common aggregation functions
    agg_funcs = ['mean', 'min', 'max', 'count']
    
    # Define named functions for quartiles
    def q1(x): return x.quantile(0.25)
    def q3(x): return x.quantile(0.75)
    
    agg_funcs_with_quartiles = ['min', q1, 'median', q3, 'max', 'mean', 'std', 'count']
    
    def is_meaningful_pivot_table(pivot_table: pd.DataFrame) -> bool:
        """
        Check if a pivot table provides meaningful insights.
        Returns False if all counts are 1 or if the table is empty.
        """
        if pivot_table.empty:
            return False
            
        # If the table has a count column, check if all counts are 1
        if isinstance(pivot_table.columns, pd.MultiIndex):
            count_cols = [col for col in pivot_table.columns if 'count' in str(col).lower()]
            if count_cols:
                for count_col in count_cols:
                    # Get the count values as a numpy array
                    count_values = pivot_table[count_col].to_numpy()
                    # Check if any value is greater than 1
                    if not np.any(count_values > 1):
                        return False
        else:
            if 'count' in pivot_table.columns:
                # Get the count values as a numpy array
                count_values = pivot_table['count'].to_numpy()
                # Check if any value is greater than 1
                if not np.any(count_values > 1):
                    return False
        
        return True
    
    def create_pivot_table(df: pd.DataFrame, values: str, index: Union[str, List[str]], aggfunc: Union[List[str], Dict[str, Union[str, callable]]]) -> pd.DataFrame:
        """
        Create a pivot table with proper handling of MultiIndex columns and duplicate column names.
        """
        try:
            # Create the pivot table
            pivot_table = pd.pivot_table(
                df,
                values=values,
                index=index,
                aggfunc=aggfunc,
                fill_value=0  # Fill NaN values with 0
            )
            
            # If the columns are MultiIndex, rename them to avoid duplicates
            if isinstance(pivot_table.columns, pd.MultiIndex):
                new_columns = []
                for col in pivot_table.columns:
                    if isinstance(col, tuple):
                        # For MultiIndex columns, combine the levels with an underscore
                        new_col = f"{col[0]}_{col[1]}"
                    else:
                        new_col = str(col)
                    new_columns.append(new_col)
                pivot_table.columns = new_columns
            
            # Rename columns to be more descriptive
            column_mapping = {}
            for col in pivot_table.columns:
                # Handle both direct column names and names with underscores
                base_name = col.split('_')[0] if '_' in col else col
                if base_name == 'min':
                    column_mapping[col] = f'Min {values}'
                elif base_name == 'q1':
                    column_mapping[col] = f'Q1 (25th) {values}'
                elif base_name == 'median':
                    column_mapping[col] = f'Median (50th) {values}'
                elif base_name == 'q3':
                    column_mapping[col] = f'Q3 (75th) {values}'
                elif base_name == 'max':
                    column_mapping[col] = f'Max {values}'
                elif base_name == 'mean':
                    column_mapping[col] = f'Mean {values}'
                elif base_name == 'std':
                    column_mapping[col] = f'Std Dev {values}'
                elif base_name == 'count':
                    column_mapping[col] = f'Count {values}'
                else:
                    column_mapping[col] = col
            
            # Apply the column mapping
            pivot_table.columns = [column_mapping.get(col, col) for col in pivot_table.columns]
            
            return pivot_table
            
        except Exception as e:
            st.warning(f"Could not create pivot table for {values} by {index}: {str(e)}")
            return pd.DataFrame()
    
    # Create pivot tables for numeric metrics by categorical columns
    for metric in numeric_cols:
        # Single categorical column analysis
        for cat_col in categorical_cols:
            # Skip if the categorical column is time-based
            if any(term in cat_col.lower() for term in ['year', 'month', 'week', 'day', 'period', 'quarter', 'date']):
                continue
            try:
                # Basic analysis
                pivot_table = create_pivot_table(df, metric, cat_col, agg_funcs)
                if is_meaningful_pivot_table(pivot_table):
                    description = f"{metric} by {cat_col}"
                    suggestions.append((description, pivot_table))
                
                # Only create quartile analysis if:
                # 1. The categorical column has enough unique values (not too granular)
                # 2. The metric is continuous (not discrete)
                # 3. We have enough data points per category
                if (df[cat_col].nunique() <= 20 and  # Not too many categories
                    pd.api.types.is_float_dtype(df[metric]) and  # Continuous metric
                    df.groupby(cat_col)[metric].count().min() >= 5):  # At least 5 data points per category
                    
                    pivot_table = create_pivot_table(df, metric, cat_col, agg_funcs_with_quartiles)
                    if is_meaningful_pivot_table(pivot_table):
                        description = f"Detailed {metric} Analysis by {cat_col} (with quartiles)"
                        suggestions.append((description, pivot_table))
            except Exception as e:
                st.warning(f"Could not create pivot table for {metric} by {cat_col}: {str(e)}")
                continue
        
        # Two categorical columns analysis
        for i, cat_col1 in enumerate(categorical_cols):
            if any(term in cat_col1.lower() for term in ['year', 'month', 'week', 'day', 'period', 'quarter', 'date']):
                continue
            for cat_col2 in categorical_cols[i+1:]:
                if any(term in cat_col2.lower() for term in ['year', 'month', 'week', 'day', 'period', 'quarter', 'date']):
                    continue
                try:
                    # Basic analysis
                    pivot_table = create_pivot_table(df, metric, [cat_col1, cat_col2], agg_funcs)
                    if is_meaningful_pivot_table(pivot_table):
                        description = f"{metric} by {cat_col1} and {cat_col2}"
                        suggestions.append((description, pivot_table))
                    
                    # Only create quartile analysis if:
                    # 1. Both categorical columns have few enough unique values
                    # 2. The metric is continuous
                    # 3. We have enough data points per combination
                    if (df[cat_col1].nunique() <= 10 and  # Not too many categories
                        df[cat_col2].nunique() <= 10 and
                        pd.api.types.is_float_dtype(df[metric]) and
                        df.groupby([cat_col1, cat_col2])[metric].count().min() >= 5):
                        
                        pivot_table = create_pivot_table(df, metric, [cat_col1, cat_col2], agg_funcs_with_quartiles)
                        if is_meaningful_pivot_table(pivot_table):
                            description = f"Detailed {metric} Analysis by {cat_col1} and {cat_col2} (with quartiles)"
                            suggestions.append((description, pivot_table))
                except Exception as e:
                    st.warning(f"Could not create pivot table for {metric} by {cat_col1} and {cat_col2}: {str(e)}")
                    continue
    
    return suggestions

def rank_pivot_tables(suggestions: List[Tuple[str, pd.DataFrame]], df: pd.DataFrame, api_key: str) -> Tuple[List[Tuple[str, pd.DataFrame, str]], str, str]:
    """
    Use OpenAI to rank pivot tables by relevance and return the top 6 with explanations.
    Returns a tuple containing (list of (description, pivot_table, explanation), raw_response, prompt)
    """
    if not suggestions:
        return [], "", ""
    
    # Create a prompt for OpenAI
    prompt = f"""As a business intelligence expert, analyze these pivot tables for a dataset with columns: {', '.join(df.columns)}

    Your task is to identify the 6 most valuable pivot tables that would provide the strongest business insights and actionable recommendations.
    Consider:
    1. Strategic decision-making value
    2. Operational insights
    3. Performance metrics
    4. Business trends and patterns
    5. Areas requiring attention
    6. Opportunities for improvement

    For each selected table, provide a brief explanation of why it's valuable (max 2 sentences).
    IMPORTANT: Use the actual pivot table numbers (0 to {len(suggestions)-1}) as indices, not sequential numbers.
    
    Return your response as a JSON array with exactly 6 objects, each containing:
    - "index": the actual pivot table number (0 to {len(suggestions)-1})
    - "explanation": a brief explanation of why this pivot table is valuable
    
    Example format:
    [
        {{"index": 0, "explanation": "This pivot table shows..."}},
        {{"index": 1, "explanation": "This analysis reveals..."}},
        ...
    ]
    
    Pivot Tables:
    {chr(10).join(f"{i}. {desc}" for i, (desc, _) in enumerate(suggestions))}"""
    
    try:
        # Create OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Get ranking from OpenAI using the new API format
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a business intelligence expert specializing in data analysis and strategic insights. Your task is to identify the most valuable pivot tables that drive business decisions. Always use the actual pivot table numbers as indices, not sequential numbers. Return your response in JSON format."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get the raw response
        raw_response = response.choices[0].message.content
        
        # Parse the response
        try:
            # Extract JSON from the response
            import json
            # Find the first '[' and last ']' to extract the JSON array
            json_start = raw_response.find('[')
            json_end = raw_response.rfind(']') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = raw_response[json_start:json_end]
            rankings = json.loads(json_str)
            
            # Validate the response
            if not isinstance(rankings, list):
                raise ValueError("Response is not a JSON array")
            if len(rankings) != 6:
                raise ValueError(f"Expected 6 rankings, got {len(rankings)}")
            
            # Extract indices and explanations
            indices = [item["index"] for item in rankings]
            explanations = {item["index"]: item["explanation"] for item in rankings}
            
            # Validate indices
            if not all(0 <= idx < len(suggestions) for idx in indices):
                raise ValueError("Invalid index found in response")
            
            # Return the top 6 pivot tables with explanations
            return [(suggestions[i][0], suggestions[i][1], explanations[i]) 
                    for i in indices], raw_response, prompt
            
        except Exception as e:
            st.warning(f"Error parsing OpenAI response: {str(e)}")
            # If parsing fails, return the first 6 suggestions with default explanations
            return [(desc, table, "Selected as one of the top pivot tables") 
                    for desc, table in suggestions[:6]], raw_response, prompt
            
    except Exception as e:
        st.warning(f"Could not rank pivot tables using OpenAI: {str(e)}")
        # If OpenAI fails, return the first 6 suggestions with default explanations
        return [(desc, table, "Selected as one of the top pivot tables") 
                for desc, table in suggestions[:6]], "", prompt

def create_visualization(pivot_table: pd.DataFrame, description: str) -> None:
    """Create appropriate visualization based on pivot table structure"""
    try:
        # Reset index to make it available for plotting
        plot_df = pivot_table.reset_index()
        
        if isinstance(pivot_table.columns, pd.MultiIndex):
            # For pivot tables with multiple aggregation functions
            # Get the first level of the multi-index
            first_level = pivot_table.columns.get_level_values(0)
            # Get the second level (the actual metric names)
            second_level = pivot_table.columns.get_level_values(1)
            
            # Create a new dataframe with the correct column names
            new_columns = [f"{first_level[i]}_{second_level[i]}" for i in range(len(first_level))]
            plot_df.columns = list(plot_df.columns[:len(plot_df.columns)-len(new_columns)]) + new_columns
            
            # Melt the dataframe
            id_vars = list(plot_df.columns[:len(plot_df.columns)-len(new_columns)])
            plot_df = pd.melt(
                plot_df,
                id_vars=id_vars,
                var_name='Metric',
                value_name='Value'
            )
            
            # Create visualization based on number of index columns
            if len(id_vars) == 1:
                fig = px.bar(
                    plot_df,
                    x=id_vars[0],
                    y='Value',
                    color='Metric',
                    title=description,
                    barmode='group'
                )
            else:
                fig = px.bar(
                    plot_df,
                    x=id_vars[0],
                    y='Value',
                    color='Metric',
                    facet_col=id_vars[1],
                    title=description,
                    barmode='group'
                )
        else:
            # For simple crosstabs
            fig = px.bar(
                plot_df,
                x=plot_df.columns[0],
                y=plot_df.columns[1],
                title=description
            )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not create visualization: {str(e)}")

def main():
    # Set page config must be the first Streamlit command
    st.set_page_config(page_title="CSV Pivot Table Analyzer", layout="wide")
    
    # Rest of the Streamlit commands
    st.title("CSV Pivot Table Analyzer")
    st.write("Upload your CSV file to get suggested pivot tables and insights.")
    
    # Add OpenAI API key input
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Display basic information about the dataset
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Rows", len(df))
            with col2:
                st.metric("Number of Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Display column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Data Type': df.dtypes.astype(str),
                'Unique Values': df.nunique(),
                'Missing Values': df.isnull().sum(),
                'Used for Analysis': [is_meaningful_numeric_column(df, col) if pd.api.types.is_numeric_dtype(df[col]) 
                                    else not is_id_or_reference_column(col) for col in df.columns]
            })
            st.dataframe(col_info)
            
            # Display the first few rows
            st.subheader("First Few Rows")
            st.dataframe(df.head())
            
            # Get and display suggested pivot tables
            st.subheader("Most Relevant Pivot Tables")
            suggestions = analyze_dataframe(df)
            
            if not suggestions:
                st.warning("No suitable pivot tables could be generated. Make sure your CSV has both numeric and categorical columns.")
            else:
                # Rank and get top 6 pivot tables
                if api_key:
                    top_suggestions, raw_response, prompt = rank_pivot_tables(suggestions, df, api_key)
                    if raw_response:
                        with st.expander("View OpenAI's Analysis"):
                            st.subheader("Prompt sent to OpenAI:")
                            st.text(prompt)
                            st.write("---")
                            st.subheader("OpenAI's Response:")
                            st.text(raw_response)
                else:
                    st.warning("No OpenAI API key provided. Showing first 6 pivot tables.")
                    top_suggestions = [(desc, table, "Selected as one of the top pivot tables") 
                                     for desc, table in suggestions[:6]]
                
                for i, (description, pivot_table, explanation) in enumerate(top_suggestions):
                    with st.expander(f"Pivot Table {i+1}: {description}"):
                        st.write("**Why this pivot table is valuable:**")
                        st.write(explanation)
                        st.write("---")
                        st.dataframe(pivot_table)
                        create_visualization(pivot_table, description)

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main() 