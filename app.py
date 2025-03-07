import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Tuple
import openai

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
    # If already numeric, return True
    if pd.api.types.is_numeric_dtype(df[col]):
        return True
    
    # If not string, return False
    if not pd.api.types.is_string_dtype(df[col]):
        return False
    
    # Try to convert to numeric, handling currency symbols and commas
    try:
        # Remove currency symbols and commas, then try to convert
        test_values = df[col].astype(str).str.replace('$', '').str.replace(',', '')
        pd.to_numeric(test_values, errors='raise')
        return True
    except:
        return False

def analyze_dataframe(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """
    Analyze the dataframe and suggest relevant pivot tables.
    Returns a list of tuples containing (description, pivot_table)
    """
    suggestions = []
    
    # Get all columns except ID/reference columns
    all_cols = [col for col in df.columns if not is_id_or_reference_column(col)]
    
    # Get numeric and categorical columns
    numeric_cols = [col for col in all_cols if is_numeric_column(df, col)]
    categorical_cols = [col for col in all_cols if not is_numeric_column(df, col) and not pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Convert string numeric columns to actual numeric values
    for col in numeric_cols:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    
    # Define common aggregation functions
    agg_funcs = ['mean', 'min', 'max', 'count']
    
    # 1. Basic Statistics for each numeric column
    for num_col in numeric_cols:
        # Basic statistics
        stats = df[num_col].describe()
        description = f"Basic Statistics for {num_col}"
        suggestions.append((description, pd.DataFrame(stats)))
        
        # Distribution by categorical columns
        for cat_col in categorical_cols:
            if df[cat_col].nunique() <= 20:
                # Average by category
                avg_by_cat = pd.pivot_table(
                    df,
                    values=num_col,
                    index=cat_col,
                    aggfunc='mean'
                )
                description = f"Average {num_col} by {cat_col}"
                suggestions.append((description, avg_by_cat))
                
                # Distribution statistics by category
                dist_by_cat = pd.pivot_table(
                    df,
                    values=num_col,
                    index=cat_col,
                    aggfunc=['mean', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
                )
                description = f"Distribution of {num_col} by {cat_col}"
                suggestions.append((description, dist_by_cat))
                
                # Count by category
                count_by_cat = pd.pivot_table(
                    df,
                    values=num_col,
                    index=cat_col,
                    aggfunc='count'
                )
                description = f"Count of {num_col} by {cat_col}"
                suggestions.append((description, count_by_cat))
    
    # 2. Combined Analysis: Numeric by two categoricals
    for num_col in numeric_cols:
        for i, cat1 in enumerate(categorical_cols):
            for cat2 in categorical_cols[i+1:]:
                if df[cat1].nunique() <= 20 and df[cat2].nunique() <= 20:
                    # Average by both categories
                    avg_by_both = pd.pivot_table(
                        df,
                        values=num_col,
                        index=[cat1, cat2],
                        aggfunc='mean'
                    )
                    description = f"Average {num_col} by {cat1} and {cat2}"
                    suggestions.append((description, avg_by_both))
                    
                    # Distribution by both categories
                    dist_by_both = pd.pivot_table(
                        df,
                        values=num_col,
                        index=[cat1, cat2],
                        aggfunc=['mean', 'min', 'max']
                    )
                    description = f"Distribution of {num_col} by {cat1} and {cat2}"
                    suggestions.append((description, dist_by_both))
    
    # 3. Categorical Relationships
    for i, cat1 in enumerate(categorical_cols):
        for cat2 in categorical_cols[i+1:]:
            if df[cat1].nunique() <= 20 and df[cat2].nunique() <= 20:
                # Count distribution
                crosstab = pd.crosstab(df[cat1], df[cat2], margins=True)
                description = f"Distribution of {cat1} by {cat2}"
                suggestions.append((description, crosstab))
                
                # Percentage distribution (row-wise)
                crosstab_pct_row = pd.crosstab(df[cat1], df[cat2], normalize='index') * 100
                description = f"Percentage Distribution of {cat1} by {cat2} (Row-wise)"
                suggestions.append((description, crosstab_pct_row))
                
                # Percentage distribution (column-wise)
                crosstab_pct_col = pd.crosstab(df[cat1], df[cat2], normalize='columns') * 100
                description = f"Percentage Distribution of {cat1} by {cat2} (Column-wise)"
                suggestions.append((description, crosstab_pct_col))
    
    # 4. Time-based Analysis (if applicable)
    date_cols = [col for col in all_cols if pd.api.types.is_datetime64_any_dtype(df[col])]
    for date_col in date_cols:
        # Extract year, month, and day
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Day'] = df[date_col].dt.day
        
        # Analyze numeric columns by time dimensions
        for num_col in numeric_cols:
            # By year
            yearly_stats = pd.pivot_table(
                df,
                values=num_col,
                index='Year',
                aggfunc=agg_funcs
            )
            description = f"Yearly Statistics of {num_col}"
            suggestions.append((description, yearly_stats))
            
            # By month
            monthly_stats = pd.pivot_table(
                df,
                values=num_col,
                index='Month',
                aggfunc=agg_funcs
            )
            description = f"Monthly Statistics of {num_col}"
            suggestions.append((description, monthly_stats))
            
            # By year and month
            yearly_monthly_stats = pd.pivot_table(
                df,
                values=num_col,
                index=['Year', 'Month'],
                aggfunc=agg_funcs
            )
            description = f"Yearly and Monthly Statistics of {num_col}"
            suggestions.append((description, yearly_monthly_stats))
    
    # Clean up temporary columns
    if 'Year' in df.columns:
        df.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)
    
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
    Format your response as:
    indices: 0,1,2,3,4,5
    explanations:
    0. [explanation for first table]
    1. [explanation for second table]
    etc.
    
    Pivot Tables:
    {chr(10).join(f"{i}. {desc}" for i, (desc, _) in enumerate(suggestions))}"""
    
    try:
        # Set the API key
        openai.api_key = api_key
        
        # Get ranking from OpenAI using the older API format
        response = openai.ChatCompletion.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a business intelligence expert specializing in data analysis and strategic insights. Your task is to identify the most valuable pivot tables that drive business decisions."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get the raw response
        raw_response = response.choices[0].message.content
        
        # Parse the response
        content = raw_response
        indices_line = content.split('\n')[0]
        explanations = content.split('explanations:')[1].strip().split('\n')
        
        # Get indices
        indices = [int(idx.strip()) for idx in indices_line.split('indices:')[1].strip().split(',')]
        
        # Create explanations dictionary
        explanations_dict = {}
        for exp in explanations:
            if exp.strip():
                idx, text = exp.split('.', 1)
                explanations_dict[int(idx.strip())] = text.strip()
        
        # Return the top 6 pivot tables with explanations, raw response, and prompt
        return [(suggestions[i][0], suggestions[i][1], explanations_dict.get(i, "No explanation provided")) 
                for i in indices if i < len(suggestions)], raw_response, prompt
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
    st.set_page_config(page_title="CSV Pivot Table Analyzer", layout="wide")
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