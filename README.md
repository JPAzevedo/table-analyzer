# CSV Pivot Table Analyzer

A Streamlit application that analyzes CSV files and generates meaningful pivot tables with AI-powered insights.

## Features

- Upload and analyze CSV files
- Automatic detection of numeric and categorical columns
- Generation of relevant pivot tables
- AI-powered analysis of pivot tables using OpenAI
- Interactive visualizations using Plotly
- Support for formatted numbers (including currency)

## Local Development

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployment

This application is designed to be deployed on Streamlit Cloud:

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and the `app.py` file
6. Click "Deploy!"

## Environment Variables

When deploying, you'll need to set up the following environment variable:
- `OPENAI_API_KEY`: Your OpenAI API key

## Usage

1. Upload a CSV file
2. Enter your OpenAI API key (if you want AI-powered insights)
3. View the generated pivot tables and insights
4. Interact with the visualizations

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt` 