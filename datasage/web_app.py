"""
DataSage Streamlit Web Interface

A friendly GUI for exploring data quality with AI assistance.
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Import DataSage modules
from datasage.profiler import profile_df
from datasage.model import LocalLLMGenerator
from datasage.enhanced_generator import EnhancedLLMGenerator
from datasage.prompt_builder import PromptBuilder
from datasage.report_formatter import ReportFormatter
from datasage.verifier import OutputVerifier


def get_sample_datasets() -> Dict[str, str]:
    """Get available sample datasets from seaborn."""
    return {
        "Tips Dataset": "tips",
        "Titanic Dataset": "titanic", 
        "Flights Dataset": "flights",
        "Car Crashes Dataset": "car_crashes",
        "Iris Dataset": "iris",
        "Penguins Dataset": "penguins",
        "Diamonds Dataset": "diamonds",
        "MPG Dataset": "mpg"
    }


def load_sample_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a sample dataset from seaborn."""
    try:
        return sns.load_dataset(dataset_name)
    except Exception as e:
        st.error(f"Failed to load {dataset_name}: {str(e)}")
        return pd.DataFrame()


def display_dataframe_info(df: pd.DataFrame) -> None:
    """Display basic information about the dataframe."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_values:,}")
    
    with col4:
        memory_usage = df.memory_usage(deep=True).sum()
        memory_mb = memory_usage / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")


def get_data_exploration_insights(df: pd.DataFrame) -> Dict:
    """Generate comprehensive data exploration insights for data scientists."""
    insights = {
        'numerical_summary': {},
        'categorical_summary': {},
        'correlations': None,
        'missing_patterns': {},
        'outliers': {},
        'distributions': {},
        'recommendations': []
    }
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numerical analysis
    if numerical_cols:
        numerical_df = df[numerical_cols]
        insights['numerical_summary'] = {
            'columns': numerical_cols,
            'statistics': numerical_df.describe(),
            'skewness': numerical_df.skew(),
            'kurtosis': numerical_df.kurtosis()
        }
        
        # Correlation analysis
        if len(numerical_cols) > 1:
            insights['correlations'] = numerical_df.corr()
        
        # Outlier detection using IQR method
        for col in numerical_cols:
            Q1 = numerical_df[col].quantile(0.25)
            Q3 = numerical_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = numerical_df[(numerical_df[col] < lower_bound) | (numerical_df[col] > upper_bound)][col]
            insights['outliers'][col] = len(outliers)
    
    # Categorical analysis
    if categorical_cols:
        insights['categorical_summary'] = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            insights['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'distribution': value_counts.head(10).to_dict()
            }
    
    # Missing value patterns
    missing_data = df.isnull().sum()
    insights['missing_patterns'] = {
        'total_missing': missing_data.sum(),
        'missing_by_column': missing_data[missing_data > 0].to_dict(),
        'missing_percentage': (missing_data / len(df) * 100).round(2).to_dict()
    }
    
    # Generate recommendations
    recommendations = []
    
    # Missing data recommendations
    high_missing = missing_data[missing_data > len(df) * 0.3]
    if len(high_missing) > 0:
        recommendations.append(f"Consider dropping columns with >30% missing values: {list(high_missing.index)}")
    
    # Correlation recommendations
    if insights['correlations'] is not None:
        high_corr = insights['correlations'][insights['correlations'].abs() > 0.8]
        if not high_corr.empty:
            recommendations.append("High correlations detected - consider feature selection or dimensionality reduction")
    
    # Outlier recommendations
    high_outlier_cols = [col for col, count in insights['outliers'].items() if count > len(df) * 0.05]
    if high_outlier_cols:
        recommendations.append(f"Investigate outliers in: {high_outlier_cols}")
    
    # Cardinality recommendations
    high_cardinality = [col for col, info in insights['categorical_summary'].items() 
                       if info['unique_values'] > len(df) * 0.5]
    if high_cardinality:
        recommendations.append(f"High cardinality categorical columns may need encoding: {high_cardinality}")
    
    insights['recommendations'] = recommendations
    return insights


def display_data_exploration(df: pd.DataFrame):
    """Display comprehensive data exploration visualizations and statistics."""
    st.subheader("🔍 Data Exploration & Insights")
    
    # Get exploration insights
    with st.spinner("🔍 Analyzing dataset patterns..."):
        insights = get_data_exploration_insights(df)
    
    # Display key insights summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        numerical_cols = len(insights['numerical_summary'].get('columns', []))
        st.metric("Numerical Columns", numerical_cols)
    with col2:
        categorical_cols = len(insights['categorical_summary'])
        st.metric("Categorical Columns", categorical_cols)
    with col3:
        total_missing = insights['missing_patterns']['total_missing']
        st.metric("Missing Values", f"{total_missing:,}")
    with col4:
        total_outliers = sum(insights['outliers'].values()) if insights['outliers'] else 0
        st.metric("Outliers Detected", total_outliers)
    
    # Tabs for different exploration views
    explore_tab1, explore_tab2, explore_tab3, explore_tab4 = st.tabs([
        "📊 Distributions", "🔗 Correlations", "❓ Missing Data", "⚠️ Outliers & Quality"
    ])
    
    with explore_tab1:
        display_distributions(df, insights)
    
    with explore_tab2:
        display_correlations(df, insights)
    
    with explore_tab3:
        display_missing_patterns(df, insights)
    
    with explore_tab4:
        display_quality_issues(df, insights)


def display_distributions(df: pd.DataFrame, insights: Dict):
    """Display distribution plots for numerical and categorical variables."""
    numerical_cols = insights['numerical_summary'].get('columns', [])
    categorical_cols = list(insights['categorical_summary'].keys())
    
    if numerical_cols:
        st.subheader("📈 Numerical Distributions")
        
        # Select columns to plot
        selected_num_cols = st.multiselect(
            "Select numerical columns to visualize:",
            numerical_cols,
            default=numerical_cols[:4] if len(numerical_cols) <= 4 else numerical_cols[:2]
        )
        
        if selected_num_cols:
            # Create distribution plots
            n_cols = min(2, len(selected_num_cols))
            n_rows = (len(selected_num_cols) + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(selected_num_cols):
                sns.histplot(data=df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(len(selected_num_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Display statistics
            st.subheader("📊 Statistical Summary")
            st.dataframe(df[selected_num_cols].describe())
    
    if categorical_cols:
        st.subheader("📋 Categorical Distributions")
        
        # Select columns to plot
        selected_cat_cols = st.multiselect(
            "Select categorical columns to visualize:",
            categorical_cols,
            default=categorical_cols[:2] if len(categorical_cols) <= 2 else [categorical_cols[0]]
        )
        
        if selected_cat_cols:
            for col in selected_cat_cols:
                value_counts = df[col].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(data=df, y=col, order=value_counts.index, ax=ax)
                ax.set_title(f'Distribution of {col} (Top 10)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


def display_correlations(df: pd.DataFrame, insights: Dict):
    """Display correlation analysis."""
    if insights['correlations'] is not None:
        st.subheader("🔗 Correlation Matrix")
        
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(insights['correlations'], dtype=bool))
        sns.heatmap(insights['correlations'], 
                   mask=mask, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   ax=ax)
        plt.title('Correlation Matrix (Lower Triangle)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Highlight strong correlations
        st.subheader("🎯 Strong Correlations")
        corr_pairs = []
        for i in range(len(insights['correlations'].columns)):
            for j in range(i+1, len(insights['correlations'].columns)):
                corr_val = insights['correlations'].iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    corr_pairs.append({
                        'Variable 1': insights['correlations'].columns[i],
                        'Variable 2': insights['correlations'].columns[j],
                        'Correlation': round(corr_val, 3),
                        'Strength': 'Very Strong' if abs(corr_val) > 0.8 else 'Strong'
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.reindex(corr_df.Correlation.abs().sort_values(ascending=False).index)
            st.dataframe(corr_df, width='stretch')
        else:
            st.info("No strong correlations (>0.5) detected between variables.")
    
    else:
        st.info("Correlation analysis requires at least 2 numerical columns.")


def display_missing_patterns(df: pd.DataFrame, insights: Dict):
    """Display missing data analysis."""
    st.subheader("❓ Missing Data Analysis")
    
    missing_data = insights['missing_patterns']
    
    if missing_data['total_missing'] > 0:
        # Missing data summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Missing Values", f"{missing_data['total_missing']:,}")
            
        with col2:
            missing_pct = (missing_data['total_missing'] / (len(df) * len(df.columns))) * 100
            st.metric("Overall Missing %", f"{missing_pct:.2f}%")
        
        # Missing data by column
        if missing_data['missing_by_column']:
            st.subheader("📊 Missing Values by Column")
            
            missing_df = pd.DataFrame([
                {
                    'Column': col,
                    'Missing Count': count,
                    'Missing %': missing_data['missing_percentage'][col]
                }
                for col, count in missing_data['missing_by_column'].items()
            ]).sort_values('Missing Count', ascending=False)
            
            # Bar chart of missing values
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=missing_df, x='Missing Count', y='Column', ax=ax)
            ax.set_title('Missing Values by Column')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Data table
            st.dataframe(missing_df, width='stretch')
            
            # Missing data pattern visualization
            st.subheader("🔍 Missing Data Patterns")
            missing_matrix = df.isnull()
            
            if missing_matrix.sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(missing_matrix, 
                           yticklabels=False, 
                           cbar=True, 
                           cmap='viridis',
                           ax=ax)
                ax.set_title('Missing Data Pattern (Yellow = Missing)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    else:
        st.success("🎉 No missing values detected in this dataset!")


def display_quality_issues(df: pd.DataFrame, insights: Dict):
    """Display data quality issues and outliers."""
    st.subheader("⚠️ Data Quality Assessment")
    
    # Outliers analysis
    if insights['outliers']:
        st.subheader("📈 Outliers Detection")
        
        outlier_data = [
            {
                'Column': col,
                'Outliers Count': count,
                'Outliers %': round((count / len(df)) * 100, 2)
            }
            for col, count in insights['outliers'].items()
            if count > 0
        ]
        
        if outlier_data:
            outlier_df = pd.DataFrame(outlier_data).sort_values('Outliers Count', ascending=False)
            st.dataframe(outlier_df, width='stretch')
            
            # Visualize outliers for selected column
            high_outlier_cols = [item['Column'] for item in outlier_data if item['Outliers %'] > 5]
            if high_outlier_cols:
                selected_col = st.selectbox("Select column to visualize outliers:", high_outlier_cols)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Box plot
                sns.boxplot(data=df, y=selected_col, ax=ax1)
                ax1.set_title(f'Box Plot - {selected_col}')
                
                # Histogram with outliers highlighted
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                sns.histplot(data=df, x=selected_col, kde=True, ax=ax2, alpha=0.7)
                ax2.axvline(lower_bound, color='red', linestyle='--', label='Outlier Bounds')
                ax2.axvline(upper_bound, color='red', linestyle='--')
                ax2.set_title(f'Distribution - {selected_col}')
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.success("🎉 No significant outliers detected!")
    
    # Recommendations
    if insights['recommendations']:
        st.subheader("💡 Data Quality Recommendations")
        for i, rec in enumerate(insights['recommendations'], 1):
            st.write(f"{i}. {rec}")
    else:
        st.success("✅ No major data quality issues detected!")


def generate_profile_report(df: pd.DataFrame) -> Optional[str]:
    """Generate AI-powered data quality report."""
    try:
        # Initialize components
        llm = EnhancedLLMGenerator()  # Use enhanced generator
        prompt_builder = PromptBuilder()
        formatter = ReportFormatter(use_enhanced_fallback=True)  # Enable enhanced fallback
        verifier = OutputVerifier()
        
        # Generate profile
        with st.spinner("🔍 Analyzing data structure..."):
            profile = profile_df(df)
        
        # Build prompt
        with st.spinner("📝 Building analysis prompt..."):
            prompt = prompt_builder.build_summary_prompt(profile)
        
        # Generate report
        with st.spinner("🤖 AI is analyzing your data..."):
            raw_output = llm.generate_markdown(prompt)
        
        # Format and verify
        with st.spinner("✨ Formatting report..."):
            title = f"Data Quality Report: {st.session_state.dataset_name or 'Dataset'}"
            formatted_report = formatter.assemble_report([raw_output], profile, title=title)
            
        with st.spinner("🔍 Verifying output quality..."):
            annotated_report, verification_results = verifier.verify_and_annotate_report(
                formatted_report, profile, add_annotations=True
            )
            verification_summary = verifier.get_verification_summary()
            
            if verification_summary.get('total_claims_checked', 0) > 0:
                success_rate = verification_summary.get('success_rate', 0)
                if success_rate < 90:  # Less than 90% accurate
                    st.warning(f"⚠️ Report quality check: {success_rate:.1f}% of claims verified successfully.")
                
            # Use the annotated report as the final output
            formatted_report = annotated_report
            
        return formatted_report
        
    except Exception as e:
        st.error(f"Failed to generate report: {str(e)}")
        return None


def generate_dataset_insights(df: pd.DataFrame, user_question: str, profile_data: Optional[Dict] = None) -> str:
    """Generate AI insights based on user questions about the dataset."""
    try:
        # Get or generate profile data
        if profile_data is None:
            profile_data = profile_df(df)
        
        # Get comprehensive exploration insights
        exploration_insights = get_data_exploration_insights(df)
        
        # Build enhanced context using PromptBuilder
        builder = PromptBuilder()
        basic_context = builder.build_summary_prompt(profile_data)
        exploration_context = builder.format_exploration_insights(exploration_insights)
        
        dataset_context = basic_context + exploration_context
        
        # Create a specialized prompt for Q&A
        prompt = f"""You are a data analyst assistant helping users understand their dataset. Answer the user's question about their data in a helpful, clear, and accurate way.

{dataset_context}

User Question: {user_question}

Please provide a helpful answer based on the dataset information above. Be specific and reference actual data when possible. If you need to make calculations or observations, be clear about what you're basing them on.

Answer:"""
        
        # Generate response using enhanced generator
        llm = EnhancedLLMGenerator()
        response = llm.generate_markdown(prompt)
        
        # Clean up the response
        if response.startswith("Answer:"):
            response = response[7:].strip()
        
        return response
        
    except Exception as e:
        return f"Sorry, I encountered an error while analyzing your question: {str(e)}"


def display_chat_interface(df: pd.DataFrame):
    """Display the interactive chat interface for dataset insights."""
    st.markdown("*Ask me anything about your dataset and I'll provide insights based on the data!*")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'dataset_profile' not in st.session_state:
        st.session_state.dataset_profile = None
    
    # Pre-generate profile for faster responses
    if st.session_state.dataset_profile is None:
        with st.spinner("🔍 Analyzing dataset for faster insights..."):
            st.session_state.dataset_profile = profile_df(df)
    
    # Example questions
    with st.expander("💡 Example Questions You Can Ask"):
        st.markdown("""
        **Data Quality Questions:**
        - "What are the main data quality issues in this dataset?"
        - "Which columns have missing values and how should I handle them?"
        - "Are there any outliers I should be concerned about?"
        
        **Exploratory Questions:**
        - "What are the key patterns in this data?"
        - "Which columns are most important for analysis?"
        - "What insights can you provide about [specific column]?"
        
        **Practical Questions:**
        - "How should I clean this data for analysis?"
        - "What analysis would you recommend for this dataset?"
        - "What are the potential use cases for this data?"
        """)
    
    # Quick action buttons
    st.markdown("**Quick Insights:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Data Quality Overview"):
            question = "What are the main data quality issues and strengths in this dataset?"
            with st.spinner("Analyzing data quality..."):
                response = generate_dataset_insights(df, question, st.session_state.dataset_profile)
                st.session_state.chat_history.append({"question": question, "answer": response})
                st.rerun()
    
    with col2:
        if st.button("📊 Key Patterns"):
            question = "What are the most interesting patterns and insights in this dataset?"
            with st.spinner("Finding patterns..."):
                response = generate_dataset_insights(df, question, st.session_state.dataset_profile)
                st.session_state.chat_history.append({"question": question, "answer": response})
                st.rerun()
    
    with col3:
        if st.button("💡 Analysis Recommendations"):
            question = "What analysis and visualization approaches would you recommend for this dataset?"
            with st.spinner("Generating recommendations..."):
                response = generate_dataset_insights(df, question, st.session_state.dataset_profile)
                st.session_state.chat_history.append({"question": question, "answer": response})
                st.rerun()
    
    # Chat interface
    user_question = st.text_input(
        "Ask a question about your dataset:",
        placeholder="e.g., What are the main issues with missing data in this dataset?",
        key="user_question"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🤖 Ask AI", type="primary")
    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask_button and user_question.strip():
        with st.spinner("🤔 Thinking about your question..."):
            response = generate_dataset_insights(df, user_question, st.session_state.dataset_profile)
            st.session_state.chat_history.append({"question": user_question, "answer": response})
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**🙋 You asked:** {chat['question']}")
                st.markdown(f"**🤖 AI Response:**")
                st.markdown(chat['answer'])
                st.markdown("---")
    else:
        st.info("💬 Start a conversation by asking a question about your dataset!")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DataSage",
        page_icon="🧙‍♂️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧙‍♂️ DataSage")
    st.subheader("*Your casual AI companion for data quality insights*")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'report_content' not in st.session_state:
        st.session_state.report_content = None

    # Sidebar for data loading
    with st.sidebar:
        st.header("📊 Load Your Data")
        
        # Data source selection
        data_source = st.radio(
            "Choose data source:",
            ["Upload CSV File", "Sample Datasets"],
            key="data_source"
        )
        
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type="csv",
                help="Upload your CSV file to get started with analysis"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.report_generated = False
                    st.session_state.report_content = None
                    st.success(f"✅ Loaded {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        else:  # Sample Datasets
            sample_datasets = get_sample_datasets()
            selected_dataset = st.selectbox(
                "Choose a sample dataset:",
                [""] + list(sample_datasets.keys()),
                key="sample_dataset"
            )
            
            if selected_dataset and selected_dataset != "":
                if st.button("Load Sample Dataset", width='stretch'):
                    dataset_key = sample_datasets[selected_dataset]
                    df = load_sample_dataset(dataset_key)
                    
                    if not df.empty:
                        st.session_state.df = df
                        st.session_state.dataset_name = selected_dataset
                        st.session_state.report_generated = False
                        st.session_state.report_content = None
                        st.success(f"✅ Loaded {selected_dataset}")
        
        # Clear data button in sidebar
        if st.session_state.df is not None:
            st.markdown("---")
            if st.button("🗑️ Clear Data", width='stretch'):
                st.session_state.df = None
                st.session_state.dataset_name = None
                st.session_state.report_generated = False
                st.session_state.report_content = None
                # Clear chat history too
                if 'chat_history' in st.session_state:
                    st.session_state.chat_history = []
                if 'dataset_profile' in st.session_state:
                    st.session_state.dataset_profile = None
                st.rerun()  # Rerun to clear the interface

    # Main content area
    if st.session_state.df is not None:
        # Tab-based navigation
        tab1, tab2, tab3 = st.tabs(["📊 Dataset Explorer", "📋 AI Report", "💬 Chat with AI"])
        
        with tab1:
            st.subheader(f"📊 Dataset: {st.session_state.dataset_name}")
            
            # Display basic info
            display_dataframe_info(st.session_state.df)
            
            st.markdown("---")
            
            # Data preview options
            col1, col2 = st.columns([1, 3])
            with col1:
                preview_rows = st.number_input(
                    "Rows to preview:", 
                    min_value=5, 
                    max_value=min(1000, len(st.session_state.df)), 
                    value=min(10, len(st.session_state.df)),
                    step=5
                )
            
            # Display the dataframe
            st.subheader("📋 Data Preview")
            st.dataframe(
                st.session_state.df.head(preview_rows), 
                width='stretch',
                height=400
            )
            
            # Column information
            st.subheader("📈 Column Information")
            
            col_info = []
            for col in st.session_state.df.columns:
                col_data = st.session_state.df[col]
                col_info.append({
                    'Column': col,
                    'Type': str(col_data.dtype),
                    'Non-Null Count': col_data.count(),
                    'Null Count': col_data.isnull().sum(),
                    'Unique Values': col_data.nunique()
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, width='stretch')
            
            # Add comprehensive data exploration
            st.markdown("---")
            display_data_exploration(st.session_state.df)

        with tab2:
            st.subheader("🤖 AI Data Quality Report")
            
            # Generate/Regenerate button
            if not st.session_state.report_generated:
                if st.button("🤖 Generate AI Report", type="primary", width='stretch'):
                    with st.spinner("Generating your personalized data quality report..."):
                        report = generate_profile_report(st.session_state.df)
                        if report:
                            st.session_state.report_content = report
                            st.session_state.report_generated = True
                            st.success("✅ Report generated successfully!")
                            st.rerun()  # Rerun to show the report
                st.info("👆 Click the button above to generate your AI-powered data quality report!")
            else:
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🔄 Regenerate", width='stretch'):
                        with st.spinner("Regenerating your data quality report..."):
                            report = generate_profile_report(st.session_state.df)
                            if report:
                                st.session_state.report_content = report
                                st.success("✅ Report regenerated successfully!")
                                st.rerun()  # Rerun to update the report
                
                with col2:
                    # Download button for the report
                    if st.session_state.report_content:
                        report_bytes = st.session_state.report_content.encode('utf-8')
                        st.download_button(
                            label="📥 Download Report",
                            data=report_bytes,
                            file_name=f"datasage_report_{st.session_state.dataset_name}.md",
                            mime="text/markdown",
                            width='stretch'
                        )
            
            if st.session_state.report_generated and st.session_state.report_content:
                st.markdown("---")
                # Display the report
                st.markdown(st.session_state.report_content)
                
            elif st.session_state.report_generated:
                st.error("❌ Report generation failed. Please try again.")
        
        with tab3:
            display_chat_interface(st.session_state.df)
    
    else:
        # Welcome message when no data is loaded
        st.markdown("""
        ## 👋 Welcome to DataSage!
        
        **Your friendly AI companion for data quality insights.**
        
        ### Getting Started:
        1. 📊 **Load your data** using the sidebar
           - Upload a CSV file, or
           - Try one of our sample datasets
        
        2. 🤖 **Generate AI insights** with one click
           - Get automated data quality analysis
           - Discover patterns and potential issues
           - Receive actionable recommendations
        
        3. 📋 **Explore results** in organized tabs
           - Browse your dataset interactively
           - Read your personalized AI report
        
        ### ✨ What makes DataSage special?
        - **AI-powered analysis** using local language models
        - **Privacy-focused** - your data never leaves your machine  
        - **Beginner-friendly** - no coding required
        - **Comprehensive insights** - from basic stats to advanced patterns
        
        **Ready to dive in?** Choose a data source from the sidebar! 🚀
        """)


if __name__ == "__main__":
    main()
