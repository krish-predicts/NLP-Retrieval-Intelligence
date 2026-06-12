import pandas as pd
from src.insights.pipeline import run_insights

if __name__ == "__main__":
    print("Initializing NLP Retrieval Intelligence Pipeline...")
    
    # 1. Create or load your dummy/real DataFrame here
    # (Replacing this with your actual data source)
    data = {"text": ["Sample customer feedback sentence."]}
    df = pd.DataFrame(data)
    
    # 2. Run the main orchestration script
    try:
        output_df = run_insights(df)
        print("Pipeline executed successfully!")
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
        