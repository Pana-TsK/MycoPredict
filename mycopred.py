import argparse
import sys
import logging
from main import Main  

# Set up logging for the CLI
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the logo
logo = """
███╗   ███╗██╗   ██╗ ██████╗ ██████╗ ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗ ██████╗ ██████╗ 
████╗ ████║╚██╗ ██╔╝██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
██╔████╔██║ ╚████╔╝ ██║     ██║   ██║██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║   ██║██████╔╝
██║╚██╔╝██║  ╚██╔╝  ██║     ██║   ██║██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║   ██║██╔══██╗
██║ ╚═╝ ██║   ██║   ╚██████╗╚██████╔╝██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ╚██████╔╝██║  ██║
╚═╝     ╚═╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
                                                                                                          
"""

# Function to display the logo
def display_logo():
    print(logo)

# Define the main function to run the pipeline
def run_pipeline(input_path, output_path):
    display_logo()
    try:
        # Create an instance of Main
        main_instance = Main()

        # Run the pipeline (load data, preprocess, predict, and export)
        result_df = main_instance(input_path=input_path, output_path=output_path)

        # Output the results
        print("Pipeline completed successfully.")
        print(f"Results have been saved to: {output_path}")
        print(result_df.head())  # Display the first few rows of the output for confirmation

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        sys.exit(1)

# Set up the command-line argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="MycoPred CLI for predicting and processing Mycobacterium tuberculosis data.")

    # Define the input and output arguments
    parser.add_argument(
        'input_path', type=str, help='Path to the input CSV file containing SMILES data'
    )
    parser.add_argument(
        '--output_path', type=str, default='results.csv', help='Path to save the output results (default: results.csv)'
    )

    return parser.parse_args()

# Main entry point for the script
if __name__ == '__main__':
    args = parse_arguments()

    # Run the pipeline with the provided input and output paths
    run_pipeline(args.input_path, args.output_path)
