import argparse

# Create the argument parser
parser = argparse.ArgumentParser()

# Add a boolean argument
parser.add_argument('--flag', action='store_true', default=False)

# Parse the command-line arguments
args = parser.parse_args()

# Access the boolean value
flag_value = args.flag

# Print the boolean value
print(flag_value)