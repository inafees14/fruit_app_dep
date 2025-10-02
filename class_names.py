import os

# --- IMPORTANT ---
# Set this path to the main folder containing your training image subfolders 
# (e.g., the folder that contains 'Apple', 'Banana', etc.).
DATA_DIR = "E:/Desktop/Plants - Copy" 

# --- SCRIPT ---
def generate_class_names_file():
    """Finds class folders, sorts them, and writes to class_names.txt."""
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Training data directory not found at '{DATA_DIR}'")
        print("Please update the DATA_DIR variable in the script.")
        return

    try:
        # Find all subdirectories, which are the class names
        class_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        
        # Sort them alphabetically to match the order Keras uses for training
        class_names.sort()
        
        if not class_names:
            print(f"Error: No class folders found in '{DATA_DIR}'.")
            return
            
        # Write the sorted names to the class_names.txt file
        output_path = "class_names.txt"
        with open(output_path, "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        print(f"✅ Successfully created '{output_path}' with {len(class_names)} classes.")
        print("--- File Contents ---")
        with open(output_path, "r") as f:
            print(f.read())
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_class_names_file()