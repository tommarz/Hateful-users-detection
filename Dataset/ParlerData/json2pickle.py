import sys
import json
import pickle


json_path = "/sise/home/tommarz/Hateful-users-detection/Dataset/ParlerData/all_users_posts_list.json"
pickle_path = json_path.replace('.json', '.p')
try:
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)

    print(f"JSON file '{json_path}' successfully converted to pickle file '{pickle_path}'.")

except FileNotFoundError:
    print(f"Error: JSON file '{json_path}' not found.")

except json.JSONDecodeError:
    print(f"Error: Invalid JSON file '{json_path}'.")

except Exception as e:
    print(f"Error occurred: {str(e)}")
