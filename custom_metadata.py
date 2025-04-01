import json

# Load the JSON data from the file into a dictionary
with open('/home/ubuntu/drumgpt_prompts.json', 'r') as f:
    file_preset_breakdown_dict = json.load(f)

def get_custom_metadata(info, audio):
    
    # Pass in the relative path of the audio file as the prompt
    prompt = "Generate drum kit: " + file_preset_breakdown_dict[info["relpath"]]
    print(prompt)
    return {"prompt": prompt}