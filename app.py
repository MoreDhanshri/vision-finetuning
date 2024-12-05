import streamlit as st
import os
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import json

# App title
st.title('Image Processing and JSONL Creator')

# Image upload
uploaded_images = st.file_uploader('Upload images', type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'], accept_multiple_files=True)

# CSV upload
uploaded_csv = st.file_uploader('Upload CSV file containing questions and answers', type='csv')

# Process button
if st.button('Process'):
    if uploaded_images and uploaded_csv:
        # Load the CSV file
        df = pd.read_csv(uploaded_csv)
        questions = df['question'].tolist()
        answers = df['answer'].tolist()

        # Convert images to JPEG and encode in base64
        images_base64 = []
        for image_file in uploaded_images:
            try:
                # Open and convert the image
                img = Image.open(image_file)
                img = img.convert('RGB')
                # Save as JPEG to a buffer
                buffered = BytesIO()
                img.save(buffered, format='JPEG')
                # Encode to base64
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                images_base64.append(img_str)
            except Exception as e:
                st.error(f"Failed to process {image_file.name}: {e}")

        # Create DataFrame
        ds_train = pd.DataFrame({
            'question': questions,
            'answer': answers,
            'image': images_base64
        })
        SYSTEM_PROMPT = """
        Generate an answer to the question based on the image of the device provided. Questions will inquire about the make and model of the device in the image.

        You will read the question and examine the corresponding image to provide an accurate answer.

        ## Steps

        1. **Read the Question**: Carefully analyze the question to understand what information is being asked.

        2. **Examine the Image**:
        - **Identify Relevant Text**: For questions requiring specific details like make or model, focus on the relevant text or labels within the image to extract the necessary information. There may be multiple relevant text areas in the image, so be sure to consider all relevant sections.
        - **Analyze the Whole Image**: Ensure to consider the entire device, including logos, labels, and other identifying marks that can help determine the make and model.

        3. **Formulate a Reasoned Answer**:
        - Extract the exact text from the image for both the make and model.
        - If either the make or model is not clearly identifiable, use the information available to provide the best possible answer.

        4. **Format the Output**:
        - Provide the answer in JSON format with keys "make" and "model".
        - If the make or model cannot be determined, use "Unknown" as the value for that key.

        ## Output Format

        Provide your answer in a concise and clear JSON format. The JSON object should have two keys: "make" and "model".

        ### Example:
        {  
        "make": "TRENDnet",  
        "model": "TEG-S708"  
        }

        If the make or model is not identifiable, return "Unknown" for that key.
        {  
        "make": "Unknown",  
        "model": "TEG-S708"  
        }

        # Notes

        - Always prioritize accuracy and clarity in your responses.
        - If multiple devices are listed, return all the makes and models in the same order as they appear in the image.
        - If the information is not present in the image, try to reason about the question using the information you can gather from the image.
        - Ensure reasoning steps logically lead to the conclusions before stating your final answer.

        """

        # Generate JSONL data
        json_data = []
        for idx, example in ds_train.iterrows():
            # ...existing code...
            system_message = {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            }
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question [{idx}]: {example['question']}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example['image']}"}}
                ]
            }
            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": example["answer"]}]
            }
            all_messages = [system_message, user_message, assistant_message]
            json_data.append({"messages": all_messages})
            # ...existing code...

        # Convert JSON data to JSONL string
        jsonl_str = '\n'.join([json.dumps(message) for message in json_data])

        # Provide download link for JSONL file
        st.download_button(
            label='Download JSONL file',
            data=jsonl_str,
            file_name='visionFT_device.jsonl',
            mime='application/json'
        )
    else:
        st.error('Please upload both images and CSV file')