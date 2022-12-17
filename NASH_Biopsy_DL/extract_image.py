import lxml.etree as ET
import requests
import os
import docx
import scandir
import pandas as pd
from tqdm import tqdm 

def img_from_xml(xml_file_name):
    tree = ET.parse(xml_file_name)
    root = tree.getroot()

    # Find all elements with the tag 'image'
    images = root.findall('.//image')

    # Iterate through the image elements and download the images
    count = 0
    for image in images:
        count+=1
        # Get the image URL
        url = image.get('url')
        # Download the image
        response = requests.get(url)
        # Save the image to a file
        # To use image_name: os.path.basename(url)
        # To use image number: count
        open(os.path.join('biopsy_images', xml_file_name+'_'+count), 'wb').write(response.content)

def img_from_docx(docx_file_name):
    # To use install python-docx
    document = docx.Document(docx_file_name)

    # Iterate through the paragraphs in the document
    for paragraph in document.paragraphs:
        # Iterate through the inline shapes in the paragraph 
        # Remove the following loop if only first image needs to be extracted 
        for shape in paragraph.inline_shapes:
            # Get the image file name
            image_file = shape.filename
            # Save the image to a file
            open(os.path.join('biopsy_images', image_file), 'wb').write(shape.blob)

# def NAS_from_docx(docx_file_name):
    

def main(folder_path,excel_path):
    # dir_list = os.listdir(folder_path)[:1]

    df = pd.read_excel(excel_path).set_index('SLIDE NO')
    for slide_number in tqdm(df.index):
        if slide_number.endswith('22'):
            index = slide_number.replace('-','').replace('/','')
            index_lower = index.lower()
            for paths, dirs, files in scandir.walk(folder_path):
            #for (paths, dirs, files) in os.walk(folder):
                for file in files:
                    if index in file or index_lower in file:
                        # with open(os.path.join(paths, file), 'r') as f:
                        img_from_docx(os.path.join(paths, file))
        
