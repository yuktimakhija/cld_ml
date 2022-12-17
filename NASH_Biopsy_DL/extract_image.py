import lxml.etree as ET
import requests
import os
import docx

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
        open(os.path.join('images', xml_file_name+'_'+count), 'wb').write(response.content)

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
            open(os.path.join('images', image_file), 'wb').write(shape.blob)

def main(folder_path):
    dir_list = os.listdir(folder_path)[:1]
    for fname in dir_list:
        patient_id = fname[:-4]
        # eg: patient_id = 'IPID0073044'
        
