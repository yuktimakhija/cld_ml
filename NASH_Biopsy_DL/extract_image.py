import lxml.etree as ET
import shutil
import os, re
import docx
import scandir
import docx2txt
import pandas as pd
from tqdm import tqdm 
import tempfile
import subprocess

def img_from_xml(xml_file_name):
    tree = ET.parse(xml_file_name)
    root = tree.getroot()

    # Find all elements with the tag 'image'
    images = root.findall('.//image')

    # Iterate through the image elements and download the images
    # count = 0
    # for image in images:
    #     count+=1
    #     # Get the image URL
    #     url = image.get('url')
    #     # Download the image
    #     response = requests.get(url)
    #     # Save the image to a file
    #     # To use image_name: os.path.basename(url)
    #     # To use image number: count
    #     open(os.path.join('biopsy_images', xml_file_name+'_'+count), 'wb').write(response.content)

def img_from_docx(docx_file, short_filename,gt_df):
    print('img_from_docx started')
    # To use install python-docx
    if docx_file.endswith('.doc'):
       # converting .doc to .docx
       doc_file = docx_file
       docx_file = docx_file + 'x'
       if not os.path.exists(docx_file):
          os.system('antiword ' + doc_file + ' > ' + docx_file)
    # document = docx.Document(docx_file)

    # Iterate through the paragraphs in the document
    # count = 0
    # for paragraph in document.paragraphs:
    #     # Iterate through the inline shapes in the paragraph 
    #     # Remove the following loop if only first image needs to be extracted 
    #     for shape in paragraph.inline_shapes:
    #         # Get the image file name
    #         # Save the image to a file
    #         open(os.path.join('biopsy_images', short_filename+'_'+count+'.jpg'), 'wb').write(shape.blob)
    #         count += 1

    get_imgs_from_document(docx_file, short_filename, 'docx',gt_df)
    
    print('img_from_docx ended')

def NAS_from_docx(docx_file_name):
    # Extract text from DOCX file
    text = docx2txt.process(docx_file_name)

def get_imgs_from_document(filename, short_filename, filetype, gt_df):
    # filetype is rtx or docx
    tempdir = tempfile.TemporaryDirectory()
    text = subprocess.check_output(['pandoc', filename, '-f', filetype, '-t', 'plain', '--extract-media', tempdir.name])

    # search for the ground truth string
    # if re.search('S(\d.\d+)A(\d)F(\d)', str(text)):
    if re.search('S(\d)A(\d)F(\d)', str(text)):
        for paths, dirs, files in scandir.walk(tempdir.name):
            for file in files:
                count = 0
                if file.endswith('.jpg'):
                    count+=1
                    if filetype == 'rtf':
                        srcpath = os.path.join(tempdir.name, file)
                    elif filetype == 'docx':
                        srcpath = os.path.join(tempdir.name, 'media', file)

                    destpath = os.path.join('biopsy_images', short_filename+'_'+count+'.jpg')
                    shutil.copy(srcpath, destpath)
                    gt = re.findall('S(\d)A(\d)F(\d)',str(text))
                    gt_df[short_filename] = gt

    tempdir.cleanup()



def main(folder_path,excel_path,gt_csv_path,year):
    # dir_list = os.listdir(folder_path)[:1]

    # df = pd.read_excel(excel_path).set_index('SLIDE NO')
    df = pd.read_excel(excel_path)
    gt_df = pd.read_csv(gt_csv_path).set_index('slide_number')
    for slide_number in tqdm(df.index):
    # for i in tqdm(df.index):
        # slide_number = df.loc[i,"SLIDE NO"]
        # print(slide_number)
        # print(str(slide_number))
        # paths, dirs, files = next(iter(scandir.walk(dir)))
        if str(slide_number).endswith(year):
            # print('2022 file found')
            index = str(slide_number).replace('-','').replace('/','')
            index_lower = index.lower()
            for paths, dirs, files in scandir.walk(folder_path):
            #for (paths, dirs, files) in os.walk(folder):
                for file in files:
                    if index in file or index_lower in file:
                        # with open(os.path.join(paths, file), 'r') as f:
                        filename = os.path.join(paths, file)
                        if filename.endswith('.docx'):
                            img_from_docx(filename, file,gt_df)
                        else:
                            # if open(filename)[:5] == '{\\rtf':
                            if open(input(filename), "r").readlines()[:5] == '{\\rtf':
                                get_imgs_from_document(filename, file, 'rtf',gt_df)
    gt_df.to_csv('ground_truth.csv')

main("D:/HISTO and CYTO REPORT/2022/",'copy_nash.xlsx','ground_truth.csv','22')
main("F:/HISTO AND CYTO REPORTS/2021/",'copy_nash.xlsx','ground_truth.csv','21')