from flask import Flask, request,jsonify, send_file
from flask_cors import CORS
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pdf2docx import Converter
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from torchvision import transforms
import numpy as np
import easyocr
import pandas as pd
from tqdm.auto import tqdm
import openpyxl
import fitz   

app = Flask(__name__)
CORS(app)

model = None
structure_model = None
device = None

def initialize_models():
    global model, structure_model, device
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
    structure_model.to(device)

initialize_models()
def initialize() :
    import layoutparser as lp
    from pdf2image import convert_from_path
    from docx import Document
    import pytesseract
    import openpyxl
    import io
    from torchvision import transforms
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
        return resized_image

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects
@app.route('/process_ pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Define paths for the uploaded PDF and the output Word file
        pdf_path = os.path.join(os.getcwd(), 'uploaded.pdf')
        word_path = os.path.join(os.getcwd(), 'output.docx')
        
        try:
            # Save the uploaded PDF to a temporary file
            file.save(pdf_path)

            # Convert the PDF to Word
            pdf_to_word(pdf_path, word_path)

            # Check if the Word file was created successfully
            if not os.path.exists(word_path):
                return jsonify({'error': 'Word file was not created'}), 500

            # Send the Word document as a response
            return send_file(word_path, as_attachment=True, download_name='converted.docx')

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        finally:
            # Clean up the uploaded PDF and generated Word file after sending the response
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
             

    return jsonify({'error': 'File upload failed'}), 500

def pdf_to_word(pdf_file, word_file):
    """Convert PDF to Word using pdf2docx"""
    cv = Converter(pdf_file)
    cv.convert(word_file, start=0, end=None)
    cv.close()
def objects_to_crops(img, tokens, objects, class_thresholds, padding=50):
    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}
        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
        cropped_img = img.crop(bbox)
        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0] - bbox[0],
                             token['bbox'][1] - bbox[1],
                             token['bbox'][2] - bbox[0],
                             token['bbox'][3] - bbox[1]]

        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens
        table_crops.append(cropped_table)

    return table_crops

def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        row_cells.sort(key=lambda x: x['column'][0])
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    cell_coordinates.sort(key=lambda x: x['row'][1])
    return cell_coordinates

def apply_ocr(cropped_table, cell_coordinates):
    reader = easyocr.Reader(['en'])
    data = {}
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            result = reader.readtext(cell_image)
            if len(result) > 0:
                text = " ".join([x[1] for x in result])
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + [""] * (max_num_columns - len(row_data))
        data[row] = row_data

    return data

def save_as_xlsx(data, sheet_name, workbook):
    worksheet = workbook.create_sheet(title=sheet_name)

    for row, row_text in data.items():
        worksheet.append(row_text)


@app.route('/upload', methods=['POST'])
def upload_file():
    global model, structure_model, device
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        # Create the workbook to store all sheets (tables)
        workbook = openpyxl.Workbook()
        workbook.remove(workbook.active)  # Remove the default sheet

        # Determine if the file is a PDF
        if file.filename.lower().endswith('.pdf'):
            # Read file content into memory
            file_content = file.read()

            # Open the PDF from the bytes
            pdf = fitz.open(stream=file_content, filetype="pdf")
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img = img.convert("RGB")

                # Processing the image (same as before)
                width, height = img.size
                resized_img = img.resize((int(0.6 * width), int(0.6 * height)))

                detection_transform = transforms.Compose([
                    MaxResize(800),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                pixel_values = detection_transform(img).unsqueeze(0)
                pixel_values = pixel_values.to(device)

                with torch.no_grad():
                    outputs = model(pixel_values)

                id2label = model.config.id2label
                id2label[len(model.config.id2label)] = "no object"

                objects = outputs_to_objects(outputs, img.size, id2label)

                tokens = []
                detection_class_thresholds = {
                    "table": 0.5,
                    "table rotated": 0.5,
                    "no object": 10
                }

                tables_crops = objects_to_crops(img, tokens, objects, detection_class_thresholds, padding=50)

                if not tables_crops:
                    continue  # Skip to the next page if no tables are detected

                for idx, table_crop in enumerate(tables_crops):
                    cropped_table = table_crop['image'].convert("RGB")

                    structure_transform = transforms.Compose([
                        MaxResize(1000),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                    structure_pixel_values = structure_transform(cropped_table).unsqueeze(0)
                    structure_pixel_values = structure_pixel_values.to(device)

                    with torch.no_grad():
                        outputs = structure_model(structure_pixel_values)

                    id2label_structure = structure_model.config.id2label
                    id2label_structure[len(structure_model.config.id2label)] = "no object"

                    table_structure = outputs_to_objects(outputs, cropped_table.size, id2label_structure)

                    cell_coordinates = get_cell_coordinates_by_row(table_structure)

                    data = apply_ocr(cropped_table, cell_coordinates)

                    # Save each detected table in a new sheet
                    sheet_name = f"Table_Page{page_num + 1}_Table{idx + 1}"
                    save_as_xlsx(data, sheet_name, workbook)

            pdf.close()

        # Save the workbook
        output_path = os.path.join(os.getcwd(), "tables_detected.xlsx")
        workbook.save(output_path)
        

        return send_file(output_path, as_attachment=True)




@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    file = request.files['file']
    if file and file.filename.endswith('.pdf'):
        
        pdf_path = 'input.pdf'
        file.save(pdf_path)

       
        images = convert_from_path(pdf_path)

    
        layout_model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
        )

        
        output_dir = 'processed_images'
        os.makedirs(output_dir, exist_ok=True)

        
        doc = Document()
 
        for i, image in enumerate(images):
            image = image.convert("RGB")
            layout = layout_model.detect(image)
            
            
            output_image_path = os.path.join(output_dir, f'page_{i+1}.png')
            image.save(output_image_path)
            
             
            doc.add_paragraph(f'Page {i+1}')
            
            for element in layout:
                 
                x1, y1, x2, y2 = element.coordinates
                
               
                cropped_image = image.crop((x1, y1, x2, y2))
              
                
                 
                label = element.type
                
                if label == "Text":
                    extracted_text = pytesseract.image_to_string(cropped_image, lang='eng')
                    doc.add_heading(extracted_text.strip(), level=1)
                
                elif label == "Table":
                    
                    width, height = cropped_image.size
                    resized_img = cropped_image.resize((int(0.6 * width), int(0.6 * height)))

                    detection_transform = transforms.Compose([
                        MaxResize(800),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                    pixel_values = detection_transform(cropped_image).unsqueeze(0)
                    pixel_values = pixel_values.to(device)

                    with torch.no_grad():
                        outputs = table_model(pixel_values)

                    id2label = table_model.config.id2label
                    id2label[len(table_model.config.id2label)] = 'no object'

                    img_objects = outputs_to_objects(outputs, resized_img.size, id2label)

                    for obj in img_objects:
                        cropped_tables = objects_to_crops(cropped_image, [], img_objects, {'table': 0.6, 'table rotated': 0.5})
                        for table in cropped_tables:
                            table_data = structure_model(detection_transform(table["image"]).unsqueeze(0).to(device))

                            table_objects = outputs_to_objects(table_data, table["image"].size, id2label)
                            cell_coordinates = get_cell_coordinates_by_row(table_objects)

                            data = apply_ocr(table["image"], cell_coordinates)
                            

                            
                            rows = list(data.values())
                            table = doc.add_table(rows=len(rows), cols=len(rows[0]))

                            for row_idx, row in enumerate(rows):
                                for col_idx, cell in enumerate(row):
                                    table.cell(row_idx, col_idx).text = cell
                else:
                    doc.add_paragraph(f'{label}:\n{extracted_text.strip()}')

       
        output_doc_path = 'output_processed.docx'
        doc.save(output_doc_path)
        return send_file(output_doc_path, as_attachment=True, download_name='output_processed.docx')
    



if __name__ == '__main__':
    app.run(debug=True)