from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify, abort
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from line_segmentation import segment_lines
import cv2
import numpy as np
from models import db, Document

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///documents.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Initialize TrOCR processor and model
model_id_handwritten = "microsoft/trocr-base-handwritten"
processor_handwritten = TrOCRProcessor.from_pretrained(model_id_handwritten)
model_handwritten = VisionEncoderDecoderModel.from_pretrained(model_id_handwritten)

with app.app_context():
    db.create_all()

def ocr_handwritten(image, processor, model):

    # Perform OCR
    pixel_values = processor(image, return_tensors='pt').pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

@app.route('/')
def index():
    documents = Document.query.order_by(Document.created_at.desc()).all()
    return render_template('index.html', documents=documents)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        nparr = np.frombuffer(file.read(), np.uint8)
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        segmented_lines = segment_lines(image_cv2)

        recognized_texts = []
        for i, line in enumerate(segmented_lines):
            # Convert segmented line to PIL image
            line_pil = Image.fromarray(line)
            recognized_text = ocr_handwritten(line_pil, processor_handwritten, model_handwritten)
            recognized_texts.append(recognized_text)
            print(f"Line {i+1} - Recognized Text:", recognized_text)

        # Save recognized text as a new Document record
        full_text = '\n'.join(recognized_texts)
        new_doc = Document(text=full_text)
        db.session.add(new_doc)
        db.session.commit()

        return redirect(url_for('index'))

@app.route('/download/<int:doc_id>', methods=['GET'])
def download(doc_id):
    document = Document.query.get_or_404(doc_id)
    text = document.text

    # Create a text file with extracted text
    filename = f"document_{doc_id}.txt"
    with open(filename, "w") as file:
        file.write(text)

    # Send the text file to the user for download
    return send_file(filename, as_attachment=True)

# REST API endpoints for CRUD operations

@app.route('/api/documents', methods=['GET'])
def api_get_documents():
    documents = Document.query.order_by(Document.created_at.desc()).all()
    return jsonify([doc.to_dict() for doc in documents])

@app.route('/api/documents/<int:doc_id>', methods=['GET'])
def api_get_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    return jsonify(document.to_dict())

@app.route('/api/documents', methods=['POST'])
def api_create_document():
    data = request.get_json()
    if not data or 'text' not in data:
        abort(400, description="Missing 'text' in request data")
    new_doc = Document(text=data['text'])
    db.session.add(new_doc)
    db.session.commit()
    return jsonify(new_doc.to_dict()), 201

@app.route('/api/documents/<int:doc_id>', methods=['PUT'])
def api_update_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    data = request.get_json()
    if not data or 'text' not in data:
        abort(400, description="Missing 'text' in request data")
    document.text = data['text']
    db.session.commit()
    return jsonify(document.to_dict())

@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def api_delete_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    db.session.delete(document)
    db.session.commit()
    return '', 204

# UI routes for CRUD operations

@app.route('/documents')
def list_documents():
    documents = Document.query.order_by(Document.created_at.desc()).all()
    return render_template('list.html', documents=documents)

@app.route('/documents/new', methods=['GET', 'POST'])
def create_document():
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            new_doc = Document(text=text)
            db.session.add(new_doc)
            db.session.commit()
            return redirect(url_for('list_documents'))
    return render_template('form.html', action="Create", document=None)

@app.route('/documents/<int:doc_id>/edit', methods=['GET', 'POST'])
def edit_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            document.text = text
            db.session.commit()
            return redirect(url_for('list_documents'))
    return render_template('form.html', action="Edit", document=document)

@app.route('/documents/<int:doc_id>/delete', methods=['POST'])
def delete_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    db.session.delete(document)
    db.session.commit()
    return redirect(url_for('list_documents'))

if __name__ == '__main__':
    app.run(debug=True)
