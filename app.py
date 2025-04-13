
import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import uuid
import base64
import tempfile
import os

# Roboflow API Client
ROBOFLOW_API_KEY = "Sev5fKxDoVMkzkzBsAJj"
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

MODEL_A_ID = "tb-stained-images/14"
MODEL_B_ID = "confocal-microscopic-images/8"

# Access the models
project_a = rf.workspace().project(MODEL_A_ID.split('/')[0])
model_a = project_a.version(int(MODEL_A_ID.split('/')[1])).model

project_b = rf.workspace().project(MODEL_B_ID.split('/')[0])
model_b = project_b.version(int(MODEL_B_ID.split('/')[1])).model

# Risk Assessment Logic (remains the same)
def assess_risk(model_a_result, model_b_result, age, is_smoker, is_alcoholic):
    risk_a = model_a_result.lower() if model_a_result else None
    risk_b = model_b_result.lower() if model_b_result else None
    final_risk = "Unknown"
    notes = []

    if risk_a and risk_b:
        if risk_a == "severe" or risk_b == "severe":
            final_risk = "High Risk"
        elif risk_a == "moderate" and risk_b == "moderate":
            final_risk = "Medium Risk"
        elif risk_a == "mild" and risk_b == "mild":
            final_risk = "Low Risk"
        else:
            final_risk = risk_b.capitalize() + " (Mixed Results - Using Model B)"
    elif risk_b:
        final_risk = risk_b.capitalize() + " (Model B Result)"
    elif risk_a:
        final_risk = risk_a.capitalize() + " (Model A Result)"

    if age > 50:
        notes.append("Patient is over 50, which may indicate higher risk.")
        if final_risk == "Medium Risk":
            final_risk = "High Risk (Age Factor)"
        elif final_risk == "Low Risk" and (is_smoker or is_alcoholic):
            final_risk = "Medium Risk (Age + Lifestyle)"
        elif final_risk == "Mild (Mixed Results - Using Model B)" and (is_smoker or is_alcoholic):
            final_risk = "Moderate (Age + Lifestyle)"


    if is_smoker:
        notes.append("Patient is a smoker, which may indicate higher risk.")
        if final_risk == "Low Risk":
            final_risk = "Medium Risk (Smoking Factor)"
        elif final_risk.startswith("Mild") and not final_risk.endswith("Factor)"):
            final_risk = final_risk.replace("Mild", "Moderate")
        elif final_risk == "Medium Risk" and not final_risk.endswith("Factor)"):
            final_risk = "High Risk (Smoking Factor)"

    if is_alcoholic:
        notes.append("Patient consumes alcohol, which may indicate higher risk.")
        if final_risk == "Low Risk":
            final_risk = "Medium Risk (Alcohol Factor)"
        elif final_risk.startswith("Mild") and not final_risk.endswith("Factor)"):
            final_risk = final_risk.replace("Mild", "Moderate")
        elif final_risk == "Medium Risk" and not final_risk.endswith("Factor)"):
            final_risk = "High Risk (Alcohol Factor)"

    return final_risk, ", ".join(notes)

def visualize_results(image_bytes, results, color=(255, 0, 0)):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        if results and results.get('predictions'):
            for prediction in results['predictions']:
                polygon_points = None
                if 'points' in prediction:
                    if isinstance(prediction['points'], list) and all(isinstance(item, str) for item in prediction['points']):
                        try:
                            coords = [int(coord) for p in prediction['points'] for coord in p.split('-')]
                            polygon_points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                        except ValueError:
                            pass # Handle cases where string parsing fails
                    elif isinstance(prediction['points'], list) and all(isinstance(item, dict) for item in prediction['points']) and 'x' in prediction['points'][0] and 'y' in prediction['points'][0]:
                        polygon_points = [(int(p['x']), int(p['y'])) for p in prediction['points']]

                if polygon_points and len(polygon_points) > 2 and 'confidence' in prediction and 'class' in prediction:
                    draw.polygon(polygon_points, outline=color, width=2)
                    text = f"{prediction['class']} ({prediction['confidence']:.2f})"
                    avg_x = sum(p[0] for p in polygon_points) // len(polygon_points) if polygon_points else 0
                    avg_y = sum(p[1] for p in polygon_points) // len(polygon_points) if polygon_points else 0
                    draw.text((avg_x + 5, avg_y + 5), text, fill=color, font=font)
                elif 'x' in prediction and 'y' in prediction and 'width' in prediction and 'height' in prediction and 'confidence' in prediction and 'class' in prediction:
                    x = int(prediction['x'])
                    y = int(prediction['y'])
                    width = int(prediction['width'])
                    height = int(prediction['height'])
                    draw.rectangle([(x, y), (x + width, y + height)], outline=color, width=2)
                    text = f"{prediction['class']} ({prediction['confidence']:.2f})"
                    draw.text((x + 5, y + 5), text, fill=color, font=font)
        return image
    except Exception as e:
        st.error(f"Error visualizing results: {e}")
        return None

def create_report(image_bytes_tb, image_bytes_confocal, results_a, results_b, patient_id, age, is_smoker, is_alcoholic, final_risk, risk_notes):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Risk Assessment Report")
    c.setFont("Helvetica", 12)
    y_position = 730

    def add_line(text, y):
        c.drawString(50, y, text)
        return y - 15

    y_position = add_line(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", y_position)
    y_position = add_line(f"Patient ID: {patient_id}", y_position)
    y_position = add_line(f"Age: {age}", y_position)
    y_position = add_line(f"Smoker: {'Yes' if is_smoker else 'No'}", y_position)
    y_position = add_line(f"Alcoholic: {'Yes' if is_alcoholic else 'No'}", y_position)
    y_position = add_line("Uploaded Images:", y_position - 10)

    image_width = 200
    image_height = 150
    image_y = y_position - image_height - 10
    text_y_offset = 5

    try:
        img_tb_pil = Image.open(io.BytesIO(image_bytes_tb))
        annotated_image_a = visualize_results(image_bytes_tb, results_a)
        if annotated_image_a:
            img_buffer_a = io.BytesIO()
            annotated_image_a.save(img_buffer_a, format="PNG")
            img_reader_a = ImageReader(img_buffer_a)
            c.drawImage(img_reader_a, 50, image_y, width=image_width, height=image_height)
            c.drawString(50, image_y - text_y_offset - 10, "Model A Output")
        else:
            img_tb = ImageReader(io.BytesIO(image_bytes_tb))
            c.drawImage(img_tb, 50, image_y, width=image_width, height=image_height)
            c.drawString(50, image_y - text_y_offset - 10, "Toludine Blue Stained Image")
        y_position -= (image_height + 30)

        img_confocal_pil = Image.open(io.BytesIO(image_bytes_confocal))
        annotated_image_b = visualize_results(image_bytes_confocal, results_b, color=(0, 0, 255))
        if annotated_image_b:
            img_buffer_b = io.BytesIO()
            annotated_image_b.save(img_buffer_b, format="PNG")
            img_reader_b = ImageReader(img_buffer_b)
            c.drawImage(img_reader_b, 300, y_position - image_height - 10, width=image_width, height=image_height)
            c.drawString(300, y_position - image_height - text_y_offset - 20, "Model B Output")
        else:
            img_confocal = ImageReader(io.BytesIO(image_bytes_confocal))
            c.drawImage(img_confocal, 300, y_position - image_height - 10, width=image_width, height=image_height)
            c.drawString(300, y_position - image_height - text_y_offset - 20, "Confocal Image")
        y_position -= (image_height + 40)

    except Exception as e:
        c.drawString(50, y_position - 20, f"Error embedding images: {e}")
        y_position -= 30

    c.setFont("Helvetica", 12)
    y_position = add_line("Model Predictions:", y_position - 10)

    if results_a and results_a.get('predictions'):
        c.drawString(50, y_position, "Model A (TB Stained):")
        y_position -= 15
        for pred in results_a['predictions']:
            class_label = pred.get('class', 'N/A')
            confidence = pred.get('confidence', 0.0)
            y_position = add_line(f"- Class: {class_label}, Confidence: {confidence:.2f}", y_position)
    else:
        y_position = add_line("Model A (TB Stained): No predictions found.", y_position)

    y_position = add_line("", y_position - 5)

    if results_b and results_b.get('predictions'):
        c.drawString(50, y_position, "Model B (Confocal):")
        y_position -= 15
        for pred in results_b['predictions']:
            class_label = pred.get('class', 'N/A')
            confidence = pred.get('confidence', 0.0)
            y_position = add_line(f"- Class: {class_label}, Confidence: {confidence:.2f}", y_position)
    else:
        y_position = add_line("Model B (Confocal): No predictions found.", y_position)

    c.setFont("Helvetica-Bold", 14)
    y_position = add_line(f"Final Risk Level: {final_risk}", y_position - 10)
    c.setFont("Helvetica", 12)
    if risk_notes:
        y_position = add_line("Risk Assessment Notes:", y_position - 10)
        for note in risk_notes.split(", "):
            y_position = add_line(f"- {note}", y_position)

    c.save()
    buffer.seek(0)
    return buffer

st.title("Dysplasia Risk Assessment")

uploaded_file_tb = st.file_uploader("Upload Clinical Toludine Blue stained image", type=["png", "jpg", "jpeg"])
uploaded_file_confocal = st.file_uploader("Upload Confocal image for swapped tissue", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
with col1:
    patient_age = st.number_input("Patient Age", min_value=0, max_value=150, value=50)
with col2:
    is_smoker = st.radio("Smoker?", ["No", "Yes"])
    is_alcoholic = st.radio("Alcoholic?", ["No", "Yes"])

if uploaded_file_tb and uploaded_file_confocal:
    image_bytes_tb = uploaded_file_tb.read()
    image_bytes_confocal = uploaded_file_confocal.read()

    st.subheader("Processing Images...")

    # Save uploaded TB image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file_tb:
        tmp_file_tb.write(image_bytes_tb)
        temp_path_tb = tmp_file_tb.name

    # Run inference on Model A using the temporary file path
    results_a = model_a.predict(tmp_file_tb.name).json()
    st.subheader("Raw Model A JSON:")
    st.json(results_a) # <--- INSPECT THIS OUTPUT

    processed_image_a = visualize_results(image_bytes_tb, results_a, color=(255, 0, 0))
    if processed_image_a:
        st.image(processed_image_a, caption="Model A Output", use_column_width=True)

    # Save uploaded Confocal image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file_confocal:
        tmp_file_confocal.write(image_bytes_confocal)
        temp_path_confocal = tmp_file_confocal.name

    # Run inference on Model B using the temporary file path
    results_b = model_b.predict(tmp_file_confocal.name).json()
    st.subheader("Raw Model B JSON:")
    st.json(results_b) # <--- INSPECT THIS OUTPUT

    processed_image_b = visualize_results(image_bytes_confocal, results_b, color=(0, 0, 255))
    if processed_image_b:
        st.image(processed_image_b, caption="Model B Output", use_column_width=True)

    # Extract classifications
    class_a = results_a.get('predictions', [{}])[0].get('class')
    class_b = results_b.get('predictions', [{}])[0].get('class')

    final_risk, risk_notes = assess_risk(class_a, class_b, patient_age, is_smoker == "Yes", is_alcoholic == "Yes")
    st.subheader(f"Final Risk Level: {final_risk}")
    if risk_notes:
        st.write(f"Notes: {risk_notes}")

    # Generate Report with Annotated Images
    if st.button("Generate and Download Report"):
        patient_id = str(uuid.uuid4())[:8]
        pdf_buffer = create_report(
            image_bytes_tb, image_bytes_confocal, results_a, results_b,
            patient_id, patient_age, is_smoker == "Yes", is_alcoholic == "Yes", final_risk, risk_notes
        )
        b64 = base64.b64encode(pdf_buffer.read()).decode("utf-8")
        href = f'<a href="data:application/pdf;base64,{b64}" download="risk_assessment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Clean up temporary files
    if 'temp_path_tb' in locals() and os.path.exists(temp_path_tb):
        os.remove(temp_path_tb)
    if 'temp_path_confocal' in locals() and os.path.exists(temp_path_confocal):
        os.remove(temp_path_confocal)

else:
    st.info(
        "Please upload both the Toludine Blue stained image and "
        "the Confocal image."
    )
