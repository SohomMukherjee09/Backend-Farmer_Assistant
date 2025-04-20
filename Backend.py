import os
import logging
import pickle
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Form, Header
from starlette.responses import FileResponse  # Correct import for FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict
from io import BytesIO
import io
import uvicorn
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langdetect import detect
from langchain.schema.runnable import RunnableSequence
from deep_translator import GoogleTranslator
import pandas as pd
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from geopy.geocoders import Nominatim
import google.generativeai as genai
from meteostat import Point, Daily
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth
import base64
from gtts import gTTS
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import uuid

# Logging setup
LOG_DIR = "logs"
LOG_FILE_NAME = "application.log"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, LOG_FILE_NAME)
load_dotenv()

logging.basicConfig(
    filename=log_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Log environment variables for debugging
logger.info(f"DEVELOPMENT_MODE: {os.getenv('DEVELOPMENT_MODE')}")
logger.info(f"GOOGLE_API_KEY: {'set' if os.getenv('GOOGLE_API_KEY') else 'not set'}")
logger.info(f"OPENWEATHER_API_KEY: {'set' if os.getenv('OPENWEATHER_API_KEY') else 'not set'}")
logger.info(f"PORT: {os.getenv('PORT', '8000')}")

# Initialize Firebase Admin
firebase_creds = os.getenv("FIREBASE_CREDENTIALS")
if firebase_creds:
    try:
        cred = credentials.Certificate(json.loads(firebase_creds))
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin: {str(e)}")
else:
    logger.warning("Firebase credentials not found, running without Firebase authentication")

security = HTTPBearer()

# Mock token verification for development
async def verify_token(authorization: Optional[str] = Header(None)):
    if os.getenv("DEVELOPMENT_MODE") == "true":
        logger.info("Development mode: Bypassing token verification")
        return {"uid": "mock-user-id"}
    if not authorization:
        logger.error("No Authorization header provided")
        raise HTTPException(status_code=422, detail="No Authorization header provided")
    try:
        if authorization.startswith("Bearer "):
            id_token = authorization.split(" ")[1]
            if id_token == "mock-token":
                logger.info("Accepted mock token for development")
                return {"uid": "mock-user-id"}
            decoded_token = auth.verify_id_token(id_token)
            logger.info(f"Token verified for user: {decoded_token.get('uid')}")
            return decoded_token
        else:
            logger.error("Invalid Authorization header format")
            raise HTTPException(status_code=422, detail="Invalid Authorization header format")
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Token verification failed: {str(e)}")

app = FastAPI()

# Updated CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        # Add your production frontend domain here, e.g., "https://your-frontend-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Agricultural AI API"}

# Google Generative AI model setup
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip('"')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip('"')
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
chat_history = []
translator = GoogleTranslator(source='auto', target='en')

# TensorFlow Model Setup
model_path = "Models/model.pkl"  # Adjust path as needed
try:
    with open(model_path, "rb") as f:
        tf_model = pickle.load(f)
    logger.info(f"Pickled TensorFlow model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Failed to load TensorFlow model: {str(e)}")
    raise Exception(f"Failed to load TensorFlow model: {str(e)}")

labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load plant disease JSON
try:
    with open("plant_disease.json", 'r') as file:
        plant_disease = json.load(file)
    logger.info("Plant disease JSON loaded successfully")
except Exception as e:
    logger.error(f"Failed to load plant_disease.json: {str(e)}")
    raise Exception(f"Failed to load plant_disease.json: {str(e)}")

# Ensure upload and temp audio directories exist
os.makedirs("uploadimages", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# In-memory history store
user_history = defaultdict(list)  # {user_id: [history_entries]}

PROMPT_TEMPLATES = {
    'en': ChatPromptTemplate.from_template(
        "You are an agricultural expert. Here is the chat history:\n{chat_history}\n"
        "User: {query}\nAI:"
    ),
    'hi': ChatPromptTemplate.from_template(
        "आप एक कृषि विशेषज्ञ हैं। यहाँ पिछले वार्तालाप का इतिहास है:\n{chat_history}\n"
        "उपयोगकर्ता: {query}\nएआई:"
    ),
    'bn': ChatPromptTemplate.from_template(
        "আপনি একজন কৃষি বিশেষজ্ঞ। এখানে আগের কথোপকথনের ইতিহাস:\n{chat_history}\n"
        "ব্যবহারকারী: {query}\nএআই:"
    ),
}

def detect_language(text: str) -> str:
    try:
        lang = translator.translate(text, target='en').detected_source_language
        if lang in ['hi', 'en', 'bn']:
            return lang
    except:
        pass
    try:
        return detect(text)
    except:
        return 'en'

def get_prompt_template(language: str):
    return PROMPT_TEMPLATES.get(language, PROMPT_TEMPLATES['en'])

def clean_text(text: str) -> str:
    text = text.replace('*', '')
    text = re.sub(r'\s*>\s*', '> ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'(\d+\.)(\S)', r'\1 \2', text)
    return text.strip()

# Data models
class TextRequest(BaseModel):
    text: Optional[str] = ""
    userInput: Optional[str] = ""

    def get_input(self):
        return self.text or self.userInput or ""

class SoilWeatherRequest(BaseModel):
    N: int
    P: int
    K: int
    pH: float
    city: str

class CityOnlyRequest(BaseModel):
    city: str

# AI Response Generation
@app.post("/get-response")
async def get_response(request: TextRequest):
    logger.info(f"Received request: {request.dict()}")
    user_id = 'dev-user' if os.getenv("DEVELOPMENT_MODE") == "true" else 'unknown'
    logger.info(f"Processing request for user_id: {user_id}")
    try:
        input_text = request.get_input()
        if not input_text:
            logger.error("No input text provided")
            raise HTTPException(status_code=400, detail="No input text provided")

        language = detect_language(input_text)
        prompt = get_prompt_template(language)
        logger.info(f"Detected language: {language}")

        chat_history.append(f"User: {input_text}")
        if len(chat_history) > 10:
            chat_history.pop(0)

        full_chat_history = "\n".join(chat_history)
        conversation_chain = RunnableSequence(prompt | llm)
        response = conversation_chain.invoke({
            "chat_history": full_chat_history,
            "query": input_text
        })

        formatted_response = clean_text(response.content)
        chat_history.append(f"AI: {formatted_response}")
        return {
            "response": formatted_response,
            "language": language,
        }

    except HTTPException as he:
        logger.error(f"HTTP error: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"AI response generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI response generation failed: {str(e)}")

# Forecast
def get_5_day_forecast(city, api_key):
    URL = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(URL)
    daily_data = defaultdict(lambda: {'temps': [], 'humidity': [], 'descriptions': []})

    if response.status_code == 200:
        data = response.json()
        for forecast in data['list']:
            date = forecast['dt_txt'].split()[0]
            daily_data[date]['temps'].append(forecast['main']['temp'])
            daily_data[date]['humidity'].append(forecast['main']['humidity'])
            daily_data[date]['descriptions'].append(forecast['weather'][0]['description'])

        forecast_summary = ""
        for i, (date, values) in enumerate(daily_data.items()):
            if i == 5:
                break
            avg_temp = sum(values['temps']) / len(values['temps'])
            avg_humidity = sum(values['humidity']) / len(values['humidity'])
            desc = max(set(values['descriptions']), key=values['descriptions'].count)
            forecast_summary += f"{date}: {avg_temp:.1f}°C, {avg_humidity:.0f}% humidity, {desc}\n"

        return forecast_summary.strip()
    else:
        return "Weather API error."

# Historical data
def get_historical_weather(location):
    start = datetime.now() - timedelta(days=730)
    end = datetime.now()
    point = Point(location.latitude, location.longitude)
    data = Daily(point, start, end).fetch()
    avg_temp = round(data['tavg'].mean(), 1)
    avg_precip = round(data['prcp'].mean(), 2)
    return f"📅 Historical Summary (Last 2 Years)\nAverage Temp: {avg_temp}°C\nAverage Precipitation: {avg_precip} mm/day"

def create_chunks(df):
    return [", ".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()]

def retrieve_relevant_chunks(chunks, query, top_n=5):
    return sorted(chunks, key=lambda x: sum(word.lower() in x.lower() for word in query.split()), reverse=True)[:top_n]

@app.post("/crop-suggestion")
async def suggest_crop(req: SoilWeatherRequest):
    try:
        if not req.city:
            raise HTTPException(status_code=400, detail="City is required")
        if None in [req.N, req.P, req.K, req.pH]:
            raise HTTPException(status_code=400, detail="All soil parameters (N, P, K, pH) are required")

        file_path = "Crop_dataset.csv"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Crop dataset file not found")

        df = pd.read_csv(file_path)

        geolocator = Nominatim(user_agent="weather_app", timeout=10)
        try:
            location = geolocator.geocode(req.city)
            if not location:
                raise HTTPException(status_code=400, detail=f"Could not find location for city: {req.city}")
        except Exception as geocode_error:
            logger.error(f"Geocoding failed for {req.city}: {str(geocode_error)}")
            raise HTTPException(status_code=400, detail="Location service unavailable")

        try:
            forecast = get_5_day_forecast(req.city, api_key=OPENWEATHER_API_KEY)
            if forecast == "Weather API error.":
                forecast = "Weather forecast unavailable"
        except Exception as weather_error:
            logger.error(f"Weather API failed: {str(weather_error)}")
            forecast = "Weather forecast unavailable"

        historical = "Historical data unavailable"
        try:
            historical = get_historical_weather(location)
        except Exception as hist_error:
            logger.error(f"Historical weather failed: {str(hist_error)}")

        query = f"N: {req.N}, P: {req.P}, K: {req.K}, pH: {req.pH}, Forecast: {forecast}, Historical: {historical}"
        context = "\n".join(retrieve_relevant_chunks(create_chunks(df), query))

        prompt = f"""You are an expert agricultural scientist providing crop recommendations.
        Analyze the following data and provide detailed, scientifically-valid crop suggestions:

        SOIL ANALYSIS:
        - Nitrogen (N): {req.N} ppm
        - Phosphorus (P): {req.P} ppm
        - Potassium (K): {req.K} ppm
        - pH Level: {req.pH}

        LOCATION: {req.city}

        WEATHER CONDITIONS:
        5-Day Forecast:
        {forecast}

        Historical Weather Patterns:
        {historical}

        RELEVANT CROP DATA:
        {context}

    GUIDELINES FOR RESPONSE:
    1. Recommend 3-5 most suitable crops based on the soil parameters and climate
    2. For each recommended crop:
   - List optimal growing conditions
   - Mention any special requirements
   - Provide planting season advice
    3. Include 1-2 alternative crops that could work with minor soil amendments
    4. Highlight any potential challenges based on current conditions
    5. Provide brief cultivation tips for each recommended crop
    6. Format the response clearly with sections and bullet points

OUTPUT FORMAT:
### Recommended Crops for {req.city}

#### [Crop 1 Name]
- Suitability: [Excellent/Good/Fair] match for current conditions
- Optimal Conditions: [Ideal soil pH, temperature range, etc.]
- Planting Window: [Best planting times]
- Key Requirements: [Water needs, sunlight, etc.]
- Yield Potential: [Expected yield under these conditions]
- Management Tips: [Specific cultivation advice]

#### [Crop 2 Name]
[...]

### Alternative Options
[List crops that could work with minor adjustments]

### Important Considerations
[Any warnings about pests, diseases, or weather risks]

Note: All recommendations should be practical, scientifically valid, and tailored to the specific location and conditions provided.
"""
        try:
            response = model.generate_content(prompt)
            return {
                "suggested_crops": response.text.strip(),
                "city": req.city
            }
        except Exception as ai_error:
            logger.error(f"AI generation failed: {str(ai_error)}")
            raise HTTPException(status_code=500, detail="AI service temporarily unavailable")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in crop suggestion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.post("/forecast")
async def get_forecast_only(req: CityOnlyRequest):
    try:
        forecast = get_5_day_forecast(req.city, api_key=OPENWEATHER_API_KEY)
        return {"forecast": forecast}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Weather forecast failed")

# Simple test endpoint
@app.post("/test")
async def test_endpoint(request: dict = None):
    if request is None:
        request = {}
    logger.info(f"Test endpoint called with request: {request}")
    return {"message": "Test successful", "received": request}

# Simple response endpoint
@app.post("/simple-response")
async def simple_response(request: dict = None):
    if request is None:
        request = {}
    logger.info(f"Simple response requested with input: {request}")
    return {
        "response": "This is a fixed response from the server. Growing tomatoes requires well-drained soil, regular watering, and plenty of sunlight. Start by planting seedlings after the last frost, and provide support as they grow. Water at the base of the plant to prevent leaf diseases.",
        "language": "en"
    }

@app.post("/classify")
async def classify_image(image: UploadFile = File(...), language: str = Form(...)):
    try:
        image_data = await image.read()
        image = Image.open(io.BytesIO(image_data))
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        prompt = """
You are a plant pathologist AI. Based on the image provided, classify the plant disease in the image.

Please provide the following information in a structured format:
1. Disease Name: The name of the disease (if any).
2. Affected Plant Part: Which part seems to be affected (leaf, stem, fruit, etc.).
3. Detailed Description: Visual symptoms and explanation of how the disease looks.
4. Cause: What causes this disease (bacteria, fungus, virus, etc.).
5. Severity Level: Mild, moderate, or severe.
6. Recommended Treatment: Suggested pesticides or organic treatments.
7. Preventive Measure: How to prevent it in the future.
8. Impact on Crop Yield: How this disease may affect overall productivity.

Be detailed and use layman-friendly language so that even a farmer with no technical background can understand and act upon it. Dont use * IN THE FORMAT NO NEED TO MAKE IT BOLD
"""
        response = model.generate_content([
            {"text": prompt},
            {"mime_type": "image/png", "data": image_bytes.read()}
        ])
        diagnosis_text = response.text

        lang_map = {"English": "en", "Hindi": "hi", "Bengali": "bn"}
        lang_code = lang_map.get(language, "en")
        if language != "English":
            translator = GoogleTranslator(source='auto', target=lang_code)
            diagnosis_text = translator.translate(diagnosis_text)

        voice_text = diagnosis_text[:100]
        tts = gTTS(text=voice_text, lang=lang_code)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")

        return {"diagnosis": diagnosis_text, "audio": audio_base64}

    except Exception as e:
        logger.error(f"Image classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image classification failed: {str(e)}")

@app.post("/upload-image")
async def upload_image(
    image: UploadFile = File(...),
    language: str = Form("English"),
    decoded_token: dict = Depends(verify_token)
):
    try:
        user_id = decoded_token.get("uid", "mock-user-id")
        logger.info(f"Processing image upload for user_id: {user_id}")

        # Validate image
        if not image:
            logger.error("No image provided")
            raise HTTPException(status_code=400, detail="No image provided")

        # Save image
        image_data = await image.read()
        image_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        image_path = os.path.join("uploadimages", image_filename)
        try:
            with open(image_path, "wb") as f:
                f.write(image_data)
            if not os.path.exists(image_path):
                raise FileNotFoundError("Image save failed")
        except Exception as e:
            logger.error(f"Failed to save image to {image_path}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

        # TensorFlow prediction
        def extract_features(image_path):
            image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
            feature = tf.keras.utils.img_to_array(image)
            feature = np.array([feature])
            return feature

        img = extract_features(image_path)
        prediction = tf_model.predict(img)
        predicted_index = prediction.argmax()
        predicted_label = labels[predicted_index]
        try:
            disease_info = plant_disease[predicted_index]
        except IndexError:
            disease_info = {"name": predicted_label}
        logger.info(f"TensorFlow prediction: {predicted_label}")

        # Gemini API diagnosis
        image = Image.open(image_path)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        prompt = """
You are a plant pathologist AI. The image provided has been classified as '{}' by a TensorFlow model.

DO NOT SHOW THE RESULT OF THE COMPARISONS BETWEEN TENSORFLOW AND YOUR RESPONSE ONLY THE STRUCTURED FORMAT
Provide the following in a structured format:
1. Disease Name: Confirm or refine the disease name.
2. Affected Plant Part: Which part is affected (leaf, stem, fruit, etc.).
3. Detailed Description: Visual symptoms and explanation.
4. Cause: What causes this disease (bacteria, fungus, virus, etc.).
5. Severity Level: Mild, moderate, or severe.
6. Recommended Treatment: Suggested pesticides or organic treatments.
7. Preventive Measures: How to prevent it in the future.
8. Impact on Crop Yield: How this disease affects productivity.

Use layman-friendly language for farmers with no technical background.
under each point DO NOT PROVIDE FURTHER POINTS GIVE PARAGRAPHS INSTEAD
""".format(predicted_label)

        try:
            response = model.generate_content([
                {"text": prompt},
                {"mime_type": "image/png", "data": image_bytes.read()}
            ])
            diagnosis_text = response.text
            match = re.search(r'1\.\s*Disease Name.*', diagnosis_text, re.DOTALL)
            if not match:
                logger.warning(f"Structured format not found for {image_path}")
                diagnosis_text = (
                    "1. Disease Name: Unknown\n"
                    "2. Affected Plant Part: Unknown\n"
                    "3. Detailed Description: Unable to identify disease from image.\n"
                    "4. Cause: Unknown\n"
                    "5. Severity Level: Unknown\n"
                    "6. Recommended Treatment: Consult a local expert.\n"
                    "7. Preventive Measures: Maintain plant health.\n"
                    "8. Impact on Crop Yield: Unknown"
                )
            else:
                diagnosis_text = match.group(0)
        except Exception as e:
            logger.error(f"Gemini diagnosis failed: {str(e)}")
            diagnosis_text = f"Error fetching diagnosis: {str(e)}"

        # Language selection and translation
        lang_map = {"English": "en", "Hindi": "hi", "Bengali": "bn"}
        lang_code = lang_map.get(language, "en")
        if language != "English":
            translator = GoogleTranslator(source='auto', target=lang_code)
            translated_text = translator.translate(diagnosis_text)
        else:
            translated_text = diagnosis_text

        # Generate audio
        voice_text = translated_text[:100]
        tts = gTTS(text=voice_text, lang=lang_code)
        audio_filename = f"diagnosis_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join("temp_audio", audio_filename)
        try:
            tts.save(audio_path)
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            audio_base64 = None

        # Update history
        history_entry = {
            "image_path": image_path,
            "predicted_label": predicted_label,
            "diagnosis": translated_text,
            "language": language,
            "audio_path": f"/temp_audio/{audio_filename}" if audio_base64 else None
        }
        user_history[user_id].insert(0, history_entry)
        user_history[user_id] = user_history[user_id][:5]  # Limit to 5 entries

        return {
            "result": True,
            "image_path": image_path,
            "prediction": predicted_label,
            "diagnosis": translated_text,
            "audio": audio_base64,
            "history": user_history[user_id]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in image upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Endpoint to clear history
@app.post("/clear-history")
async def clear_history(decoded_token: dict = Depends(verify_token)):
    user_id = decoded_token.get("uid", "mock-user-id")
    user_history[user_id] = []
    logger.info(f"Cleared history for user_id: {user_id}")
    return {"message": "History cleared"}

# Serve uploaded images
@app.get("/uploadimages/{filename}")
async def get_uploaded_image(filename: str):
    file_path = os.path.join("uploadimages", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

# Serve audio files
@app.get("/temp_audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join("temp_audio", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Uvicorn on host=0.0.0.0, port={port}")
    uvicorn.run("Backend:app", host="0.0.0.0", port=port, reload=os.getenv("DEVELOPMENT_MODE") == "true")
