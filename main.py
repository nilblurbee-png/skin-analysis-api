from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import json
import requests

# Import deepface with error handling
try:
    from deepface import DeepFace
    import cv2
    import numpy as np
    DEEPFACE_AVAILABLE = True
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    print(f"Warning: DeepFace not available. Please install dependencies: {e}")


def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable types to native Python types"""
    if DEEPFACE_AVAILABLE:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
    
    # Handle dict and list recursively
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    
    # Try to convert to float if it's a number-like object
    try:
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
    except (AttributeError, ValueError):
        pass
    
    return obj


app = FastAPI(title="Age, Emotion, and Gender Detection API", version="1.0.0")

# Add CORS middleware to allow requests from React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Age, Emotion, and Gender Detection API",
        "endpoints": {
            "/analyze": "POST - Upload an image to detect age, emotion, and gender",
            "/skin-analysis": "POST - Upload an image for comprehensive skin analysis"
        }
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image and get age, emotion, and gender detection results.
    
    Args:
        file: Image file to analyze (supports common image formats)
    
    Returns:
        JSON response with age, gender, and emotion information
    """
    # Check if DeepFace is available
    if not DEEPFACE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="DeepFace module not available. Please install dependencies: pip install -r requirements.txt"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Create a temporary file to save the uploaded image
    tmp_file_path = None
    try:
        # Read file contents
        contents = await file.read()
        
        # Create temporary file and write contents
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        # Analyze the image
        try:
            results = DeepFace.analyze(
                img_path=tmp_file_path,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False  # Continue even if face detection fails
            )
            
            # Handle both single result and list of results
            if isinstance(results, list):
                result = results[0]
            else:
                result = results
            
            # Extract age (convert to native Python int)
            age = int(result.get("age", 0))
            
            # Extract gender and convert numpy types
            gender_dict = result.get("gender", {})
            gender_dict = convert_to_serializable(gender_dict)
            if gender_dict:
                dominant_gender = max(gender_dict, key=gender_dict.get)
                gender_confidence = float(gender_dict[dominant_gender])
            else:
                dominant_gender = "Unknown"
                gender_confidence = 0.0
            
            # Extract emotion and convert numpy types
            emotion_dict = result.get("emotion", {})
            emotion_dict = convert_to_serializable(emotion_dict)
            if emotion_dict:
                dominant_emotion = max(emotion_dict, key=emotion_dict.get)
                emotion_confidence = float(emotion_dict[dominant_emotion])
            else:
                dominant_emotion = "Unknown"
                emotion_confidence = 0.0
            
            # Prepare response with all values converted to native Python types
            response = {
                "success": True,
                "age": age,
                "gender": {
                    "prediction": dominant_gender,
                    "confidence": round(gender_confidence, 2),
                    "all_predictions": gender_dict
                },
                "emotion": {
                    "prediction": dominant_emotion,
                    "confidence": round(emotion_confidence, 2),
                    "all_predictions": emotion_dict
                }
            }
            
            return JSONResponse(content=response)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing image: {str(e)}"
            )
    
    finally:
        # Clean up temporary file (close it first on Windows)
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                # On Windows, we need to ensure the file is closed before deletion
                import time
                time.sleep(0.1)  # Small delay to ensure file is released
                os.unlink(tmp_file_path)
            except (PermissionError, OSError) as e:
                # If deletion fails, try to delete on next attempt or ignore
                # The OS will clean up temp files eventually
                pass


@app.post("/skin-analysis")
async def skin_analysis(file: UploadFile = File(...)):
    """
    Upload an image and get comprehensive skin analysis results.
    
    Args:
        file: Image file to analyze (supports common image formats)
    
    Returns:
        JSON response with detailed skin analysis information
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Get API key from environment variable
    api_key = os.getenv("AILABAPI_API_KEY", "")
    
    # Create a temporary file to save the uploaded image
    tmp_file_path = None
    try:
        # Read file contents
        contents = await file.read()
        
        # Create temporary file and write contents
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        # Prepare request to ailabapi
        url = "https://www.ailabapi.com/api/portrait/analysis/skin-analysis"
        
        # Open the temporary file for the request
        with open(tmp_file_path, 'rb') as image_file:
            files = {"image": (file.filename or "image.jpg", image_file, file.content_type)}
            headers = {"ailabapi-api-key": api_key}
            
            # Make request to ailabapi
            response = requests.post(url, files=files, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"External API error: {response.text}"
                )
            
            data = response.json()
        
        # Mapping dictionaries
        yes_no_mapping = {0: "No", 1: "Yes"}
        eyelids_mapping = {
            0: "Single eyelids",
            1: "Parallel Double Eyelids",
            2: "Scalloped Double Eyelids"
        }
        skin_type_mapping = {
            0: "Oily skin",
            1: "Dry skin",
            2: "Neutral skin",
            3: "Combination skin"
        }
        
        # Fields that use Yes/No mapping
        yes_no_fields = [
            "pores_left_cheek", "nasolabial_fold", "eye_pouch", "forehead_wrinkle",
            "skin_spot", "acne", "pores_forehead", "pores_jaw", "eye_finelines",
            "dark_circle", "crows_feet", "pores_right_cheek", "blackhead",
            "glabella_wrinkle", "mole"
        ]
        
        # Transform the result data
        if "result" in data:
            result = data["result"]
            
            # Transform Yes/No fields
            for field in yes_no_fields:
                if field in result and "value" in result[field]:
                    result[field]["value_label"] = yes_no_mapping.get(result[field]["value"], "Unknown")
            
            # Transform eyelid fields
            if "left_eyelids" in result and "value" in result["left_eyelids"]:
                result["left_eyelids"]["value_label"] = eyelids_mapping.get(result["left_eyelids"]["value"], "Unknown")
            
            if "right_eyelids" in result and "value" in result["right_eyelids"]:
                result["right_eyelids"]["value_label"] = eyelids_mapping.get(result["right_eyelids"]["value"], "Unknown")
            
            # Transform skin_type
            if "skin_type" in result:
                if "skin_type" in result["skin_type"]:
                    result["skin_type"]["skin_type_label"] = skin_type_mapping.get(result["skin_type"]["skin_type"], "Unknown")
                if "details" in result["skin_type"]:
                    for detail in result["skin_type"]["details"]:
                        if "value" in detail:
                            detail["value_label"] = skin_type_mapping.get(detail["value"], "Unknown")
        
        return JSONResponse(content=data)
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling external API: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                import time
                time.sleep(0.1)
                os.unlink(tmp_file_path)
            except (PermissionError, OSError):
                pass


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}