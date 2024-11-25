import uuid  
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import firebase_admin
from firebase_admin import credentials, firestore, storage
import time
import threading
from deepface import DeepFace
import requests
from io import BytesIO
from PIL import Image
import traceback
from concurrent.futures import ThreadPoolExecutor
app = Flask(__name__)

TARGET_SIZE = (640, 480)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('paw-senshi-firebase-adminsdk-tgb8t-e8b2ae5418.json') 
firebase_admin.initialize_app(cred, {
    'storageBucket': 'paw-senshi.appspot.com' 
})

db = firestore.client()
bucket = storage.bucket() 

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is running", 200

# Load the two YOLOv8 models once
pet_model = YOLO("petdetection.pt")
activity_model = YOLO("ActivityRecognition.pt")

# Variables to track throughput and processed frames
request_count = 0
processed_frames = 0  # Counter for processed frames
start_time = time.time()
throughput_lock = threading.Lock()

executor = ThreadPoolExecutor(max_workers=10)
@app.route('/capture-frame', methods=['POST'])
def capture_frame():
    try:
        # Check for JSON content type
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.json
        rtsp_url = data.get('rtspUrl')
        owner_id = data.get('userId')  # Fetch owner_id from JSON

        # Validate inputs
        if not rtsp_url:
            return jsonify({"error": "RTSP URL is missing"}), 400
        if not owner_id:
            return jsonify({"error": "Owner ID is missing"}), 400

        # Start capturing from RTSP stream
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open RTSP stream"}), 500

        def process_frame(frame):
            response_data = {'detections': [], 'activities': [], 'matched_pets': [], 'imageUrl': None}
            
            # Resize frame to the target size
            frame_resized = cv2.resize(frame, TARGET_SIZE)

            # Step 1: Pet Detection
            result = pet_model.track(source=frame_resized, persist=True, tracker='botsort.yaml', show=False)
            class_names = []
            if result[0].boxes is not None:
                class_ids = result[0].boxes.cls.cpu().numpy()
                class_names = [result[0].names[int(class_id)] for class_id in class_ids]
            
            response_data['detections'] = class_names

            # Step 2: Check if dog or cat is detected
            if 'dog' in class_names or 'cat' in class_names:
                activity_classes = []

                # Step 2.1: Activity Recognition
                activity_result = activity_model(frame_resized, agnostic_nms=True)[0]
                activity_detections = sv.Detections.from_ultralytics(activity_result)
                activity_classes = activity_detections.data['class_name'].tolist()

                # Filter for relevant activities
                detected_activities = [
                    activity for activity in activity_classes 
                    if activity in ['Falling', 'dermatitis', 'flea_allergy', 'ringworm', 'scabies', 'Leprosy', 'Sitting']
                ]

                response_data['activities'] = detected_activities

                # Step 3: If relevant activity is detected, compute embedding
                if detected_activities:
                    with app.app_context():  # Ensure Firebase and app context are properly initialized
                        pets_query = db.collection('pets').where('ownerId', '==', owner_id).stream()
                        pet_embeddings = {}

                        for doc in pets_query:
                            pet_data = doc.to_dict()
                            if "embedding" in pet_data:
                                pet_embeddings[doc.id] = pet_data["embedding"]

                        if not pet_embeddings:
                            print(f"No pets found for owner_id: {owner_id}")

                        detections = result[0].boxes.xyxy.cpu().numpy()
                        matched_pets = []

                        for box in detections:
                            x1, y1, x2, y2 = map(int, box[:4])
                            cropped_pet = frame_resized[y1:y2, x1:x2]

                            cropped_embedding = DeepFace.represent(cropped_pet, model_name='Facenet', enforce_detection=False)[0]["embedding"]

                            for pet_id, stored_embedding in pet_embeddings.items():
                                norm_cropped = cropped_embedding / np.linalg.norm(cropped_embedding)
                                norm_stored = stored_embedding / np.linalg.norm(stored_embedding)

                                similarity = np.dot(norm_cropped, norm_stored)

                                if similarity > 0.6:  # Threshold for cosine similarity
                                    matched_pets.append(pet_id)
                                    print(f"Matched Pet ID: {pet_id}")
                                    break

                        response_data['matched_pets'] = matched_pets

                        # Save evidence to Firebase for detected activities
                        for activity in detected_activities:
                            activity_ref = db.collection('detected_activities').document(f'{owner_id}_{activity}')
                            activity_doc = activity_ref.get()

                            if activity_doc.exists:
                                activity_ref.update({
                                    'activityName': activity,
                                    'count': firestore.Increment(1),  # Increment count
                                    'timestamp': firestore.SERVER_TIMESTAMP # Append the new timestamp to the array
                                })
                            else:
                                activity_ref.set({
                                    'activityName': activity,
                                    'userId': owner_id,
                                    'count': 1,
                                    'timestamp': firestore.SERVER_TIMESTAMP
                                })

                        # Save image evidence to Firebase Storage
                        pet_name = None
                        for pet_id in matched_pets:
                            pet_doc = db.collection('pets').document(pet_id).get()
                            if pet_doc.exists:
                                pet_name = pet_doc.to_dict().get('name', "unknown_pet")
                                break

                        image_filename = f'inference/{owner_id}/{pet_name}_{activity}_{uuid.uuid4()}.jpg'
                        blob = bucket.blob(image_filename)

                        _, buffer = cv2.imencode('.jpg', frame)
                        blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

                        image_url = blob.public_url
                        activity_ref.set({
                            'imageUrl': image_url,
                            'activityName': activity,
                            'userId': owner_id,
                            'timestamp': firestore.SERVER_TIMESTAMP
                        }, merge=True)

                        response_data['imageUrl'] = image_url

            return response_data

        # Process the first frame in a separate thread
        def capture_and_process():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                cap.release()
                return jsonify({"error": "Failed to capture frame"}), 500

            response_data = process_frame(frame)

            # Return the processed frame response
            return jsonify(response_data), 200

        # Run the frame processing in a separate thread using ThreadPoolExecutor
        future = executor.submit(capture_and_process)
        response_data = future.result()

        cap.release()
        return response_data

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500


# Function to compute embeddings
def compute_embedding(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        img_array = np.array(img)

        embedding = DeepFace.represent(img_array, model_name='Facenet', enforce_detection=False)[0]["embedding"]

        return embedding
    except Exception as e:
        raise ValueError(f"Error in compute_embedding: {str(e)}")

@app.route('/precompute_embeddings', methods=['POST'])
def precompute_embeddings():
    try:
        pets_ref = db.collection('pets')
        pets_docs = pets_ref.stream()

        existing_embeddings = {}
        for doc in pets_docs:
            data = doc.to_dict()
            pet_id = doc.id
            if "embedding" in data:
                existing_embeddings[pet_id] = data["picture"]

        request_data = request.json
        print("Incoming Request Data:", request_data)  # Log the request data to debug
        updated_pets = request_data.get("updated_pets", [])

        for pet in updated_pets:
            pet_id = pet["pet_id"]
            image_url = pet["image_url"]

            if pet_id not in existing_embeddings or existing_embeddings[pet_id] != image_url:
                embedding = compute_embedding(image_url)
                pets_ref.document(pet_id).set({
                    "image_url": image_url,
                    "embedding": embedding
                }, merge=True)

        return jsonify({"message": "Embeddings updated successfully"}), 200

    except Exception as e:
        error_trace = traceback.format_exc()  # Get the full traceback
        print("Error Traceback:", error_trace)  # Print traceback to logs for debugging
        return jsonify({
            "error": str(e),
            "traceback": error_trace
        }), 500

@app.route('/video_feed', methods=['POST'])
def video_feed():
    global request_count, processed_frames

    pet_name = "unknown_pet"
    with throughput_lock:
        request_count += 1
        processed_frames += 1

    request_start_time = time.time()

    try:
        owner_id = request.form.get('owner_id')
        if not owner_id:
            return jsonify({"error": "Owner ID is not received"}), 400
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        frame_resized = cv2.resize(frame, TARGET_SIZE)

        # Step 1: Pet Detection
        result = pet_model.track(source=frame_resized, persist=True, tracker='botsort.yaml', show=False)

        class_names = []
        if result[0].boxes is not None:
            class_ids = result[0].boxes.cls.cpu().numpy()
            class_names = [result[0].names[int(class_id)] for class_id in class_ids]

        response_data = {'detections': class_names}

        # Step 2: Check if dog or cat is detected
        if 'dog' in class_names or 'cat' in class_names:
            activity_classes = []

            # Step 2.1: Activity Recognition
            activity_result = activity_model(frame_resized, agnostic_nms=True)[0]
            activity_detections = sv.Detections.from_ultralytics(activity_result)
            activity_classes = activity_detections.data['class_name'].tolist()

            response_data['activities'] = activity_classes  # Save all detected activities

            # Step 3: If any activity is detected, compute embedding
            if activity_classes:
                pets_query = db.collection('pets').where('ownerId', '==', owner_id).stream()
                pet_embeddings = {}

                for doc in pets_query:
                    pet_data = doc.to_dict()
                    if "embedding" in pet_data:
                        pet_embeddings[doc.id] = pet_data["embedding"]

                if not pet_embeddings:
                    print(f"No pets found for owner_id: {owner_id}")

                detections = result[0].boxes.xyxy.cpu().numpy()
                matched_pets = []

                for box in detections:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cropped_pet = frame_resized[y1:y2, x1:x2]

                    cropped_embedding = DeepFace.represent(cropped_pet, model_name='Facenet', enforce_detection=False)[0]["embedding"]

                    for pet_id, stored_embedding in pet_embeddings.items():
                        norm_cropped = cropped_embedding / np.linalg.norm(cropped_embedding)
                        norm_stored = stored_embedding / np.linalg.norm(stored_embedding)

                        similarity = np.dot(norm_cropped, norm_stored)

                        if similarity > 0.6:  # Threshold for cosine similarity
                            matched_pets.append(pet_id)
                            print(f"Matched Pet ID: {pet_id}")
                            break

                response_data['matched_pets'] = matched_pets

                # Save evidence for all detected activities (no filtering)
                for activity in activity_classes:
                    activity_ref = db.collection('detected_activities').document(f'{owner_id}_{activity}')
                    activity_doc = activity_ref.get()

                    if activity_doc.exists:
                        activity_ref.update({
                            'activityName': activity,
                            'count': firestore.Increment(1),
                            'timestamp': firestore.SERVER_TIMESTAMP
                        })
                    else:
                        activity_ref.set({
                            'activityName': activity,
                            'userId': owner_id,
                            'count': 1,
                            'timestamp': firestore.SERVER_TIMESTAMP
                        })

                    for pet_id in matched_pets:
                        pet_doc = db.collection('pets').document(pet_id).get()
                        if pet_doc.exists:
                            pet_name = pet_doc.to_dict().get('name', "unknown_pet")
                            break

                    # Save image evidence to Firebase Storage
                    image_filename = f'inference/{owner_id}/{pet_name}_{activity}_{uuid.uuid4()}.jpg'
                    blob = bucket.blob(image_filename)

                    _, buffer = cv2.imencode('.jpg', frame)
                    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

                    image_url = blob.public_url

                    activity_ref.set({
                        'imageUrl': image_url,
                        'activityName': activity,
                        'userId': owner_id,
                        'timestamp': firestore.SERVER_TIMESTAMP
                    }, merge=True)

                    response_data['imageUrl'] = image_url

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
