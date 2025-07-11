from fastapi import FastAPI,File,UploadFile
from App.model_helper import predict
app = FastAPI()


@app.get('/')
async def root():
    return {'Running Smooth'}
@app.post("/predict")
async def get_prediction(file : UploadFile = File(...)):
    try:
        print("Debugging chal rhi heavy")
        image_bytes = await file.read()
        image_path = "temp_file.jpg"
        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        prediction = predict(image_path)
        return {'prediction': prediction}
    except Exception as e:
        return {'error': str(e)}