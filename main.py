import torch
import cv2
from ultralytics import YOLO
import fastapi
from fastapi import FastAPI, Request, File, UploadFile
import numpy as np
import uvicorn
app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = YOLO('runs/detect/train/weights/best.pt')
model = YOLO('model.pt')
#model = YOLO('/home/champion/Documents/.Codes/Python/Sortify/frames/train/runs11/detect/train/weights/model.pt')
model.to(device)
# print(model)

model_outputs = {
    "class_names": '', 
    "bounding_box": ''
}

@app.post('/perform_detections')
async def perform_detections(file: UploadFile = File(...)):
    frame = await file.read()
    nparr = np.frombuffer(frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        # model_outputs['class_names'] = []
        # model_outputs['bounding_box'] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            class_name = model.names[class_ids[i]]
            # model_outputs['bounding_boxes'].append([x1, y1, x2, y2])
            model_outputs['class_names'] = (str(class_name))
            model_outputs['bounding_box'] = (f'{x1}:{y1}:{x2}:{y2}')


    return {'results': model_outputs}

if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=4000)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.resize(frame, (512, 512))

#     results = model(frame)

#     for result in results:
#         boxes = result.boxes.xyxy
#         class_ids= result.boxes.cls
#         confidence = result.boxes.conf
#         for i in range(len(boxes)):
#             x1, y1, x2, y2 = boxes[i].cpu().numpy()
#             class_name = model.names[int(class_ids[i])]
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (233, 0, 0), 2)
#             # cv2.putText(frame, class_name, (int(x1//2)-30, int(y1//2)-30), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 20, 23), 2)
#             cv2.putText(frame, class_name, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 20, 23), 2)

#             #cv2.line(frame, (0, 360), (1100, 360), (0, 255, 0), 2)
            
#     cv2.imshow('frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
