
from ultralytics import YOLO

model=YOLO("dataset/runs/detect/train/weights/best.pt")
classes={'00':0,'01':1, '02':2, '03':3, '04':4, '05':5, '06':6, '07':7, "08":8, "09":9, '10':"A", '11':'B', '12':'D',
    '13':'E', '14':'G', '15':'H', '16':'J', '17':'K', '18':'L', '19':'N', '20':'R', '21':'S', '22':'T', '23':'U', '24':'V', '25':'X',
    '26':'Z'}

results=model.predict("cars/car4.jpg",save=True)