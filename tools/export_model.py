import os
from pathlib import Path
root = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(root / 'yolo-config')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.chdir(root)
from ultralytics import YOLO
import shutil
import subprocess
import sys
model = YOLO(str(root.parent / 'models' / 'yolov8n-seg.pt'))
exported = model.export(format='onnx', imgsz=320, opset=17, dynamic=False,
                        simplify=False, nms=False, device='cpu', batch=1)
destination = root.parent / 'person-seg-320.onnx'
shutil.copy2(exported, destination)
print(destination)
subprocess.run([sys.executable, str(root / 'optimize_model.py')], check=True)

