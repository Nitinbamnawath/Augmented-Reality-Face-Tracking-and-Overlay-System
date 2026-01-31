import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image

mp_face = mp.solutions.face_mesh
mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

sticker = None
mode = "nose" 

cap = cv2.VideoCapture(0)

print("Press 'o' to choose an image.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    
    if res.multi_face_landmarks and sticker is not None:
        for face in res.multi_face_landmarks:
            lm = face.landmark
            
            nx = int(lm[1].x * w)
            ny = int(lm[1].y * h)
            e1x = int(lm[234].x * w)
            e1y = int(lm[234].y * h)
            e2x = int(lm[454].x * w)
            e2y = int(lm[454].y * h)
            
            face_width = np.sqrt((e1x-e2x)**2 + (e1y-e2y)**2)
            
            tx = nx
            ty = ny
            scale = 1.0
            
            if mode == "nose":
                tx = nx
                ty = ny
                scale = face_width * 0.45 
            elif mode == "eyes":
                l_eye_x = int(lm[133].x * w)
                r_eye_x = int(lm[362].x * w)
                l_eye_y = int(lm[133].y * h)
                r_eye_y = int(lm[362].y * h)
                tx = (l_eye_x + r_eye_x) // 2
                ty = (l_eye_y + r_eye_y) // 2
                scale = face_width * 0.8
            elif mode == "forehead":
                top_x = int(lm[10].x * w)
                top_y = int(lm[10].y * h)
                tx = top_x
                ty = int(top_y + face_width * 0.3)
                scale = face_width * 0.9
            elif mode == "head":
                top_x = int(lm[10].x * w)
                top_y = int(lm[10].y * h)
                tx = top_x
                ty = int(top_y - face_width * 0.1) 
                scale = face_width * 1.1
            elif mode == "mouth":
                mx = int(lm[13].x * w)
                my = int(lm[13].y * h)
                tx = mx
                ty = my
                scale = face_width * 0.6
            elif mode == "chin":
                cx = int(lm[152].x * w)
                cy = int(lm[152].y * h)
                tx = cx
                ty = cy
                scale = face_width * 1.2
            elif mode == "cheeks":
                scale = face_width * 0.35
            
            s_h = sticker.shape[0]
            s_w = sticker.shape[1]
            ratio = s_h / s_w
            new_w = int(scale)
            new_h = int(new_w * ratio)
            
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(sticker, (new_w, new_h))
                
                locs = []
                if mode == "cheeks":
                    locs.append((e1x, int(e1y + face_width*0.1)))
                    locs.append((e2x, int(e2y + face_width*0.1)))
                else:
                    locs.append((tx, ty))
                    
                for (px, py) in locs:
                    y1 = int(py - new_h/2)
                    y2 = int(py + new_h/2)
                    x1 = int(px - new_w/2)
                    x2 = int(px + new_w/2)
                    
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 > w: x2 = w
                    if y2 > h: y2 = h
                    
                    if x2 > x1 and y2 > y1:
                        orig_y1 = int(py - new_h/2)
                        orig_x1 = int(px - new_w/2)
                        
                        off_y = 0
                        off_x = 0
                        
                        if orig_y1 < 0: off_y = -orig_y1
                        if orig_x1 < 0: off_x = -orig_x1
                        
                        sticker_crop = resized[off_y : off_y + (y2-y1), off_x : off_x + (x2-x1)]
                        
                        alpha = sticker_crop[:,:,3] / 255.0
                        inv = 1.0 - alpha
                        bg = frame[y1:y2, x1:x2]
                        for c in range(3):
                            bg[:,:,c] = (alpha * sticker_crop[:,:,c] + inv * bg[:,:,c])
                        frame[y1:y2, x1:x2] = bg

    cv2.putText(frame, "Mode: " + mode, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Final", frame)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
        
    if k == ord('o'):
        print("Please select an image")
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename()
        root.destroy()
        
        if path:
            print("Processing file:", path)
            pil = Image.open(path).convert("RGB")
            arr = np.array(pil)
            img_rgb = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
            mask = np.zeros(img_rgb.shape[:2], np.uint8)
            
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            h_g, w_g = img_rgb.shape[:2]
            rect = (1, 1, w_g-2, h_g-2)
            
            cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            
            b, g, r = cv2.split(img_rgb)
            a = mask2 * 255 
            
            sticker = cv2.merge([b, g, r, a])
            print("Done")
            
    if k == ord('1'): mode = "head"
    if k == ord('2'): mode = "forehead"
    if k == ord('3'): mode = "eyes"
    if k == ord('4'): mode = "nose"
    if k == ord('5'): mode = "mouth"
    if k == ord('6'): mode = "chin"
    if k == ord('7'): mode = "cheeks"

cap.release()
cv2.destroyAllWindows()
