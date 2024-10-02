import streamlit as st
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import tempfile
import cv2
eye_model=cv2.CascadeClassifier('eye.xml')
drowsymodel=load_model('drowsiness.h5')
st.set_page_config(page_title="Drowsiness Detection System")
st.title("DROWSINESS DETECTION SYSTEM")
choice=st.sidebar.selectbox("My Menu",("Home","Image","Video","Camera"))
if choice=="Home":
    st.header("WELCOME")
    st.write("This is a web application developed by Richa Kaushik as a part of Training Project")
elif choice== "Image":
    file=st.file_uploader("Upload Image")
    if file:
        b=file.getvalue()
        d=np.frombuffer(b,np.uint8)
        img=cv2.imdecode(d,cv2.IMREAD_COLOR)
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        eyes=eye_model.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5,minSize=(24,24))
        for(ex,ey,el,ew) in eyes:
            eye_img=gray_img[ey:ey+ew,ex:ex+el]
            eye_img=cv2.resize(eye_img,(24,24))
            eye_img=eye_img.astype("float")/255.0
            eye_img=img_to_array(eye_img)
            eye_img=np.expand_dims(eye_img,axis=0)
            pred=drowsymodel.predict(eye_img)[0][0]
            if pred<0.8:
                cv2.rectangle(img,(ex,ey),(ex+el,ey+ew),(0,0,255),4)
                cv2.putText(img, 'Close', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)
            else:
                cv2.rectangle(img,(ex,ey),(ex+el,ey+ew),(0,255,0),4)
                cv2.putText(img, 'Open', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)
            st.image(img, channels='BGR', width=400)
elif choice == "Video":
    file = st.file_uploader("Upload Video")
    window = st.empty()
    if file:
        tfile = tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid = cv2.VideoCapture(tfile.name)
        i=1
        while vid.isOpened():
            flag,frame=vid.read()
            if flag:
                gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                eyes=eye_model.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5,minSize=(24,24))
                for(ex,ey,el,ew) in eyes:
                    eye_img=gray_frame[ey:ey+ew,ex:ex+el]
                    cv2.imwrite('temp.jpg',eye_img)
                    eye_img=cv2.resize(eye_img,(24,24))
                    eye_img=eye_img.astype("float")/255.0
                    eye_img=img_to_array(eye_img)
                    eye_img=np.expand_dims(eye_img,axis=0)
                    pred=drowsymodel.predict(eye_img)[0][0]
                    if pred<0.8:
                        cv2.rectangle(frame,(ex,ey),(ex+el,ey+ew),(0,0,255),4)
                        cv2.putText(frame, 'Close', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 4)
                        path="C:/projects/drowsinessd/Scripts/data/"+str(i)+".jpg"
                        cv2.imwrite(path,eye_img)
                        i=i+1
                    else:
                        cv2.rectangle(frame,(ex,ey),(ex+el,ey+ew),(0,255,0),4)
                        cv2.putText(frame, 'Open', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 4)
                    window.image(frame, channels='BGR')
elif choice == "Camera":
    btn = st.button("Start Camera")
    window = st.empty()
    btn2=st.button('Stop Camera')
    if btn2:
        st.session_state.camera_running=False
    if btn:
        st.session_state.camera_running=True
        vid = cv2.VideoCapture(0)
        i=1
        while vid.isOpened():
            flag, frame = vid.read()
            if flag:
                gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                eyes=eye_model.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5,minSize=(24,24))
                for(ex,ey,el,ew) in eyes:
                    eye_img=gray_frame[ey:ey+ew,ex:ex+el]
                    cv2.imwrite('temp.jpg',eye_img)
                    eye_img=cv2.resize(eye_img,(24,24))
                    eye_img=eye_img.astype("float")/255.0
                    eye_img=img_to_array(eye_img)
                    eye_img=np.expand_dims(eye_img,axis=0)
                    pred=drowsymodel.predict(eye_img)[0][0]
                    if pred<0.8:
                        cv2.rectangle(frame,(ex,ey),(ex+el,ey+ew),(0,0,255),4)
                        cv2.putText(frame, 'Close', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 4)
                        path="C:/projects/drowsinessd/Scripts/data/"+str(i)+".jpg"
                        cv2.imwrite(path,eye_img)
                        i=i+1
                    else:
                        cv2.rectangle(frame,(ex,ey),(ex+el,ey+ew),(0,255,0),4)
                        cv2.putText(frame, 'Open', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 4)
                    window.image(frame, channels='BGR')
  
   
        
