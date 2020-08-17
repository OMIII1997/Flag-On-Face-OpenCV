import cv2


""" Define fucntion to Display any processed image  """
def show(img):
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 
""" Loading cascade classifiers to detect face and Eyes through which we will get coordinates to place flag """
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
leftEyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
rightEyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')


""" Flag image to mask on face """
imgFlag = cv2.imread(r"C:\Master\Learning\OpenCV\Flag_On_Face\flag.png",-1)
imgFlag = cv2.resize(imgFlag,(50,50))

#show(imgFlag)

imgFlag_ = cv2.flip(imgFlag,1) 
#show(imgFlag_)

""" imgFlag_ for Rigth Side of the face and imgFlag for Left side of the face  """


# Creating the mask for the Flag
""" same as above 2 Similar Objects and one with _ is for the rightEye side """

orig_mask = imgFlag[:,:,3] 
orig_mask_ = imgFlag_[:,:,3]

#show(orig_mask)

# Creating the inverted mask for the flag
orig_mask_inv = cv2.bitwise_not(orig_mask)
orig_mask_inv_ = cv2.bitwise_not(orig_mask_)

#show(orig_mask_inv)

# Convert Flag image to BGR
# and save the original image size (used later when re-sizing the image)
imgFlag = imgFlag[:,:,0:3]
imgFlag_ = imgFlag_[:,:,0:3]

#show(imgFlag)

origFlagHeight, origFlagWidth = imgFlag.shape[:2]
origFlagHeight_, origFlagWidth_ = imgFlag_.shape[:2]




#Capturing video from camera

video_capture = cv2.VideoCapture(0)
 
#Looping over the video frames

while True:
    # Reading frames
    ret, frame = video_capture.read()
    """ Flipping frame to get mirror image  """
    frame = cv2.flip(frame,1) 
 
    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
 
    # Detect faces in input video stream
    faces = faceCascade.detectMultiScale(gray,1.4,5)
    """1.4/5 :->Changing these values can improve performace """
 
   # Iterate over each face found
    for (x, y, w, h) in faces:
        """ To see detected faces with bounding box Un-Comment below line """
        # face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)
        
        """ Selecting  ROI (Region Of Interest) """
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
 
        # Detect a leftEye within the region bounded by each face (the ROI)
        
        """ Detected co-ordinates of leftEye eye in 'leftEye' variable """
        
        leftEye = leftEyeCascade.detectMultiScale(roi_gray,1.6,7)  
        """1.6/7 :->Changing these values can improve performace """
        
            
 
        for (nx,ny,nw,nh) in leftEye:
        
            """ To see detected eye Un-Comment below line to see with bounding box"""
            
            #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),1)
            
 
            
                
            flagWidth = nw
            flagHeight = flagWidth * origFlagHeight / origFlagWidth
     
            # Center the flag on the bottom of the leftEye
            x1 = nx - (flagWidth/4)
            x2 = nx + nw + (flagWidth/4)
            y1 = ny + nh - (flagHeight/2)
            y2 = ny + nh + (flagHeight/2)
     
            
            # Re-calculate the width and height of the flag image
            flagWidth = int(x2 - x1)
            flagHeight = int(y2 - y1)
            
             
            # Re-size the original image and the masks to the flag sizes
            # calcualted above
            
            flag = cv2.resize(imgFlag, (flagWidth,flagHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (flagWidth,flagHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (flagWidth,flagHeight), interpolation = cv2.INTER_AREA)
     
            # take ROI for flag from background equal to size of flag image
            #print(y1,y2,x1,x2)
            y1=int(y1+25)
            y2=int(y2+25)
            x1=int(x1+20)
            x2=int(x2+20)
            roi = roi_color[y1:y2, x1:x2]
     
            # roi_bg contains the original image only where the flag is not
            # in the region that is the size of the flag.
            try:
                roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            
     
                # roi_fg contains the image of the flag only where the flag is
                roi_fg = cv2.bitwise_and(flag,flag,mask = mask)
     
                # join the roi_bg and roi_fg
                dst = cv2.add(roi_bg,roi_fg)
     
                # place the joined image, saved to dst back over the original image
                roi_color[y1:y2, x1:x2] = dst
            
            except:
                pass
                
     
            break
        
        
        
        
        
        """ Similar functionality for Right Eye"""
        
        """ Detecting Right eye from face region """
    
        rightEye = rightEyeCascade.detectMultiScale(roi_gray,1.6,7)
        """1.6/7 :->Changing these values can improve performace """
            
 
        for (nx,ny,nw,nh) in rightEye:
            """ To see detected faces Un-Comment below line"""
                 
            #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),1)
            
 
            """ Here we are using flag variable with '_' which is assigned for right eye """
            
            flagWidth_ = nw
            flagHeight_ = flagWidth_ * origFlagHeight_ / origFlagWidth_
     
            # Center the flag on the bottom of the RightEye
            x1 = nx - (flagWidth_/4)
            x2 = nx + nw + (flagWidth_/4)
            y1 = ny + nh - (flagHeight_/2)
            y2 = ny + nh + (flagHeight_/2)
     
            
     
            # Re-calculate the width and height of the flag image
            flagWidth_ = int(x2 - x1)
            flagHeight_ = int(y2 - y1)
            
            
     
            # Re-size the original image and the masks to the flag sizes
            # calcualted above
            flag_ = cv2.resize(imgFlag_, (flagWidth_,flagHeight_), interpolation = cv2.INTER_AREA)
            mask_ = cv2.resize(orig_mask_, (flagWidth_,flagHeight_), interpolation = cv2.INTER_AREA)
            mask_inv_ = cv2.resize(orig_mask_inv_, (flagWidth_,flagHeight_), interpolation = cv2.INTER_AREA)
     
            # take ROI for flag from background equal to size of flag image
            
            y1=int(y1+25)
            y2=int(y2+25)
            x1=int(x1-20)
            x2=int(x2-20)
            roi = roi_color[y1:y2, x1:x2]
     
            # roi_bg contains the original image only where the flag is not
            # in the region that is the size of the flag.
            try:
                roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv_)
            
     
                # roi_fg contains the image of the flag only where the flag is
                roi_fg = cv2.bitwise_and(flag_,flag_,mask = mask_)
     
                # join the roi_bg and roi_fg
                dst = cv2.add(roi_bg,roi_fg)
     
                # place the joined image, saved to dst back over the original image
                roi_color[y1:y2, x1:x2] = dst
            
            except:
                pass
                
     
            break

 
    # Display the resulting frame
    cv2.imshow('Indian Flag Badge', frame)
 
    """ Press 'q' to stop """
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
""" When everything is done, release the capturing function and destroy all cv2 windows """
video_capture.release()
cv2.destroyAllWindows()
