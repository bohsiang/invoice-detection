
##load model

import cv2
import numpy as np
import re
import threading
from multiprocessing import Queue    #使用多核心的模組 Queue
import logging
import time
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cfg
from network import East
from preprocess import resize_image
from nms import nms
import os


from config import *
from apphelper.image import union_rbox,adjust_box_to_origin,base64_to_PIL
from application import trainTicket,idcard 
if yoloTextFlag =='keras' or AngleModelFlag=='tf' or ocrFlag=='keras':
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf
        from keras import backend as K
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.3 ## GPU最大占用量
        config.gpu_options.allow_growth = True ##GPU是否可动态增加
        K.set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())
    
    else:
      ##CPU启动
      os.environ["CUDA_VISIBLE_DEVICES"] = ''

if yoloTextFlag=='opencv':
    scale,maxScale = IMGSIZE
    from text.opencv_dnn_detect import text_detect
elif yoloTextFlag=='darknet':
    scale,maxScale = IMGSIZE
    from text.darknet_detect import text_detect
elif yoloTextFlag=='keras':
    scale,maxScale = IMGSIZE[0],2048
    from text.keras_detect import  text_detect
else:
     print( "err,text engine in keras\opencv\darknet")
     
from text.opencv_dnn_detect import angle_detect

if ocr_redis:
    ##多任务并发识别
    from apphelper.redisbase import redisDataBase
    ocr = redisDataBase().put_values
else:   
    from crnn.keys import alphabetChinese,alphabetEnglish
    if ocrFlag=='keras':
        from crnn.network_keras import CRNN
        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelKerasLstm
            else:
                ocrModel = ocrModelKerasDense
        else:
            ocrModel = ocrModelKerasEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
            
    elif ocrFlag=='torch':
        from crnn.network_torch import CRNN
        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelTorchLstm
            else:
                ocrModel = ocrModelTorchDense
                
        else:
            ocrModel = ocrModelTorchEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
    elif ocrFlag=='opencv':
        from crnn.network_dnn import CRNN
        ocrModel = ocrModelOpencv
        alphabet = alphabetChinese
    else:
        print( "err,ocr engine in keras\opencv\darknet")
     
    nclass = len(alphabet)+1   
    if ocrFlag=='opencv':
        crnn = CRNN(alphabet=alphabet)
    else:
        crnn = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
    if os.path.exists(ocrModel):
        crnn.load_weights(ocrModel)
    else:
        print("download model or tranform model with tools!")
        
    ocr = crnn.predict_job
    
   
from main import TextOcrModel

model =  TextOcrModel(ocr,text_detect,angle_detect)




#def predict(img):          
        #image = cv2.putText(img,digital_num,(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),10)
        #cv2.imshow('frame2', image)
        
def t2_start(q,img) :
    thread2 = threading.Thread(target=T2_job, args=(q,img))
    thread2.start() 
    
     
def T2_job(q,img):
    print('T2 start')
    t = time.time()

    if img is not None:
        img = np.array(img)
                
    #H,W = img.shape[:2]
    
    ## predict 
    
    result,angle= model.model(img,
    scale=scale,
    maxScale=maxScale,
    detectAngle=True,##是否进行文字方向检测，通过web传参控制
    MAX_HORIZONTAL_GAP=100,##字符之间的最大间隔，用于文本行的合并
    MIN_V_OVERLAPS=0.6,
    MIN_SIZE_SIM=0.6,
    TEXT_PROPOSALS_MIN_SCORE=0.1,
    TEXT_PROPOSALS_NMS_THRESH=0.3,
    TEXT_LINE_NMS_THRESH = 0.99,##文本行之间测iou值
    LINE_MIN_SCORE=0.1,
    leftAdjustAlph=0.01,##对检测的文本行进行向左延伸
    rightAdjustAlph=0.01,##对检测的文本行进行向右延伸
    )
    
    
    
    p = re.compile(r'^[A-z]+[0-9]*$')
    result = union_rbox(result,0.2)
    
    print(result)
    
    res = [{'text':x['text'],
    'name':str(i),
    'box':{'cx':x['cx'],
    'cy':x['cy'],
    'w':x['w'],
    'h':x['h'],
    'angle':x['degree']}
    } for i,x in enumerate(result) if re.match(p,x['text'])]
    
    
    #print("res:" + str(res))
    logging.basicConfig(filename='logging.txt',
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    
    logging.info("res:" + str(res))

    
    if res != []:
        digital_re = re.compile(r'\d{8}')
        digital_num =  re.findall(digital_re,res[0]['text'])
        q.put(digital_num)
    #pre_enable=1
    print('T2 finish')
    timeTake = time.time()-t
    logging.info("Execution time: " + str(timeTake))




east = East()
east_detect = east.east_network()
east_detect.summary()
east_detect.load_weights(cfg.model_weights_path)


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def crop_rectangle(img, geo):
    rect = cv2.minAreaRect(geo.astype(int))
    center, size, angle = rect[0], rect[1], rect[2]
    print(angle)
    if(angle > -45):
        center = tuple(map(int, center))
        size = tuple([int(rect[1][0] + 100), int(rect[1][1] + 100)])
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1] + 100), int(rect[1][0]) + 100])
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop


def predict(east_detect, img_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.image_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_height = resize_image(im, cfg.image_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        quad_draw = ImageDraw.Draw(quad_im)
        txt_items = []
        flag = False
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):
            if np.amin(score) > 0:
                flag = True
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=2, fill='blue')
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
                if cfg.detection_box_crop:
                    img_crop = crop_rectangle(im_array, rescaled_geo)
                    cv2.imwrite(os.path.join('output_crop', img_path.split('/')[-1].split('.')[0] + '.jpg'),cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        if flag:
            quad_im.save(os.path.join('output', img_path.split('/')[-1].split('.')[0] + '_predict.jpg'))
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(os.path.join("output_txt", img_path.split('/')[-1].split('.')[0] + '.txt'), 'w') as f_txt:
                f_txt.writelines(txt_items)


def predict_num(frame):
    cv2.imwrite("output.jpg", frame)
    predict(east_detect, "output.jpg", cfg.pixel_threshold)
    

def main(q):
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
            
    
    
    while(True):
                # Capture frame-by-frame
        
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR) 
                # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            
                # color to gray
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                # 顯示圖片           
        
        cv2.imwrite("output.jpg", frame)
        predict(east_detect, "output.jpg", cfg.pixel_threshold)
    
        
        #im = Image.open("output.jpg")
        if os.path.exists(r'C:\Users\test\Desktop\final_presentation\output\output_predict.jpg'):
            img = cv2.imread(r'C:\Users\test\Desktop\final_presentation\output\output_predict.jpg')
            #os.remove(r'C:\Users\test\Desktop\final_presentation\output_crop\output.jpg')
            t2_start(q,img)
            
        global res_before,res_new
        
        while(q.empty()==False):
            
            res_before=q.get()
            #cv2.putText(img, "result = "+str(label_dict[q.get()]), (0,25), 0, 1, (0,255,0),2)
            print(res_before)
            break
        #cv2.imshow('frame', frame)
        
        #img = frame
        if(res_new!=res_before):
            res_new=res_before
            #t3_start(client)
        else:
            cv2.putText(frame, "result = "+ str(res_new), (0,25), 0, 1, (0,255,0),2)
            
        cv2.imshow("frame",frame)
        
        # 按下 q 鍵離開迴圈        
        if cv2.waitKey(1) == ord('q'):
            #cv2.imwrite('output.jpg', frame)
            break
    
            # 釋放該攝影機裝置
    cap.release()
    cv2.destroyAllWindows()
    #thread.join()
    


if __name__ == '__main__':
    try:
        os.remove(r'C:\Users\test\Desktop\final_presentation\output\output_predict.jpg')
    except:
        print("not exist")
    q = Queue()
    res_before = None
    res_new = None
    #frame = None
    main(q)
    
    