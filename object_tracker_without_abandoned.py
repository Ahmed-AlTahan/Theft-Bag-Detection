import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import math
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

bags = dict()
bagcents = dict()
percents = dict()
curBagcent = dict()
curPercent = dict()

stolen = False
ownerofbag = -1

abandoned = False
leftbag = -1

def main(_argv):
    global ownerofbag, stolen, leftbag, abandoned
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person','backpack','suitcase']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            if track.class_name == "backpack" or track.class_name == "suitcase":
                bagcent = track.to_tlwh()
                curBagcent[str(track.track_id)] = bagcent #[bagcent[0] + (bagcent[2]/2), bagcent[1] + (bagcent[3]/2)]
            if track.class_name == "person":
                percent = track.to_tlwh()
                curPercent[str(track.track_id)] = percent #[bagcent[0] + (bagcent[2]/2), bagcent[1] + (bagcent[3]/2)]
          
        
        
        
        
        # update tracks
        for track in tracker.tracks:
            """if track.class_name == "backpack" or track.class_name == "suitcase":
                if track.is_deleted():
                    print("DELETED")
                elif track.is_tentative():
                    print("TENTATIVE")
            if track.time_since_update > 1 and (track.class_name == "backpack" or track.class_name == "suitcase"):
                print("Bag deleted! Checking master")
                for tra in tracker.tracks:
                    if tra.track_id == int(bags[str(track.track_id)]):
                        if track.time_since_update <= 1:
                            print("The bag "+str(track.track_id)+" is out of the shot, but the master is not")
                        else:
                            print("The bag "+str(track.track_id)+" is out of the shot, so is its master")
                        break"""

            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            if track.class_name == "backpack" or track.class_name == "suitcase":
                #bagcent = track.to_tlwh()
                #bagcents[str(track.track_id)] = bagcent #[bagcent[0] + (bagcent[2]/2), bagcent[1] + (bagcent[3]/2)]
                if str(track.track_id) not in bags.keys():
                    bags[str(track.track_id)] = "-1"
                    print("Added new bag: "+str(track.track_id))
                if int(bags[str(track.track_id)]) == -1: #the current object is a bag, and it has no master
                    print("Bag "+str(track.track_id)+" does not have a master, finding a master")
                    mindist = 10e8
                    foundid = -1
                    for trac in tracker.tracks:
                        if not trac.is_confirmed() or trac.time_since_update > 1:
                            continue 
                        if trac.class_name == "person":
                            bagcent = track.to_tlwh()
                            bagcent = [bagcent[0] + (bagcent[2]/2), bagcent[1] + (bagcent[3]/2)]
                            percent = trac.to_tlwh()
                            percent = [percent[0] + (percent[2]/2), percent[1] + (percent[3]/2)]
                            dist = math.sqrt((bagcent[0] - percent[0])**2 + (bagcent[1] - percent[1])**2)
                            if dist <= mindist and dist <= 2.5*track.to_tlwh()[2]:
                                mindist = dist
                                foundid = trac.track_id
                    bags[str(track.track_id)] = str(foundid)
                    if foundid != -1:
                        print("Bag "+str(track.track_id)+" now has a master: "+str(foundid))
                else:   #the current object is a bag, and it has a master
                    """for tra in tracker.tracks:
                        if tra.track_id == int(bags[str(track.track_id)]) and not tra.is_confirmed():
                            print("The bag "+str(track.track_id)+" has been abandoned!")"""
                    if len(bagcents.keys()) != 0:
                        print("Study the theft of bag: "+str(track.track_id))
                        #print("Study the theft of bag: "+str(track.track_id))
                        mcent = curPercent[bags[str(track.track_id)]]
                        mc = [mcent[0] + (mcent[2]/2), mcent[1] + (mcent[3]/2)] #master center
                        bcent = curBagcent[str(track.track_id)]
                        bc = [bcent[0] + (bcent[2]/2), bcent[1] + (bcent[3]/2)] #bag center
                        
                        mcentp = percents[bags[str(track.track_id)]] #master bb previous
                        mcp = [mcentp[0] + (mcentp[2]/2), mcentp[1] + (mcentp[3]/2)] #master center previous
                        bcentp = bagcents[str(track.track_id)] #bag bb previous
                        bcp = [bcentp[0] + (bcentp[2]/2), bcentp[1] + (bcentp[3]/2)] #bag center previous
                        
                        distb = math.sqrt((bc[0] - bcp[0])**2 + (bc[1] - bcp[1])**2) #delta center bag previous 
                        distbm = math.sqrt((bc[0] - mc[0])**2 + (bc[1] - mc[1])**2) #distance from master
                        distm = math.sqrt((mc[0] - mcp[0])**2 + (mc[1] - mcp[1])**2) #delta center master previous
                        if distbm >= 4.5*mcent[2]:
                            print("THEEEEEEEFFFFFTTTTT!!!!")
                            stolen = True
                            ownerofbag = int(bags[str(track.track_id)])
                        else:
                            stolen = False
                            ownerofbag = -1
                        if stolen:
                            cv2.line(frame, (int(mc[0]), int(mc[1])), (int(bc[0]), int(bc[1])), (255, 0, 0), 3)
                        else:
                            cv2.line(frame, (int(mc[0]), int(mc[1])), (int(bc[0]), int(bc[1])), (0, 0, 255), 3)
                    """if track.is_deleted():
                        for tra in tracker.tracks:
                            if tra.track_id == int(bags[str(track.track_id)]):
                                if tra.is_confirmed"""
                    #pass #study the cases here [theft, abandoned, neutral]
            #if track.class_name == "person":
               # percent = track.to_tlwh()
               # percents[str(track.track_id)] = percent #[percent[0] + (percent[2]/2), percent[1] + (percent[3]/2)]
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            if track.track_id == ownerofbag:
                cv2.putText(frame, class_name + "-" + str(track.track_id) + " - ROBBED",(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            else:
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        peopleinf = 0
        bagsinf = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            if track.class_name == "backpack" or track.class_name == "suitcase":
                bagsinf += 1
                bagcent = track.to_tlwh()
                bagcents[str(track.track_id)] = bagcent #[bagcent[0] + (bagcent[2]/2), bagcent[1] + (bagcent[3]/2)]
            if track.class_name == "person":
                peopleinf += 1
                percent = track.to_tlwh()
                percents[str(track.track_id)] = percent #[bagcent[0] + (bagcent[2]/2), bagcent[1] + (bagcent[3]/2)]
        pplinfo = "People in frame: "+str(peopleinf)
        bagsinfo = "Bags in frame: "+str(bagsinf)
        theftinfo = "Theft detected: "+str(stolen)
        if stolen:
            theftcase = "Person "+str(ownerofbag)+" had his bag stolen!!"
        """cv2.putText(frame, pplinfo,(10, 30),0, 1, (0,225,0),2)
        cv2.putText(frame, bagsinfo,(10, 60),0, 1, (0,225,0),2)
        cv2.putText(frame, theftinfo,(10, 90),0, 1, (0,225,0),2)
        if stolen:
            cv2.putText(frame, theftcase,(10, 120),0, 1, (200,0,0),2)
            cv2.putText(frame, abandinfo,(10, 150),0, 1, (0,225,0),2)
        else:
            cv2.putText(frame, abandinfo,(10, 120),0, 1, (0,225,0),2)
        if abandoned:
            if stolen:
                cv2.putText(frame, abandcase,(10, 180),0, 1, (200,200,0),2)
            else:
                cv2.putText(frame, abandcase,(10, 150),0, 1, (200,200,0),2)"""
        botpadding = -12     #bottom padding for display info
        
        if stolen:
            cv2.putText(frame, theftcase,(10, frame_size[0] - 0+botpadding),0, 1, (200,0,0),2)
            cv2.putText(frame, theftinfo,(10, frame_size[0] - 30+botpadding),0, 1, (0,225,0),2)
            cv2.putText(frame, bagsinfo,(10, frame_size[0] - 60+botpadding),0, 1, (0,225,0),2)
            cv2.putText(frame, pplinfo,(10, frame_size[0] - 90+botpadding),0, 1, (0,225,0),2)
        else:
            cv2.putText(frame, theftinfo,(10, frame_size[0] - 0+botpadding),0, 1, (0,225,0),2)
            cv2.putText(frame, bagsinfo,(10, frame_size[0] - 30+botpadding),0, 1, (0,225,0),2)
            cv2.putText(frame, pplinfo,(10, frame_size[0] - 60+botpadding),0, 1, (0,225,0),2)
        
        """cv2.putText(frame, pplinfo,(10, 30),0, 1, (0,225,0),2)
        cv2.putText(frame, bagsinfo,(10, 60),0, 1, (0,225,0),2)
        cv2.putText(frame, theftinfo,(10, 90),0, 1, (0,225,0),2)
        if stolen:
            cv2.putText(frame, theftcase,(10, 120),0, 1, (200,0,0),2)
            cv2.putText(frame, abandinfo,(10, 150),0, 1, (0,225,0),2)
        else:
            cv2.putText(frame, abandinfo,(10, 120),0, 1, (0,225,0),2)
        if abandoned:
            if stolen:
                cv2.putText(frame, abandcase,(10, 180),0, 1, (200,200,0),2)
            else:
                cv2.putText(frame, abandcase,(10, 150),0, 1, (200,200,0),2)"""
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
