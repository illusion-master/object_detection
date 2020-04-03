import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import os
import pathlib

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

print(detection_model.inputs)
detection_model.output_dtypes
detection_model.output_shapes

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()


  return output_dict

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=False,
      min_score_thresh = .5,
      agnostic_mode = False,
      line_thickness=8)

  display(Image.fromarray(image_np))

#show_inference(detection_model, 'image/4.jpeg')
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk


def Display(model, image_path):
  list_rough.append(image_path)
  image_path= pathlib.Path(image_path)
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=False,
      min_score_thresh = .5,
      agnostic_mode = False,
      line_thickness=8)

  photo = ImageTk.PhotoImage(Image.fromarray(image_np))

  label.configure(image=photo)
  label.image = photo # keep a reference!
  return
list_rough = []
window = tk.Tk()

img_names = []
def select_folder():
    global img_name
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    data_dir = os.path.join(folder_selected)
    os.chdir(data_dir)
    data_dir = pathlib.Path(folder_selected)
    image_count = len(list(data_dir.glob('*.jpg*')))
    image_count
    img_names.append(np.array([item.name for item in data_dir.glob('*')]))
    img_name = img_names[0]
    global j
    j = j+1

i = 0
def next_img():
    global i, img_name
    if(i< len(img_names)):
        i = i+1
    else:
        i =0

def prev_img():
    global i, img_name
    if(i == 0):
        i = len(img_names)-1
    else:
        i = i-1


img_names.append([])
img_names[0].append('image/4.jpeg')

j =0

b1 = tk.Button(window, text = 'Select folder', width = 12, command = select_folder)
b1.grid(row= 0, column= 0)

b2 = tk.Button(window, text = 'next', width = 12, command = next_img)
b2.grid(row= 1, column= 0)

b3 = tk.Button(window, text = 'previous', width = 12, command = prev_img)
b3.grid(row= 2, column= 0)

label = tk.Label(window, text = 'nothing to show yet')
label.grid(row = 4, column = 5)

l1=tk.Label(window,text ='Enter minimum score to predict')
l1.grid(row=5,column=0)

min_score = tk.StringVar()
e1=tk.Entry(window,textvariable=min_score)
e1.grid(row=5,column=2)

b4 = tk.Button(window, text = 'Detect', width = 12, command =lambda: Display(detection_model, str(img_names[j][i])))
b4.grid(row= 3, column= 0)

window.mainloop()
