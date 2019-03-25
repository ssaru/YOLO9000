import matplotlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et
import numpy as np
import torch
import random
from PIL import Image, ImageDraw
from utils.util import calculate_intersection_over_union, iou_pytorch

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]

def _convert_coordinate(image_size, box_coordinate):
    image_size_keys = image_size.keys()
    box_coordinate_keys = box_coordinate.keys()
    assert "width" in image_size_keys
    assert "height" in image_size_keys
    assert "xmin" in box_coordinate_keys
    assert "ymin" in box_coordinate_keys
    assert "xmax" in box_coordinate_keys
    assert "ymax" in box_coordinate_keys
    assert isinstance(image_size, dict)
    assert isinstance(box_coordinate, dict)
    assert isinstance(image_size["width"], float)
    assert isinstance(image_size["height"], float)
    assert isinstance(box_coordinate["xmin"], float)
    assert isinstance(box_coordinate["ymin"], float)
    assert isinstance(box_coordinate["xmax"], float)
    assert isinstance(box_coordinate["ymax"], float)

    x_of_box = (box_coordinate["xmin"] + box_coordinate["xmax"]) / 2.0
    y_of_box = (box_coordinate["ymin"] + box_coordinate["ymax"]) / 2.0
    width_of_box = box_coordinate["xmax"] - box_coordinate["xmin"]
    height_of_box = box_coordinate["ymax"] - box_coordinate["ymin"]

    relative_x_of_center = x_of_box / image_size["width"]
    relative_y_of_center = y_of_box / image_size["height"]
    relative_box_width = width_of_box / image_size["width"]
    relative_box_height = height_of_box / image_size["height"]

    return [relative_x_of_center, relative_y_of_center,
            relative_box_width, relative_box_height]

def _convert_box_label_to_yolo_label(label, classes_list):
    assert isinstance(label, dict)
    assert isinstance(classes_list, list)
    for cls in classes_list:
        assert isinstance(cls, str)

    root_keys = label.keys()
    size_keys = label["size"].keys()
    number_of_objects = len(label["object"])

    assert "size" in root_keys
    assert "object" in root_keys
    assert "width" in size_keys
    assert "height" in size_keys
    assert number_of_objects != 0

    yolo_label = list()

    image_size = {
        "width": float(label["size"]["width"]),
        "height": float(label["size"]["height"]),
    }

    for _object in label["object"]:
        _object_keys = _object.keys()
        assert "name" in _object_keys
        assert "xmin" in _object_keys
        assert "ymin" in _object_keys
        assert "xmax" in _object_keys
        assert "ymax" in _object_keys

        name = _object["name"]
        cls = float(classes_list.index(name))
        box_coordinate = {
            "xmin": float(_object["xmin"]),
            "ymin": float(_object["ymin"]),
            "xmax": float(_object["xmax"]),
            "ymax": float(_object["ymax"]),
        }

        yolo_coordinate = _convert_coordinate(image_size, box_coordinate)
        yolo_coordinate.insert(0, cls)
        yolo_label.append(yolo_coordinate)

    return yolo_label

def _parse_voc(annotation_path):
    import xml.etree.ElementTree as Et
    assert isinstance(annotation_path, str)

    xml_file = open(annotation_path, "r")
    tree = Et.parse(xml_file)

    element_list = list()
    for elem in tree.iter():
        element_list.append(elem.tag)

    assert "size" in element_list
    assert "width" in element_list
    assert "height" in element_list
    assert "object" in element_list
    assert "name" in element_list
    assert "bndbox" in element_list
    assert "xmin" in element_list
    assert "ymin" in element_list
    assert "xmax" in element_list
    assert "ymax" in element_list

    result = dict()
    root = tree.getroot()

    size_tag = root.find("size")

    result["size"] = {
        "width": size_tag.find("width").text,
        "height": size_tag.find("height").text,
        "depth": size_tag.find("depth").text
    }

    result["object"] = list()

    objects = root.findall("object")
    assert objects

    for _object in objects:
        result["object"].append({
            "name": _object.find("name").text,
            "xmin": _object.find("bndbox").find("xmin").text,
            "ymin": _object.find("bndbox").find("ymin").text,
            "xmax": _object.find("bndbox").find("xmax").text,
            "ymax": _object.find("bndbox").find("ymax").text
        })

    return result

def getitem(path):
    boxes_info_dict = _parse_voc(path)
    boxes_info_yolo = _convert_box_label_to_yolo_label(boxes_info_dict, CLASSES)

    return boxes_info_yolo

def build_tensor_block(target):
    S = 13
    np_label = np.zeros((S, S, 6), dtype=np.float32)

    for _object in target:
        objectness = 1.
        cls = _object[0]
        bx = _object[1]
        by = _object[2]
        bw = _object[3]
        bh = _object[4]

        # can be acuqire grid (x,y) index when divide (1/S) of x_ratio
        scale_factor = (1 / S)
        cx = int(bx // scale_factor)
        cy = int(by // scale_factor)
        sigmoid_tx = (bx / scale_factor) - cx
        sigmoid_ty = (by / scale_factor) - cy

        # insert object row in specific label tensor index as (x,y)
        # object row follow as
        # [objectness, class, x offset, y offset, width ratio, height ratio]
        np_label[cy][cx] = np.array([objectness, sigmoid_tx, sigmoid_ty, bw, bh, cls])

    label = torch.from_numpy(np_label)
    result = [label]

    return torch.stack(result, 0)

def label_generator():
    # sample
    # {'size': {'width': '500', 'height': '375', 'depth': '3'},
    # 'object': [{'name': 'motorbike', 'xmin': '230', 'ymin': '114', 'xmax': '330', 'ymax': '221'},
    # {'name': 'motorbike', 'xmin': '5', 'ymin': '98', 'xmax': '222', 'ymax': '375'},
    # {'name': 'person', 'xmin': '361', 'ymin': '115', 'xmax': '437', 'ymax': '265'},
    # {'name': 'person', 'xmin': '1', 'ymin': '89', 'xmax': '38', 'ymax': '184'}]}

    case = [
        # 1
        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '105', 'ymin': '58', 'xmax': '270', 'ymax': '180'}]
         },

        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '280', 'ymin': '37', 'xmax': '410', 'ymax': '170'}]
         },

        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '185', 'ymin': '167', 'xmax': '264', 'ymax': '274'}]
         },

        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '250', 'ymin': '148', 'xmax': '370', 'ymax': '294'}]
         },

        # 2
        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '68', 'ymin': '58', 'xmax': '270', 'ymax': '250'}]
         },

        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '148', 'ymin': '37', 'xmax': '410', 'ymax': '170'}]
         },

        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '185', 'ymin': '167', 'xmax': '368', 'ymax': '274'}]
         },

        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '250', 'ymin': '86', 'xmax': '370', 'ymax': '294'}]
         },

        # 3
        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '105', 'ymin': '158', 'xmax': '270', 'ymax': '180'}]
         },

        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '280', 'ymin': '37', 'xmax': '310', 'ymax': '170'}]
         },

        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '236', 'ymin': '167', 'xmax': '264', 'ymax': '274'}]
         },

        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '250', 'ymin': '148', 'xmax': '370', 'ymax': '200'}]
         },

        # 4
        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '120', 'ymin': '78', 'xmax': '387', 'ymax': '300'}]
         },

        # 5
        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '245', 'ymin': '127', 'xmax': '310', 'ymax': '210'}]
         },

        # 6
        {'size': {'width': '500', 'height': '375', 'depth': '3'},
         'object': [{'name': 'motorbike', 'xmin': '230', 'ymin': '114', 'xmax': '330', 'ymax': '221'}]
         },

    ]

    return random.choice(case)

def np_iou(blockA: np.array, blockB: np.array, config: dict):

    S = config["S"]
    W = config["image_width"]
    H = config["image_height"]

    dx = W // S
    dy = H // S

    blockA = np.squeeze(blockA)
    A_tx = blockA[:, :, 0]
    A_ty = blockA[:, :, 1]
    A_tw = blockA[:, :, 2]
    A_th = blockA[:, :, 3]

    blockB = np.squeeze(blockB)
    B_tx = blockB[:, :, 0]
    B_ty = blockB[:, :, 1]
    B_tw = blockB[:, :, 2]
    B_th = blockB[:, :, 3]


    # TODO. build 13x13 [xmin, ymin, xmax, ymax] tensor block
    # TODO. & Calc IOU
    for i in range(S):
        for j in range(S):
            A_center_x = dx * i + int(dx * A_tx[j][i])
            A_center_y = dy * j + int(dx * A_tx[j][i])
            A_width = int(A_tw[j][i] * W)
            A_height = int(A_th[j][i] * H)

            Axmin = A_center_x - (A_width // 2)
            Aymin = A_center_y - (A_height // 2)
            Axmax = Axmin + A_width
            Aymax = Aymin + A_height

    print(blockA.shape)
    print(blockB.shape)

    return None

if __name__ == "__main__":

    image_path = "./data/2007_005354.jpg"
    label_path = "./data/2007_005354.xml"

    label= getitem(label_path)
    tensor_label = build_tensor_block(label)

    test_case = label_generator()
    test_yolo = _convert_box_label_to_yolo_label(test_case, CLASSES)
    tensor_testcase = build_tensor_block(test_yolo)

    np_label = tensor_label.numpy()[:, :, :, 1:5]
    np_testcase = tensor_testcase.numpy()[:, :, :, 1:5]

    config = {
        "S" : 13,
        "image_width" : 500,
        "image_height" : 375
    }
    iou = np_iou(np_label, np_testcase, config)


    exit()



def backup():
    # xml parsing
    label = dict()

    xml_file = open(label_path, 'r')
    tree = Et.parse(xml_file)
    root = tree.getroot()

    size_tag = root.find("size")

    label["size"] = {
        "width": size_tag.find("width").text,
        "height": size_tag.find("height").text,
        "depth": size_tag.find("depth").text
    }

    label["objects"] = list()

    objects = root.findall("object")
    for _object in objects:
        label["objects"].append(
            {
                "name": _object.find("name").text,
                "xmin": _object.find("bndbox").find("xmin").text,
                "ymin": _object.find("bndbox").find("ymin").text,
                "xmax": _object.find("bndbox").find("xmax").text,
                "ymax": _object.find("bndbox").find("ymax").text
            }
        )

    # print label
    print(label)

    # image open
    image = Image.open(image_path)

    # draw specific label
    # {'name': 'motorbike', 'xmin': '230', 'ymin': '114', 'xmax': '330', 'ymax': '221'}
    GT = [230, 114, 330, 221]

    """
    "1": [[105, 58, 270, 180],
          [280, 37, 410, 170],
          [185, 167, 264, 274],
          [250, 148, 370, 294]
          ],
        
    "2": [[68, 58, 270, 250],
          [148, 37, 410, 170],
          [185, 167, 368, 274],
          [250, 86, 370, 294]
          ]
          
    "3": [[105, 158, 270, 180],
          [280, 37, 310, 170],
          [236, 167, 264, 274],
          [250, 148, 370, 200]
          ],
          
    "4": [[120, 78, 387, 300],
          ],
          
    "5": [[245, 127, 312, 210],
          ],
    """
    test_case = {
        "1": [[105, 58, 270, 180],
              [280, 37, 410, 170],
              [185, 167, 264, 274],
              [250, 148, 370, 294]
              ],

        "2": [[68, 58, 270, 250],
              [148, 37, 410, 170],
              [185, 167, 368, 274],
              [250, 86, 370, 294]
              ],

        "3": [[105, 158, 270, 180],
              [280, 37, 310, 170],
              [236, 167, 264, 274],
              [250, 148, 370, 200]
              ],

        "4": [[120, 78, 387, 300],
              ],

        "5": [[245, 127, 312, 210],
              ],

        "6": [[230, 114, 330, 221],
              ],
    }

    for key in test_case:
        for dummy_pred_box in test_case[key]:
            # draw GT object
            img = image.convert('RGB')
            draw = ImageDraw.Draw(img)
            draw.rectangle(((GT[0], GT[1]), (GT[2], GT[3])), outline="blue")
            draw.text((GT[0], GT[1]), "motorbike")

            print(dummy_pred_box)
            _xmin = dummy_pred_box[0]
            _ymin = dummy_pred_box[1]
            _xmax = dummy_pred_box[2]
            _ymax = dummy_pred_box[3]

            draw.rectangle(((_xmin, _ymin), (_xmax, _ymax)), outline="red")

            iou = calculate_intersection_over_union(GT, dummy_pred_box)
            draw.text((_xmin, _ymin), str(int(100*iou)) + "%")
            print("iou : {}".format(iou))

            fig = plt.figure()
            plt.imshow(img)
            plt.show()
            plt.close()

