import matplotlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et
from PIL import Image, ImageDraw
from utils.util import calculate_intersection_over_union

if __name__ == "__main__":

    image_path = "./data/2007_005354.jpg"
    label_path = "./data/2007_005354.xml"

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

