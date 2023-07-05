
from os.path import join as pjoin
import cv2
import numpy as np

COLOR20 = np.array(
    [[230, 230, 230], [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
     [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
     [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190]])
HEIGHT = int(800)
WIDTH = int(800)
EDGE = int(40)
K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

OTHER_COLOR = np.array([230, 230, 230])

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 2
fontColor              = (0,0,0)
thickness              = 2
lineType               = 3

def save_point_cloud_to_ply(points, colors, save_name='01.ply', save_root='/scratch/genghaoran/GAPartNet/GAPartNet_inference/asset/real'):
    '''
    Save point cloud to ply file
    '''
    PLY_HEAD = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    file_sting = PLY_HEAD
    for i in range(len(points)):
        file_sting += f'{points[i][0]} {points[i][1]} {points[i][2]} {int(colors[i][0])} {int(colors[i][1])} {int(colors[i][2])}\n'
    f = open(pjoin(save_root, save_name), 'w')
    f.write(file_sting)
    f.close()

def draw_bbox(img, bbox_list, trans):
    for i,bbox in enumerate(bbox_list):
        if len(bbox) == 0:
            continue
        bbox = np.array(bbox)
        bbox = bbox * trans[0]+trans[1:4]
        K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        point2image = []
        for pts in bbox:
            x = pts[0]
            y = pts[1]
            z = pts[2]
            x_new = (np.around(x * K[0][0] / z + K[0][2])).astype(dtype=int)
            y_new = (np.around(y * K[1][1] / z + K[1][2])).astype(dtype=int)
            point2image.append([x_new, y_new])
        cl = [255,0,255]
        # import pdb
        # pdb.set_trace()
        cv2.line(img,point2image[0],point2image[1],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[2],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[3],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[4],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[6],point2image[3],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[5],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[3],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[4],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[6],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[1],color=(0,0,255),thickness=3) # red
        cv2.line(img,point2image[0],point2image[3],color=(255,0,0),thickness=3) # green
        cv2.line(img,point2image[0],point2image[2],color=(0,255,0),thickness=3) # blue
    return img

def draw_bbox_old(img, bbox_list, trans):
    for i,bbox in enumerate(bbox_list):
        if len(bbox) == 0:
            continue
        bbox = np.array(bbox)
        bbox = bbox * trans[0]+trans[1:4]
        K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        point2image = []
        for pts in bbox:
            x = pts[0]
            y = pts[1]
            z = pts[2]
            x_new = (np.around(x * K[0][0] / z + K[0][2])).astype(dtype=int)
            y_new = (np.around(y * K[1][1] / z + K[1][2])).astype(dtype=int)
            point2image.append([x_new, y_new])
        cl = [255,0,0]
        cv2.line(img,point2image[0],point2image[1],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[1],color=(255,0,0),thickness=1)
        cv2.line(img,point2image[1],point2image[2],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[2],color=(0,255,0),thickness=1)
        cv2.line(img,point2image[2],point2image[3],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[3],color=(0,0,255),thickness=1)
        cv2.line(img,point2image[3],point2image[0],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[5],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[6],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[0],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[3],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
    return img

def map2image(pts, rgb):
    # input为每个shape的info，取第idx行
    image_rgb = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    num_point = pts.shape[0]
    # print(num_point)
    # print(pts)
    # print(rgb.shape)

    point2image = {}
    for i in range(num_point):
        x = pts[i][0]
        y = pts[i][1]
        z = pts[i][2]
        x_new = (np.around(x * K[0][0] / z + K[0][2])).astype(dtype=int)
        y_new = (np.around(y * K[1][1] / z + K[1][2])).astype(dtype=int)
        point2image[i] = (y_new, x_new)

    # 还原原始的RGB图
    for i in range(num_point):
        # print(i, point2image[i][0], point2image[i][1])
        if point2image[i][0]+1 >= HEIGHT or point2image[i][0] < 0 or point2image[i][1]+1 >= WIDTH or point2image[i][1] < 0:
            continue
        image_rgb[point2image[i][0]][point2image[i][1]] = rgb[i]
        image_rgb[point2image[i][0]+1][point2image[i][1]] = rgb[i]
        image_rgb[point2image[i][0]+1][point2image[i][1]+1] = rgb[i]
        image_rgb[point2image[i][0]][point2image[i][1]+1] = rgb[i]

    # rgb_pil = Image.fromarray(image_rgb, mode='RGB')
    # rgb_pil.save(os.path.join(save_path, f'{instance_name}_{task}.png'))
    return image_rgb

def OBJfile2points(file):
    objFilePath = file
    with open(objFilePath) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3]),float(strs[4]), float(strs[5]), float(strs[6])))
            if strs[0] == "vt":
                break
        points = np.array(points)
    return points 

def FindMaxDis(pointcloud):
    max_xyz = pointcloud.max(0)
    min_xyz = pointcloud.min(0)
    center = (max_xyz + min_xyz) / 2
    max_radius = ((((pointcloud - center)**2).sum(1))**0.5).max()
    return max_radius, center

def WorldSpaceToBallSpace(pointcloud):
    """
    change the raw pointcloud in world space to united vector ball space
    pay attention: raw data changed
    return: max_radius: the max_distance in raw pointcloud to center
            center: [x,y,z] of the raw center
    """
    max_radius, center = FindMaxDis(pointcloud)
    pointcloud_normalized = (pointcloud - center) / max_radius
    return pointcloud_normalized, max_radius, center
