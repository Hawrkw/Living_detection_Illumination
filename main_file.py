from DifferenceGraph import *
from OptimizationProblem import *
from verification_matplotlib import *
import time
result_path = 'result/'
data_path = 'flash_live/flash_live2/'
scenes = ['flash_live_bright_scenes1','flash_live_bright_scenes2','flash_live_bright_scenes3',
          'flash_live_bright_scenes4','flash_live_bright_scenes5','flash_live_bright_scenes6',
          'flash_live_bright2_light1_scenes7','flash_live_bright2_light2_scenes7']
data_type = ['true','fake']
folder_path =  scenes[7] + '/' + data_type[0] + '/' + '3'
img_folder_path = data_path + folder_path
res,natural_cut_img,nose,face_area = get_difference_graph(
    natural_light_img_path = img_folder_path + '/' + '65.jpg',
           light_source_img_path =img_folder_path + '/' + '66.jpg',)

difference_graph_folder_path = result_path + folder_path
image_names = os.listdir(difference_graph_folder_path)

import os
natrual_light_img_path = os.path.join(img_folder_path, '65.jpg')

image_pathes = [""] * 3
for j in range(len(image_names)):
    image_name = image_names[j]
    if image_name == 'left-nature-fake.jpg' or image_name == 'left-nature-true.jpg':
        image_pathes[0] = (os.path.join(difference_graph_folder_path, image_name))
    elif image_name == 'top-nature-fake.jpg' or image_name == 'top-nature-true.jpg':
        image_pathes[1] = (os.path.join(difference_graph_folder_path, image_name))
    elif image_name == 'right-nature-fake.jpg' or image_name == 'right-nature-true.jpg':
        image_pathes[2] = (os.path.join(difference_graph_folder_path, image_name))
    elif image_name[-3:] == 'txt':
        A = []
        with open(os.path.join(difference_graph_folder_path, image_name), 'r') as f:
            for line in f.readlines():
                A.append(list(map(float, line.split())))
        f.close()


start_time = time.time()
A_all = get_light_direction_of_each_pixel(A,natural_cut_img,nose)
q = get_difference_image(image_pathes)
alpha_all,N_all = method2_3(A_all,q)
draw_hist(alpha_all,color='black',title='Per-pixel lighting_Reflectivity',save_path=difference_graph_folder_path + '/Reflectivity_all_hist.png')
draw_hist(N_all[:,:,0],color='blue',title = 'Per-pixel lighting_Normal direction_Z',save_path=difference_graph_folder_path + '/N_all_direction_Z.png')
draw_hist(N_all[:,:,1],color='green',title = 'Per-pixel lighting_Normal direction_X',save_path=difference_graph_folder_path + '/N_all_direction_X.png')
draw_hist(N_all[:,:,2],color='red',title = 'Per-pixel lighting_Normal direction_Y',save_path=difference_graph_folder_path + '/N_all_direction_Y.png')
draw_normal_direction_line(N_all,natrual_light_img_path,face_area,save_path = difference_graph_folder_path + '/N_all_direction.png')
# normal_all_fileName = os.path.join(path,'Normal_all.npy')
# reflectivity_all_filename = os.path.join(path,'Reflectivity_all.npy')
# np.save(normal_all_fileName,N_all)
# np.save(reflectivity_all_filename,alpha_all)
end_time1 = time.time()
print("逐像素光照计算时间：{}".format(end_time1 - start_time))


A_norm = np.linalg.norm(A,axis=1,keepdims=True)
A = A / A_norm
A = np.matrix(A)
alpha,N = method2_2(A,q)
draw_hist(alpha,color='black',title='Parallel lighting_Reflectivity',save_path=difference_graph_folder_path + '/Reflectivity_hist.png')
draw_hist(N[:,:,0],color='blue',title = 'Parallel lighting_Normal direction_Z',save_path=difference_graph_folder_path + '/N_direction_Z.png')
draw_hist(N[:,:,1],color='green',title = 'Parallel lighting_Normal direction_X',save_path=difference_graph_folder_path + '/N_direction_X.png')
draw_hist(N[:,:,2],color='red',title = 'Parallel lighting_Normal direction_Y',save_path=difference_graph_folder_path + '/N_direction_Y.png')

draw_normal_direction_line(N,natrual_light_img_path,face_area,save_path = difference_graph_folder_path + '/N_direction.png')
# normal_fileName = os.path.join(path,'Normal.npy')
# reflectivity_filename = os.path.join(path,'Reflectivity.npy')
# np.save(normal_fileName,N)
# np.save(reflectivity_filename,alpha)
cv2.waitKey(0)



