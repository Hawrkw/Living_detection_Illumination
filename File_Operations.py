import os
files_path = 'data/fake'
# 遍历文件夹
def walkFile(file):
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        # for f in files:
        #     print(os.path.join(root, f))

        # 遍历所有的文件夹
        for d in dirs:
            #遍历d中的所有文件--四张图片
            image_file_path = os.path.join(root, d)
            for root1,dirs1,files1 in os.walk(image_file_path):
                image_names = []
                for f in files1:
                    if f[-4:] == '.jpg':
                        num = f[:-4]
                        num = int(num)
                        image_names.append(num)
                #对字符串排序会出现2在15后边的情况，所以要将字符串转换为int
                image_names.sort()
                # 将图像命名为-Natural_light,left_light_source,top_light_source,right_ligth_source
                image_name = str(image_names[0]) + '.jpg'
                srcFile = os.path.join(image_file_path,image_name)
                dstFile = os.path.join(image_file_path,'natural_light.jpg')
                os.rename(srcFile,dstFile)
                image_name = str(image_names[1]) + '.jpg'
                srcFile = os.path.join(image_file_path, image_name)
                dstFile = os.path.join(image_file_path, 'left_light_source.jpg')
                os.rename(srcFile, dstFile)
                image_name = str(image_names[2]) + '.jpg'
                srcFile = os.path.join(image_file_path, image_name)
                dstFile = os.path.join(image_file_path, 'top_light_source.jpg')
                os.rename(srcFile, dstFile)
                image_name = str(image_names[3]) + '.jpg'
                srcFile = os.path.join(image_file_path, image_name)
                dstFile = os.path.join(image_file_path, 'right_ligth_source.jpg')
                os.rename(srcFile, dstFile)
                # print(srcFile)
                # print(dstFile)


walkFile(files_path)