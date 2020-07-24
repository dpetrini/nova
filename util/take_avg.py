# Tira media das imagens do diretorio
#https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
#
# Take mean and std of all images in folder
#
# "You average the variances; then you can take square root to geti
#   the average standard deviation."
#
# DGPP 06-2020
#
import numpy as np
import cv2
import os
import argparse

size_8 = False 

# MAIN
ap = argparse.ArgumentParser(description='Take mean and std of images in input folder.')
ap.add_argument("-i", "--input",  required = True,
     help = "folfder to generate patches from. (no trailing /")

args = vars(ap.parse_args())
print(args)
input_path = args['input']


# Tira media das imagens do diretorio
img_list = [f for f in os.listdir(input_path) if f.endswith('.png')]
acc = 0
acc_var = 0
n = 0
for f in img_list:
    print(f)
    image = cv2.imread(os.path.join(input_path, f), cv2.IMREAD_GRAYSCALE if size_8 else cv2.IMREAD_UNCHANGED)
    mean = image.mean()
    var = image.var()
    acc = acc + mean
    acc_var = acc_var + var
    n += 1
    print(mean, var, np.sqrt(var))
all_mean = acc/n
all_var_mean = acc_var/n
print('Mean:', all_mean, 'Acumulator, n: ', acc, n)
print('Mean var', all_var_mean, ' Sum var: ', acc_var, ' Std: ', np.sqrt(all_var_mean))

