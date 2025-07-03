import cv2
import yaml
import numpy as np

worldpts = np.array([[-45,-25,0],[45,-25,0],[45,25,0],[-45,25,0]],dtype=np.float32)
def calcp():
    pass

if __name__=='__main__':
    gtpath = 'models/gt.yml'
    # 定义相机参数
    with open(gtpath, 'r') as file:
        data = yaml.safe_load(file)
    count=0
    jieduan = 600
    imgpts = np.zeros((jieduan,8))
    for key, value in data.items():
        count += 1
        print('count=', count)
        rvec, _ = cv2.Rodrigues(np.array(value[0]['cam_R_m2c'], dtype=np.float32).reshape((3, 3)))
        tvec = np.array([value[0]['cam_t_m2c']], dtype=np.float32)
        rvec = rvec.transpose()[0]
        tvec = tvec[0]
        K = np.array([[1000, 0, 256], [0, 1000, 256], [0, 0, 1]],dtype=np.float32)
        distf = np.array([0.0,0,0,0],dtype=np.float32)

        imgp,_ = cv2.projectPoints(worldpts,rvec,tvec,K,distCoeffs=distf)
        imgp = imgp.reshape(-1)
        imgpts[count-1,:]= imgp
        print(imgp)
        if count == 600:
            break
    np.savetxt('obj5/cpts.txt',imgpts,delimiter=',')
