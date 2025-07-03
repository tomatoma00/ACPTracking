import numpy as np
import OpenGL
import yaml as yaml
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
from stl import mesh
stl_model, normals  =None,None
count = 0
# 初始化 OpenGL 窗口
def init_opengl(width, height):
    glutInit([])#
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutCreateWindow(b"OpenGL Camera Rendering")
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)

# 设置相机参数
def set_camera(K, R, t, width, height):
    # 将相机内参矩阵 K 转换为 OpenGL 的投影矩阵
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    near = 0.1
    far = 1600
    glFrustum(-cx * near / fx, (width - cx) * near / fx, (height - cy) * near / fy, -cy * near / fy, near, far)  # 透视投影

    pos = np.dot(-R.T, t.reshape(3, 1)).reshape(1, 3)[0]
    rm = R.T
    rm[:, 0] = -rm[:, 0]
    rm[:, 1] = -rm[:, 1]
    zz = np.array([rm[0, 2], rm[1, 2], rm[2, 2]])
    obj = pos + zz
    gluLookAt(pos[0], pos[1], pos[2], obj[0], obj[1], obj[2], rm[0, 1], rm[1, 1], rm[2, 1])

# 加载STL模型
def load_stl(file_path):
    global stl_model,normals
    stl_model = mesh.Mesh.from_file(file_path)
    normals = []
    for facet in stl_model.vectors:
        normal = np.cross(facet[1] - facet[0], facet[2] - facet[0])
        normal /= np.linalg.norm(normal)
        normals.append(normal)

# 渲染一个简单的 3D 立方体
def render_stlface(stl_model, normals):
    glBegin(GL_TRIANGLES)
    for i, facet in enumerate(stl_model.vectors):
        normal = normals[i]
        if normal[2] >= 0.9:
            normal[2] = (facet[0][2] - 10) / 41.0
        glColor3fv((normal + 1) / 2)  # 使用法向量作为颜色
        for vertex in facet:
            glVertex3fv(vertex)
    glEnd()

# 渲染函数
def render_scene():
    global stl_model,normals
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    render_stlface(stl_model, normals)
    save_image()
    glutSwapBuffers()


# 从 OpenGL 帧缓冲区读取图像并保存为图片
def save_image(width=512, height=512):
    # 从 OpenGL 帧缓冲区读取像素数据
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    # 将像素数据转换为 NumPy 数组
    image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    # 由于 OpenGL 的原点在左下角，而 OpenCV 的原点在左上角，因此需要翻转图像
    # image = np.flipud(image)
    # 保存为图片
    cv2.imwrite(f"obj5/renderext/{count:04d}.png", image)
    # glutSwapBuffers()

def mainrender(posepath,stlpath):
    width, height = 512, 512
    K = np.array([[1000, 0, 256], [0, 1000, 256], [0, 0, 1]], dtype=np.float32)
    distf = np.array([0.0, 0, 0, 0], dtype=np.float32)
    load_stl(stlpath)
    worldpts = np.array([[-45, -25, 0], [45, -25, 0], [45, 25, 0], [-45, 25, 0]], dtype=np.float32)
    jieduan = 3600
    imgpts = np.zeros((jieduan, 8))
    # 定义相机参数
    with open(posepath, 'r') as file:
        data = yaml.safe_load(file)
    for key, value in data.items():
        global count
        rvecori,_ = cv2.Rodrigues(np.array(value[0]['cam_R_m2c'], dtype=np.float32).reshape((3,3)))
        tvecori = np.array([value[0]['cam_t_m2c']], dtype=np.float32)
        rvecori = rvecori.transpose()[0]
        tvecori = tvecori[0]
        for radius in [-60,0,60]:
            rvec = rvecori.copy()
            tvec = tvecori.copy()
            count+=1
            print('count=',count//6+1,'radius',radius)
            tvec[2]+=radius
            R, _ = cv2.Rodrigues(rvec)
            t = tvec
            imgp, _ = cv2.projectPoints(worldpts, rvec, tvec, K, distCoeffs=distf)
            imgp = imgp.reshape(-1)
            imgpts[count - 1, :] = imgp
            init_opengl(width, height)
            set_camera(K, R, t, width, height)
            glutDisplayFunc(render_scene)
            glutMainLoopEvent()
            glutDestroyWindow(glutGetWindow())

            count += 1
            tvec+=np.random.random(3)*np.array([2,2,10])
            rvec+=np.random.random(3)*0.1
            # rvec = np.clip(rvec, 0, 1)
            R, _ = cv2.Rodrigues(rvec)
            t = tvec
            imgp, _ = cv2.projectPoints(worldpts, rvec, tvec, K, distCoeffs=distf)
            imgp = imgp.reshape(-1)
            imgpts[count - 1, :] = imgp
            init_opengl(width, height)
            set_camera(K, R, t, width, height)
            glutDisplayFunc(render_scene)
            glutMainLoopEvent()
            glutDestroyWindow(glutGetWindow())
        if count == jieduan:
            break
    np.savetxt('obj5/cptsext.txt', imgpts, delimiter=',')
# 主函数
if __name__ == "__main__":

    mainrender('models/gt.yml','models/obj_05.stl')