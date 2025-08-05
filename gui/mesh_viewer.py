import math
import time
import moderngl
from PyQt5 import QtOpenGL, QtWidgets, QtCore
import numpy as np
from moderngl import Context, Program
from pyrr import Matrix44
from simulib.simulation_functions import azelToVec
from moderngl_window.context.pyqt5 import Window
from scipy.spatial.transform import Rotation as rot


rotation_matrix = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])


def ball(size, az_samples, el_samples, rotate=True):
    m_pans, m_tilts = np.meshgrid(np.linspace(0, 2 * np.pi, az_samples, endpoint=False),
                                  np.linspace(np.pi / 2 - .1, -np.pi / 2 + .1, el_samples))
    a_pan = m_pans.flatten()
    a_tilt = m_tilts.flatten()
    boresights = azelToVec(a_pan, a_tilt).T
    new_grid = -boresights * size

    return np.dot(new_grid, rotation_matrix) if rotate else new_grid


class QGLControllerWidget(QtOpenGL.QGLWidget):
    ctx: Context = None
    prog: Program = None
    vbo = None
    vao = None
    vaos: list = []
    def __init__(self, parent=None, grid_mode=0):
        self.mesh = None
        self.parent = parent
        self.grid_mode = grid_mode
        super(QGLControllerWidget, self).__init__(parent)

        # Initialize OpenGL parameters
        self.bg_color = (0.1, 0.1, 0.1, 0.1)
        self.color_alpha = 1.0
        self.new_color = (1.0, 1.0, 1.0, self.color_alpha)
        self.fov = 75.0
        self.setMouseTracking(True)
        self.wheelEvent = self.update_zoom
        self.is_wireframe = False
        self.texture = None
        self.grid_alpha_value = 1.
        self.camera = Camera((15., 0, 0))
        self.prev_x = 0
        self.prev_y = 0

    def initializeGL(self):
        # Create a new OpenGL context
        self.ctx = moderngl.create_context()

        # Create the shader program
        self.prog = self.ctx.program(
            vertex_shader='''
                        #version 330
                        uniform mat4 Mvp;
                        in vec3 in_position;
                        in vec3 in_normal;
                        out vec3 v_vert;
                        out vec3 v_norm;
                        void main() {
                            v_vert = in_position;
                            v_norm = in_normal;
                            gl_Position = Mvp * vec4(in_position, 1.0);
                        }
                    ''',
            fragment_shader='''
                        #version 330
                        uniform vec4 Color;
                        uniform vec3 Light;
                        in vec3 v_vert;
                        in vec3 v_norm;
                        out vec4 f_color;
                        void main() {
                            float lum = dot(normalize(v_norm),
                                            normalize(v_vert - Light));
                            lum = acos(lum) / 3.14159265;
                            lum = clamp(lum, 0.0, 1.0);
                            lum = lum * lum;
                            lum = smoothstep(0.0, 1.0, lum);
                            lum *= smoothstep(0.0, 80.0, v_vert.z) * 0.3 + 0.7;
                            lum = lum * 0.8 + 0.2;
                            vec3 color = Color.rgb * Color.a;
                            f_color = vec4(color * lum, 1.0);
                        }
                    '''
        )
        # Setting shader parameters
        self.light = self.prog['Light']
        self.color = self.prog['Color']
        self.mvp = self.prog['Mvp']
        # self.prog["Texture"].value = 0
        self.light.value = (1.0, 1.0, 1.0)
        self.color.value = (1.0, 1.0, 1.0, 1.0)

        # Setting mesh parameters
        self.mesh = None
        self.ctx.point_size = 5.
        self.ctx.depth_func = '1'

    def paintGL(self):
        # OpenGL loop
        self.ctx.clear(*self.bg_color)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.wireframe = self.is_wireframe
        if self.mesh is None:
            return

        proj = Matrix44.perspective_projection(self.fov, self.aspect_ratio, .1, 10000.)
        full_matrix = proj * self.camera.matrix

        self.mvp.write(np.ascontiguousarray(full_matrix).astype('f4'))

        # Render mesh loop
        self.color.value = self.new_color
        self.vao.render()

        # Render grid loop
        for v in self.vaos:
            v[0].render(v[1])

    def set_mesh(self, new_mesh):
        self.mesh = new_mesh
        self.mesh.shift(np.array([0, 0, 0.]), relative=False)

        # Creates an index buffer
        index_buffer = self.ctx.buffer(np.asarray(self.mesh.tri_idx, dtype="u4").tobytes())

        # Creates a list of vertex buffer objects (VBOs)
        vao_content = [(self.ctx.buffer(np.asarray(self.mesh.vertices, dtype="f4").tobytes()), '3f', 'in_position'),
                       (self.ctx.buffer(np.asarray(self.mesh.vertex_normals, dtype="f4").tobytes()), '3f', 'in_normal')]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer, 4)

    def modify_mesh(self, pos=None, att=None):
        if pos is not None:
            self.mesh.shift(pos, relative=False)
        if att is not None:
            self.mesh.rotate(att)

        # Creates an index buffer
        index_buffer = self.ctx.buffer(np.asarray(self.mesh.triangles, dtype="u4").tobytes())

        # Creates a list of vertex buffer objects (VBOs)
        vao_content = [(self.ctx.buffer(np.asarray(self.mesh.vertices, dtype="f4").tobytes()), '3f', 'in_position'),
                       (self.ctx.buffer(np.asarray(self.mesh.vertex_normals, dtype="f4").tobytes()), '3f',
                        'in_normal')]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer, 4)

    def update_grid(self, vao_idx, pts, render_type=moderngl.POINTS):
        assert vao_idx < len(self.vaos), "index is out of range for vao"
        # vbo = self.ctx.buffer(np.dot(pts, rotation_matrix).astype('f4'))
        vbo = self.ctx.buffer(np.ascontiguousarray(pts).astype('f4'))
        self.vaos[vao_idx] = [self.ctx.simple_vertex_array(self.prog, vbo, 'in_position'), render_type]

    def add_grid(self, pts, render_type=moderngl.POINTS):
        vbo = self.ctx.buffer(np.ascontiguousarray(pts).astype('f4'))
        self.vaos.append([self.ctx.simple_vertex_array(self.prog, vbo, 'in_position'), render_type])

    def resizeGL(self, width, height):
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)

    @property
    def aspect_ratio(self):
        return self.width() / max(1.0, self.height())

    def look_at(self, target, eye):
        self.camera.look_at(target, eye)

    def make_wireframe(self):
        self.is_wireframe = True

    def make_solid(self):
        self.is_wireframe = False

    def update_zoom(self, event):
        self.camera.zoom_to(event.angleDelta().y() * .01)

    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.prev_x = event.x()
            self.prev_y = event.y()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.camera.move_to(self.prev_x - event.x(), self.prev_y - event.y())
        if event.buttons() & QtCore.Qt.RightButton:
            self.camera.rotate(self.prev_x - event.x(), self.prev_y - event.y())
            self.prev_x = event.x()
            self.prev_y = event.y()


class Camera(object):

    def __init__(self, pos: np.array):
        self.pos = np.array(pos)
        self.center = np.zeros(3)

    def move_to(self, dx: float, dy: float):
        # Map to camera coordinates before applying motion change
        upconv = np.array(self.matrix[:3, 1])
        rightconv = np.array(self.matrix[:3, 0])
        motion = (upconv * -dy * 2 + rightconv * dx * 2) * .01 * np.linalg.norm(self.pos_center) / 1000.
        self.center += motion
        self.pos += motion

    def look_at(self, target: np.array, eye: np.array):
        self.center = target
        self.pos = eye

    def rotate(self, dx: float, dy: float):
        upconv = np.array(self.matrix[:3, 1])
        un = upconv / np.linalg.norm(upconv) * np.sin(dx * .005)
        rightconv = np.array(self.matrix[:3, 0])
        rn = rightconv / np.linalg.norm(rightconv) * np.sin(dy * .005)
        rv = rot.from_quat(np.array([*un, np.cos(dx * .005)])) * rot.from_quat(np.array([*rn, np.cos(dy * .005)]))
        self.pos = rv.apply(self.pos_center) + self.center

    def zoom_to(self, zoom: float):
        rng = np.linalg.norm(self.pos_center)
        self.pos = self.pos + self.pos_center / rng * zoom * (np.sqrt(rng) + 1)

    @property
    def matrix(self):
        return Matrix44.look_at(self.pos, self.center, (0, 0, 1.))

    @property
    def pos_center(self):
        return self.pos - self.center

    @property
    def rotation(self):
        return rot.from_rotvec(np.array([np.arctan2(self.pos_center[0], self.pos_center[1]), 0., np.arctan2(self.pos_center[0], self.pos_center[2])]))



