import math

import moderngl
from PyQt5 import QtOpenGL, QtWidgets, QtCore
import numpy as np
from moderngl import Context, Program
from moderngl_window.scene import OrbitCamera
from pyrr import Matrix44
import open3d as o3d
from simulib.simulation_functions import azelToVec
from moderngl_window.context.pyqt5 import Window
from gui_classes import ArcBallUtil
# from resource import shaders
from pyglm import glm


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
        self.camera = OrbitCamera(target=(0, 0, 0),
                                  radius=2.,
                                  fov=self.fov,
                                  aspect_ratio=1,
                                  near=1,
                                  far=100.)
        # self.camera._up = glm.vec3(0.0, 0.0, 1.0)
        self.camera.set_position(2000, 0, 0)
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

        lookat = Matrix44.look_at(
            (0.0, 1., 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        full_matrix = np.array(self.camera.matrix).dot(np.array(self.camera.projection.matrix)).dot(lookat)

        self.mvp.write(np.ascontiguousarray(full_matrix).astype('f4'))

        # Render mesh loop
        self.color.value = self.new_color
        self.vao.render()

        # Render grid loop
        for v in self.vaos:
            v[0].render(v[1])

    def set_mesh(self, new_mesh):
        self.mesh = new_mesh
        self.mesh.translate(np.array([0, 0, 0.]), relative=False)

        # Creates an index buffer
        index_buffer = self.ctx.buffer(np.asarray(self.mesh.triangles, dtype="u4").tobytes())

        # Creates a list of vertex buffer objects (VBOs)
        vao_content = [(self.ctx.buffer(np.asarray(self.mesh.vertices, dtype="f4").tobytes()), '3f', 'in_position'),
                       (self.ctx.buffer(np.asarray(self.mesh.vertex_normals, dtype="f4").tobytes()), '3f', 'in_normal')]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer, 4)

    def modify_mesh(self, pos=None, att=None):
        if pos is not None:
            self.mesh.translate(pos, relative=False)
        if att is not None:
            self.mesh.rotate(self.mesh.get_rotation_matrix_from_xyz(att))

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
        self.camera.projection.update(aspect_ratio=self.aspect_ratio)

    @property
    def aspect_ratio(self):
        return self.width() / max(1.0, self.height())

    def make_wireframe(self):
        self.is_wireframe = True

    def make_solid(self):
        self.is_wireframe = False

    def lookat(self, target, eye):
        self.camera.set_position(*eye)
        self.camera.look_at()

    def update_zoom(self, event):
        self.camera.zoom_state(event.angleDelta().y() * .01)

    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.prev_x = event.x()
            self.prev_y = event.y()
            self.camera.rot_state(event.x(), event.y())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.camera.rot_state(event.x() - self.prev_x, event.y() - self.prev_y)
            self.prev_x = event.x()
            self.prev_y = event.y()