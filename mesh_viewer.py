import moderngl
from PyQt5 import QtOpenGL, QtWidgets, QtCore
import numpy as np
from pyrr import Matrix44
import open3d as o3d
from pyrr.matrix44 import create_perspective_projection_matrix_from_bounds


class QGLControllerWidget(QtOpenGL.QGLWidget):

    def __init__(self, parent=None):
        self.parent = parent
        super(QGLControllerWidget, self).__init__(parent)

    def initializeGL(self):
        self.ctx = moderngl.create_context()

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

        self.light = self.prog['Light']
        self.color = self.prog['Color']
        self.mvp = self.prog['Mvp']
        self.mesh = None
        self.center = np.zeros(3)
        self.scale = 1.0

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.mesh.translate(np.array([0, 0, 0.]), relative=False)
        index_buffer = self.ctx.buffer(
            np.array(np.asarray(self.mesh.triangles), dtype="u4").tobytes())
        vao_content = [
            (self.ctx.buffer(
                np.array(np.asarray(self.mesh.vertices), dtype="f4").tobytes()),
                '3f', 'in_position'),
            (self.ctx.buffer(
                np.array(np.asarray(self.mesh.vertex_normals), dtype="f4").tobytes()),
                '3f', 'in_normal')
        ]
        self.vao = self.ctx.vertex_array(
                self.prog, vao_content, index_buffer, 4,
            )

    def paintGL(self):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        if self.mesh is None:
            return

        self.aspect_ratio = self.width()/max(1.0, self.height())
        # proj = create_perspective_projection_matrix_from_bounds(-20, 20, 10, -10, 1., 20)
        proj = Matrix44.perspective_projection(60.0, self.aspect_ratio, 5., 1000.0)
        lookat = Matrix44.look_at(
            (0.0, 2.0, 20.0),  # eye
            (0.0, 0.0, 0.0),  # target
            (0.0, 1.0, 0.0),  # up
        )

        self.light.value = (1.0, 1.0, 1.0)
        self.color.value = (1.0, 1.0, 1.0, 0.8)
        self.mvp.write(
            (proj * lookat).astype('f4'))

        self.vao.render()

    def resizeGL(self, width, height):
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)
        return