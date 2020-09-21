import glutils  # Common OpenGL utilities,see glutils.py
import sys, random, math
import OpenGL
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy
import numpy as np
import glfw

strVS = """
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoords;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoords;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;

void main()
{
    gl_Position = uPMatrix *  uMVMatrix * vec4(position, 1.0f);
    FragPos = vec3(uMVMatrix * vec4(position, 1.0f));
    Normal = mat3(transpose(inverse(uMVMatrix))) * normal;  
    TexCoords = texCoords;
} 
"""

strFS = """
#version 330 core
struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float     shininess;
};  

struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;  
in vec3 Normal;  
in vec2 TexCoords;

out vec4 color;

uniform vec3 viewPos;
uniform Material material;
uniform Light light;

void main()
{
    // Ambient
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords));

    // Diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, TexCoords));  

    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular, TexCoords));

    color = vec4(ambient + diffuse + specular, 1.0f);  
} 
"""


class FirstCube:
    def __init__(self, side):
        self.side = side

        # load shaders
        self.program = glutils.loadShaders(strVS, strFS)
        glUseProgram(self.program)
        # attributes
        self.vertIndex = glGetAttribLocation(self.program, b"position")
        self.norIndex = glGetAttribLocation(self.program, b"normal")
        self.texIndex = glGetAttribLocation(self.program, b"texCoords")

        lightPos = [1.2, 1.0, 2.0]
        s = side / 2.0
        cube_vertices = [
            -s, -s, -s,
            s, -s, -s,
            s, s, -s,
            s, s, -s,
            -s, s, -s,
            -s, -s, -s,

            -s, -s, s,
            s, -s, s,
            s, s, s,
            s, s, s,
            -s, s, s,
            -s, -s, s,

            -s, s, s,
            -s, s, -s,
            -s, -s, -s,
            -s, -s, -s,
            -s, -s, s,
            -s, s, s,

            s, s, s,
            s, s, -s,
            s, -s, -s,
            s, -s, -s,
            s, -s, s,
            s, s, s,

            -s, -s, -s,
            s, -s, -s,
            s, -s, s,
            s, -s, s,
            -s, -s, s,
            -s, -s, -s,

            -s, s, -s,
            s, s, -s,
            s, s, s,
            s, s, s,
            -s, s, s,
            -s, s, -s
        ]
        # Normals
        t = 1.0
        cube_normals = [
            0, 0, -t, 0, 0, -t, 0, 0, -t, 0, 0, -t, 0, 0, -t, 0, 0, -t,
            0, 0, t, 0, 0, t, 0, 0, t, 0, 0, t, 0, 0, t, 0, 0, t,
            -t, 0, 0, -t, 0, 0, -t, 0, 0, -t, 0, 0, -t, 0, 0, -t, 0, 0,
            t, 0, 0, t, 0, 0, t, 0, 0, t, 0, 0, t, 0, 0, t, 0, 0,
            0, -t, 0, 0, -t, 0, 0, -t, 0, 0, -t, 0, 0, -t, 0, 0, -t, 0,
            0, t, 0, 0, t, 0, 0, t, 0, 0, t, 0, 0, t, 0, 0, t, 0
        ]
        # texture coords
        quadT = [
            0, 0, t, 0, t, t, t, t, 0, t, 0, 0,
            0, 0, t, 0, t, t, t, t, 0, t, 0, 0,
            t, 0, t, t, 0, t, 0, t, 0, 0, t, 0,
            t, 0, t, t, 0, t, 0, t, 0, 0, t, 0,
            0, t, t, t, t, 0, t, 0, 0, 0, 0, t,
            0, t, t, t, t, 0, t, 0, 0, 0, 0, t
        ]
        # set up vertex array object (VAO)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # set up VBOs
        vertexData = numpy.array(cube_vertices, numpy.float32)
        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(vertexData), vertexData, GL_STATIC_DRAW)

        noData = numpy.array(cube_normals, numpy.float32)
        self.normalsBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normalsBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(noData), noData, GL_STATIC_DRAW)

        tcData = numpy.array(quadT, numpy.float32)
        self.tcBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.tcBuffer)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(tcData), tcData, GL_STATIC_DRAW)
        # enable arrays
        glEnableVertexAttribArray(self.vertIndex)
        glEnableVertexAttribArray(self.norIndex)
        glEnableVertexAttribArray(self.texIndex)
        # Position attribute
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glVertexAttribPointer(self.vertIndex, 3, GL_FLOAT, GL_FALSE, 0, None)
        # Normals attribute
        glBindBuffer(GL_ARRAY_BUFFER, self.normalsBuffer)
        glVertexAttribPointer(self.norIndex, 3, GL_FLOAT, GL_FALSE, 0, None)
        # TexCoord attribute
        glBindBuffer(GL_ARRAY_BUFFER, self.tcBuffer)
        glVertexAttribPointer(self.texIndex, 2, GL_FLOAT, GL_FALSE, 0, None)

        # unbind VAO
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render(self, pMatrix, mvMatrix, texid, texid2, a, b, c, scale, r):
        self.texid = texid
        self.texid2 = texid2
        # enable texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texid)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.texid2)
        # use shader
        glUseProgram(self.program)

        glUniformMatrix4fv(glGetUniformLocation(self.program, 'uPMatrix'),
                           1, GL_FALSE, pMatrix)
        # set modelview matrix
        glUniformMatrix4fv(glGetUniformLocation(self.program, 'uMVMatrix'),
                           1, GL_FALSE, mvMatrix)

        glUniform1i(glGetUniformLocation(self.program, "material.diffuse"), 0)
        glUniform1i(glGetUniformLocation(self.program, "material.specular"), 1)

        lightPosLoc = glGetUniformLocation(self.program, "light.position")
        glUniform3f(lightPosLoc, 1.2, 1.0, 2.0)
        # Set lights properties
        glUniform3f(glGetUniformLocation(self.program, "light.ambient"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(self.program, "light.diffuse"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(self.program, "light.specular"), 1.0, 1.0, 1.0)
        # Set material properties
        glUniform1f(glGetUniformLocation(self.program, "material.shininess"), 32.0)

        # bind VAO
        glBindVertexArray(self.vao)
        glEnable(GL_DEPTH_TEST)
        # draw
        glDrawArrays(GL_TRIANGLES, 0, 36)
        # unbind VAO
        glBindVertexArray(0)


if __name__ == '__main__':
    import sys
    import glfw
    import OpenGL.GL as gl

    camera = glutils.Camera([0.0, 0.0, 3.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0])


    def on_key(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, 1)


    # Initialize the library
    if not glfw.init():
        sys.exit()

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(300, 300, "draw Light Cube ", None, None)
    if not window:
        glfw.terminate()
        sys.exit()

    # Make the window's context current
    glfw.make_context_current(window)

    # Install a key handler
    glfw.set_key_callback(window, on_key)
    PI = 3.14159265358979323846264
    texid = glutils.loadTexture("container2.png")
    texid2 = glutils.loadTexture("container2_specular.png")
    # Loop until the user closes the window
    a = 0
    firstCube0 = FirstCube(1.0)
    while not glfw.window_should_close(window):
        # Render here
        width, height = glfw.get_framebuffer_size(window)
        ratio = width / float(height)
        gl.glViewport(0, 0, width, height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-ratio, ratio, -1, 1, 1, -1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glClearColor(0.0, 0.0, 4.0, 0.0)
        camera.eye = [5 * math.sin(a * PI / 180.0), 0, 5 * math.cos(a * PI / 180.0)]
        pMatrix = glutils.perspective(100.0, ratio, 0.1, 100.0)
        # modelview matrix
        mvMatrix = glutils.lookAt(camera.eye, camera.center, camera.up)
        glBindTexture(GL_TEXTURE_2D, texid)
        i = a
        firstCube0.render(pMatrix, mvMatrix, texid, texid2, 1.2, 1.0, 2.0, 0.5, i)
        a = a + 1
        if a > 360:
            a = 0
            # Swap front and back buffers
        glfw.swap_buffers(window)
        # Poll for and process events
        glfw.poll_events()

    glfw.terminate()