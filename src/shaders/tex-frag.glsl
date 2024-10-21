#version 460 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D tex;

void main()
{
    FragColor = vec4(vec3(texture(tex, TexCoords)), 1);
}
