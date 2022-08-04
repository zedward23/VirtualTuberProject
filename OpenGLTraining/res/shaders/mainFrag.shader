#version 330 core

layout(location = 0) out vec4 color;

uniform vec4 u_Color;
uniform float time;


void main()
{
	color = u_Color;

	//color = vec4(sin(time/50.f)*1.f, 0.f, 0.f, 0.f);
	//color = vec4(1.f, 0.f, 0.f, 0.f);
}