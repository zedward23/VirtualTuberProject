#version 330 core

layout(location = 0) in vec4 position;

uniform float time;

void main()
{
	//gl_Position = position;
	gl_Position = vec4(position.x + (sin(time * 5.f + position.y) * .25), 
					   position.y + (cos(time * 5.f + position.x) * .25), 
					   position.z, 1.f);
}