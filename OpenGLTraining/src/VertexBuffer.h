#pragma once

class VertexBuffer
{
private:
	unsigned int renderer_ID;
public:
	VertexBuffer(const void* data, unsigned int size);
	~VertexBuffer();

	void Bind() const;
	void Unbind() const;

};