#pragma once

class IndexBuffer 
{
private:
	unsigned int renderer_ID;
	unsigned int m_count;
public:
	IndexBuffer(const unsigned int* data, unsigned int count);
	~IndexBuffer();

	void Bind() const;
	void Unbind() const;

	unsigned int getCount();
};