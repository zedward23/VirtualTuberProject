#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "IndexBuffer.h"
#include "VertexBuffer.h"

//Two String Struct (used to return two strings of two shaders)
struct ShaderProgramSource
{
    std::string vertSrc;
    std::string fragSrc;
};

//Parses an formated filepath that denotes where vertex and fragment 
//shaders begin and returns both as a struct
static ShaderProgramSource ParseShader(const std::string& filepath) {
    std::ifstream stream(filepath);

    enum class ShaderType
    {
        NONE = -1,
        VERTEX = 0,
        FRAGMENT = 1
    };

    std::string line;
    std::stringstream ss[2];
    ShaderType type = ShaderType::NONE;

    while (getline(stream, line))
    {
        if (line.find("#shader") != std::string::npos) 
        {
            if (line.find("vertex") != std::string::npos)
            {
                type = ShaderType::VERTEX;
            } 
            else if (line.find("fragment") != std::string::npos)
            {
                type = ShaderType::FRAGMENT;
            }
        }
        else {
            ss[(int)type] << line << '\n';
        }
    }

    return { ss[0].str(), ss[1].str() };
}

//Parses an arbitrarily formated filepath and returns it as a string
static std::string ParseShaderSrc(const std::string& filepath) {
    std::ifstream stream(filepath);

    std::string line;
    std::stringstream ss;

    while (getline(stream, line))
    {
        ss << line << '\n';
    }

    return ss.str();
}

//Compiles shader text depending on shader type
static unsigned int CompileShader(unsigned int type, const std::string& source) 
{
    std::cout << "Ran at all" << std::endl;
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE) {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);

        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "Failed to compile "
            << (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
            << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(id);
        return 0;
    }

    return id;
}

//Sends shader info to GPU
static unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader) {
    unsigned int program = glCreateProgram();
    unsigned int vertShader = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fragShader = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);
    
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return program;
}



//Main
int main(void)
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    //Set up GLEW
    if (glewInit() != GLEW_OK)
        std::cout << "Not GLEW_OK" << std::endl;

    //std::cout << glGetString(GL_VERSION) << std::endl;

    float positions[] = {
       -0.5f, -0.5f,
        0.5f, -0.5f,
        0.5f,  0.5f,
       -0.5f,  0.5f
    };

    unsigned int indices[] = {
        0,1,
        1,2,
        2,3,
        3,0
    };


    //Vertex Buffers
    unsigned int buffer;

    VertexBuffer vb = VertexBuffer(positions, sizeof(positions));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    //Index Buffers
    IndexBuffer ib = IndexBuffer(indices, sizeof(indices));

    //Bind VAO and VBO to 0 so we don't accidentally modify them
    //glBindBuffer(GL_ARRAY_BUFFER, 0);
    //glBindVertexArray(0);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


    //ShaderProgramSource source = ParseShader("res/shaders/Basic.shader");
    unsigned int shader = CreateShader(ParseShaderSrc("res/shaders/mainVert.shader"), ParseShaderSrc("res/shaders/mainFrag.shader"));
    glUseProgram(shader);

    int location = glGetUniformLocation(shader, "u_Color");
    int time = glGetUniformLocation(shader, "time");

    std::cout << "Timer Begin" << std::endl;
    /* Loop until the user closes the window */

    float counter = 0.f;

    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader);

        counter += .01f;

        float red = (1.f + sinf(counter)) / 2.f;
        
        glUniform4f(location, 0.f, 1.f, 0.f, 1.0f);
        glUniform1f(time, counter);
        //glDrawArrays(GL_TRIANGLES, 0, 6);
        glDrawElements(GL_LINES, 8, GL_UNSIGNED_INT, nullptr);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }
    
    glDeleteProgram(shader);
    glfwTerminate();
    return 0;
}