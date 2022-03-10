using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Shapes;


namespace MeshDeformCanvas
{
    class Vec2
    {
        
        public float x { get; set; }
        public float y { get; set; }

        public Vec2(float a, float b)
        {
            x = a;
            y = b;
        }

        public Vec2 plus(Vec2 b)
        {
            return new Vec2(x + b.x, y + b.y);
        }

        public Vec2 minus(Vec2 b)
        {
            return new Vec2(x - b.x, y - b.y);
        }

        public Vec2 tiems(float b)
        {
            return new Vec2(x * b, y * b);
        }
    }

    class Mesh
    {
        public List<Vertex> vertices;
        public List<Edge> edges;

        public Mesh()
        {
            vertices = new List<Vertex>();
            edges = new List<Edge>();
        }
    }
}
